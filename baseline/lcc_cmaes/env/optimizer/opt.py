"""
CMA-ES Cooperative Coevolution Optimization Environment / CMA-ES 协同进化优化环境

This module implements a Gym environment for large-scale optimization using the
Cooperative Coevolution CMA-ES (CC-CMAES) algorithm with adaptive subspace
decomposition strategy selection (MiVD, MaVD, RaVD).

本模块实现了基于协同进化 CMA-ES (CC-CMAES) 算法的大规模优化 Gym 环境，
支持自适应子空间分解策略选择 (MiVD, MaVD, RaVD)。

Environment State Vector / 环境状态向量:
    S_GO (Global Features, 12 dims) / 全局特征:
        s1-s3: Centroid statistics (mean, max, min) / 中心统计量
        s4-s6: Correlation matrix statistics (max, min, mean) / 相关矩阵统计量
        s7-s9: Global best statistics (max, min, mean) / 全局最优统计量
        s10: Fitness boosting ratio / 适应度提升比例
        s11: Remaining FEs ratio / 剩余 FEs 比例
        s12: Sigma (step size) / 步长

    S_SD (Subgroup Features, 4*m dims) / 子群特征:
        - Correlation coefficient for each subgroup / 每个子群的相关系数
        - Delta (first-last difference) / 首尾差异
        - Variance / 方差
        - Max Euclidean distance (dmax) / 最大欧氏距离

    S_AH (Action History, 6 dims) / 动作历史:
        - Average reward for each action (3 actions) / 每个动作的平均奖励
        - Average distance improvement for each action (3 actions) / 每个动作的平均距离改进

Action Space / 动作空间:
    - 0: MiVD (Minimum Variance Decomposition) / 最小方差分解
    - 1: MaVD (Maximum Variance Decomposition) / 最大方差分解
    - 2: RaVD (Random Variance Decomposition) / 随机方差分解

Author: WCC Project
Date: 2026-03-15
"""

# =============================================================================
# Imports / 导入
# =============================================================================
# Standard library / 标准库

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import cma
import math
import itertools
from scipy.spatial.distance import pdist

# Project imports / 项目导入
from utils.options import get_options
from utils.utils import set_random_seed

# Optimizer modules / 优化器模块
from .cc_cmaes.mivd import mivd_groups
from .cc_cmaes.mavd import mavd_groups
from .cc_cmaes.ravd import ravd_groups

# Benchmark modules / 基准测试模块
# from benchmark.cec2013lsgo.cec2013 import Benchmark
from benchmark.ne.brax.brax_benchmarks import Benchmark

# =============================================================================
# Configuration / 配置
# =============================================================================
# Load options / 加载配置
opts = get_options()
set_random_seed(opts.seed)

# Configure logging / 配置日志
logger = logging.getLogger(__name__)

# =============================================================================
# Constants / 常量定义
# =============================================================================
# Initial population size / 初始化种群大小
INIT_POPULATION_SIZE = 500

# Fitness threshold for convergence / 收敛适应度阈值
FITNESS_THRESHOLD = 1e-8

# Default reward scaling factor / 默认奖励缩放因子
DEFAULT_REWARD_SCALE = 1.0

# Action name mapping / 动作名称映射
ACTION_NAMES = {
    0: 'MiVD',
    1: 'MaVD',
    2: 'RaVD'
}

# Grouping function mapping / 分组函数映射
_GROUPING_FUNCTIONS = {
    'MiVD': mivd_groups,
    'MaVD': mavd_groups,
    'RaVD': ravd_groups
}


# =============================================================================
# State Calculator / 状态计算器
# =============================================================================
class StateCalculator:
    """
    State vector calculator for CMA-ES CC optimization environment.
    CMA-ES 协同进化优化环境的状态向量计算器。

    Computes the environment state vector consisting of / 计算环境状态向量，包括:
        - S_GO (12 dims): Global features / 全局特征
        - S_SD (4*m dims): Subgroup features / 子群特征
        - S_AH (6 dims): Action history features / 动作历史特征
    """

    def __init__(
        self,
        dimension: int,
        num_subgroups: int,
        lower_bound: float,
        upper_bound: float,
        max_fes: int
    ):
        """
        Initialize state calculator / 初始化状态计算器

        Args:
            dimension: Problem dimension / 问题维度
            num_subgroups: Number of subgroups / 子群数量
            lower_bound: Lower bound / 下界
            upper_bound: Upper bound / 上界
            max_fes: Maximum function evaluations / 最大函数评估次数
        """
        self.D = dimension
        self.m = num_subgroups
        self.lb = lower_bound
        self.ub = upper_bound
        self.max_fes = max_fes
        self.search_scope_half = (self.ub - self.lb) * 0.5

    def compute_global_features(
        self,
        global_Xw: npt.NDArray[np.float64],
        global_C: npt.NDArray[np.float64],
        best_solution: npt.NDArray[np.float64],
        fes: int,
        sigma: float
    ) -> List[float]:
        """
        Compute global feature state vector (S_GO, 12 dims).
        计算全局特征状态向量 (S_GO, 12 维).

        Args:
            global_Xw: Global mean vector / 全局均值向量
            global_C: Global covariance matrix / 全局协方差矩阵
            best_solution: Current best solution / 当前最优解
            fes: Accumulated function evaluations / 累计函数评估次数
            sigma: Current step size / 当前步长

        Returns:
            Global state vector (12 dims) / 全局状态向量 (12 维)
        """
        # Centroid statistics (s1-s3) / 中心统计量
        Xw_mean = np.mean(global_Xw)
        Xw_max = np.max(global_Xw)
        Xw_min = np.min(global_Xw)

        # Correlation matrix statistics (s4-s6) / 相关矩阵统计量
        correlation_matrix = np.corrcoef(global_C)
        correlation_matrix_max = np.max(correlation_matrix)
        correlation_matrix_min = np.min(correlation_matrix)
        correlation_matrix_mean = np.mean(correlation_matrix)

        # Global best statistics (s7-s9) / 全局最优统计量
        g_best_max = np.max(best_solution)
        g_best_min = np.min(best_solution)
        g_best_mean = np.mean(best_solution)

        # Fitness boosting ratio (s10) / 适应度提升比例
        g_best_fitness_boosting_ratio = 1.0

        # Remaining FEs ratio (s11) / 剩余 FEs 比例
        fes_remaining = self.max_fes - fes
        fes_ratio = fes_remaining / self.max_fes

        # Sigma (s12) / 步长
        sigma_normalized = sigma / self.search_scope_half

        return [
            Xw_mean / self.search_scope_half,
            Xw_max / self.search_scope_half,
            Xw_min / self.search_scope_half,
            correlation_matrix_max,
            correlation_matrix_min,
            correlation_matrix_mean,
            g_best_max / self.search_scope_half,
            g_best_min / self.search_scope_half,
            g_best_mean / self.search_scope_half,
            g_best_fitness_boosting_ratio,
            fes_ratio,
            sigma_normalized
        ]

    def compute_subgroup_features(
        self,
        correlation_coefficient: npt.NDArray[np.float64],
        delta: npt.NDArray[np.float64],
        variance: npt.NDArray[np.float64],
        dmax: npt.NDArray[np.float64]
    ) -> List[float]:
        """
        Compute subgroup feature state vector (S_SD, 4*m dims).
        计算子群特征状态向量 (S_SD, 4*m 维).

        Args:
            correlation_coefficient: Correlation coefficient for each subgroup / 每个子群的相关系数
            delta: First-last difference for each subgroup / 每个子群的首尾差异
            variance: Variance for each subgroup / 每个子群的方差
            dmax: Max distance for each subgroup / 每个子群的最大距离

        Returns:
            Subgroup state vector (4*m dims) / 子群状态向量 (4*m 维)
        """
        return list(itertools.chain(
            correlation_coefficient,
            delta,
            variance,
            dmax
        ))

    def compute_action_features(
        self,
        action_num_count: npt.NDArray[np.float64],
        action_reward_sum: npt.NDArray[np.float64],
        action_best_sum: npt.NDArray[np.float64],
        search_scope_half: float
    ) -> List[float]:
        """
        Compute action history feature state vector (S_AH, 6 dims).
        计算动作历史特征状态向量 (S_AH, 6 维).

        Args:
            action_num_count: Number of times each action was selected / 每个动作被选择的次数
            action_reward_sum: Cumulative reward for each action / 每个动作的累积奖励
            action_best_sum: Cumulative distance improvement for each action / 每个动作的累积距离改进
            search_scope_half: Half of search scope / 搜索范围的一半

        Returns:
            Action history state vector (6 dims) / 动作历史状态向量 (6 维)
        """
        action_history = []
        for i in range(3):
            avg_reward = (
                action_reward_sum[i] / action_num_count[i]
                if action_num_count[i] != 0 else 0.0
            )
            avg_distance = (
                action_best_sum[i] / (2 * search_scope_half * action_num_count[i])
                if action_num_count[i] != 0 else 0.0
            )
            action_history.extend([avg_reward, avg_distance])

        return action_history

    def compute(
        self,
        global_Xw: npt.NDArray[np.float64],
        global_C: npt.NDArray[np.float64],
        best_solution: npt.NDArray[np.float64],
        fes: int,
        sigma: float,
        correlation_coefficient: npt.NDArray[np.float64],
        delta: npt.NDArray[np.float64],
        variance: npt.NDArray[np.float64],
        dmax: npt.NDArray[np.float64],
        action_num_count: npt.NDArray[np.float64],
        action_reward_sum: npt.NDArray[np.float64],
        action_best_sum: npt.NDArray[np.float64]
    ) -> List[float]:
        """
        Compute complete state vector.
        计算完整的状态向量。

        Args:
            global_Xw: Global mean vector / 全局均值向量
            global_C: Global covariance matrix / 全局协方差矩阵
            best_solution: Current best solution / 当前最优解
            fes: Accumulated function evaluations / 累计函数评估次数
            sigma: Current step size / 当前步长
            correlation_coefficient: Subgroup correlation coefficients / 子群相关系数
            delta: Subgroup deltas / 子群首尾差异
            variance: Subgroup variances / 子群方差
            dmax: Subgroup max distances / 子群最大距离
            action_num_count: Action selection counts / 动作选择计数
            action_reward_sum: Action reward sums / 动作奖励总和
            action_best_sum: Action distance sums / 动作距离总和

        Returns:
            Complete state vector / 完整状态向量
        """
        # Global features (S_GO, 12 dims) / 全局特征
        global_state = self.compute_global_features(
            global_Xw, global_C, best_solution, fes, sigma
        )

        # Subgroup features (S_SD, 4*m dims) / 子群特征
        subgroup_state = self.compute_subgroup_features(
            correlation_coefficient, delta, variance, dmax
        )

        # Action history features (S_AH, 6 dims) / 动作历史特征
        action_state = self.compute_action_features(
            action_num_count, action_reward_sum, action_best_sum, self.search_scope_half
        )

        # Combine all features / 组合所有特征
        state = []
        state.extend(global_state)
        state.extend(subgroup_state)
        state.extend(action_state)

        return state


# =============================================================================
# Reward Calculator / 奖励计算器
# =============================================================================
class RewardCalculator:
    """
    Reward calculator for CMA-ES CC optimization environment.
    CMA-ES 协同进化优化环境的奖励计算器。

    Computes reward based on fitness improvement ratio.
    基于适应度改善比例计算奖励。
    """

    def __init__(self, reward_scale: float = DEFAULT_REWARD_SCALE):
        """
        Initialize reward calculator / 初始化奖励计算器

        Args:
            reward_scale: Scaling factor for reward / 奖励缩放因子
        """
        self.reward_scale = reward_scale
        self.origin_fitness: Optional[float] = None

    def reset(self, initial_fitness: float) -> None:
        """
        Reset with initial fitness value.
        使用初始适应度值重置。

        Args:
            initial_fitness: Initial fitness value / 初始适应度值
        """
        self.origin_fitness = initial_fitness

    def compute(
        self,
        previous_fitness: float,
        current_fitness: float
    ) -> float:
        """
        Compute reward based on fitness improvement.
        基于适应度改善计算奖励。

        Args:
            previous_fitness: Fitness before action / 动作前的适应度
            current_fitness: Fitness after action / 动作后的适应度

        Returns:
            Computed reward / 计算的奖励

        Raises:
            ValueError: When origin_fitness is not set / 当初始适应度未设置时
        """
        if self.origin_fitness is None:
            raise ValueError(
                "RewardCalculator not initialized. Call reset() first. / "
                "RewardCalculator 未初始化。请先调用 reset()。"
            )

        # Avoid division by zero / 避免除零
        if abs(self.origin_fitness) < 1e-12:
            improvement = previous_fitness - current_fitness
        else:
            improvement = (previous_fitness - current_fitness) / self.origin_fitness

        return float(improvement * self.reward_scale)


# =============================================================================
# CMA-ES CC Optimization Environment / CMA-ES 协同进化优化环境
# =============================================================================
class CMAESCCEnv(gym.Env):
    """
    CMA-ES Cooperative Coevolution Optimization Environment / CMA-ES 协同进化优化环境

    A Gym environment for large-scale black-box optimization using Cooperative
    Coevolution CMA-ES with adaptive subspace decomposition selection.

    用于大规模黑盒优化的 Gym 环境，使用协同进化 CMA-ES 和自适应子空间分解选择。

    The environment maintains a global solution and covariance matrix, and
    dynamically selects between three decomposition strategies (MiVD, MaVD, RaVD)
    to optimize subspaces independently.

    环境维护全局解和协方差矩阵，动态选择三种分解策略 (MiVD, MaVD, RaVD)
    独立优化子空间。

    Action Space / 动作空间:
        - 0: MiVD (Minimum Variance Decomposition) / 最小方差分解
        - 1: MaVD (Maximum Variance Decomposition) / 最大方差分解
        - 2: RaVD (Random Variance Decomposition) / 随机方差分解

    Observation Space / 观察空间:
        - Dependent on opts.feature_num / 取决于 opts.feature_num

    Attributes:
        question (int): Problem ID / 问题 ID
        D (int): Problem dimension / 问题维度
        lb (float): Lower bound / 下界
        ub (float): Upper bound / 上界

    Example:
        >>> env = CMAESCCEnv(question=1)
        >>> state = env.reset()
        >>> action = env.action_space.sample()
        >>> next_state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        question: int = None,
        fitness_function: callable = None,
        max_fes: int = None,
        problem_dimension: int = None,
        lower_bound: float = None,
        upper_bound: float = None,
    ):
        """
        Initialize CMA-ES CC optimization environment / 初始化 CMA-ES 协同进化优化环境

        Args:
            question: Problem ID (1-15 for CEC2013). Required if fitness_function is not provided. / 问题 ID
            fitness_function: Custom fitness function. If provided, question is not used for getting the function. / 自定义适应度函数
            max_fes: Maximum function evaluations (overrides opts.max_fes if provided) / 最大函数评估次数（如果提供则覆盖 opts.max_fes）
        """
        # -------------------------------------------------------------------------
        # Problem Setup / 问题设置
        # -------------------------------------------------------------------------
        if fitness_function is not None:
            # Use custom fitness function / 使用自定义适应度函数
            self.fun = fitness_function
            self.question = question if question is not None else 1
            self.bench = Benchmark()

            # Prefer externally provided problem metadata for custom fitness.
            # 当使用自定义目标函数时，优先使用外部传入的维度和边界信息。
            if (
                problem_dimension is not None
                and lower_bound is not None
                and upper_bound is not None
            ):
                self.info = {
                    'dimension': int(problem_dimension),
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                }
            else:
                # Backward compatibility fallback / 向后兼容回退
                self.info = self.bench.get_info(self.question)
        else:
            # Use benchmark function / 使用 benchmark 函数
            self.question = question
            self.bench = Benchmark()
            self.info = self.bench.get_info(self.question)
            self.fun = self.bench.get_function(self.question)

        # -------------------------------------------------------------------------
        # CMA-ES Parameters / CMA-ES 参数
        # -------------------------------------------------------------------------
        self.D = self.info['dimension']  # Problem dimension / 问题维度
        self.lb = self.info['lower']     # Lower bound / 下界
        self.ub = self.info["upper"]     # Upper bound / 上界
        self.m = opts.m                  # Number of subgroups / 子群数量
        self.sigma = 0.5 * (self.ub - self.lb)  # Step size / 步长
        self.sub_popsize = opts.sub_popsize  # Population per subgroup / 每子群种群大小
        # Use provided max_fes if available, otherwise use opts.max_fes / 使用提供的 max_fes（如果有），否则使用 opts.max_fes
        self.MaxFES = max_fes if max_fes is not None else opts.max_fes
        self.seed = opts.seed            # Random seed / 随机种子
        self.subFEs = opts.subFEs        # FEs per subgroup / 每子群 FEs

        # -------------------------------------------------------------------------
        # Gym Space Configuration / Gym 空间配置
        # -------------------------------------------------------------------------
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(opts.feature_num, 1),
            dtype=np.float32
        )

        # -------------------------------------------------------------------------
        # Initial State / 初始状态
        # -------------------------------------------------------------------------
        self.state: List[float] = []
        self.done = False

        # -------------------------------------------------------------------------
        # Fitness Recording / 适应度记录
        # -------------------------------------------------------------------------
        # Wrap the fitness function to record all evaluations / 包装适应度函数以记录所有评估
        self._original_fun = self.fun
        self._fitness_record: List[float] = []
        self.fun = self._recorded_function

        # -------------------------------------------------------------------------
        # Helper Classes / 辅助类
        # -------------------------------------------------------------------------
        # State calculator for computing state vector / 状态计算器
        self.state_calculator = StateCalculator(
            dimension=self.D,
            num_subgroups=self.m,
            lower_bound=self.lb,
            upper_bound=self.ub,
            max_fes=self.MaxFES
        )

        # Reward calculator for computing optimization reward / 奖励计算器
        self.reward_calculator = RewardCalculator()

        logger.info(f"初始化 CMA-ES CC 环境 / Initialized CMA-ES CC env: question={question}, D={self.D}")

    def _recorded_function(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper for fitness function that records all evaluations.
        记录所有评估的适应度函数包装器。

        Args / 参数:
            x: Input points / 输入点

        Returns / 返回:
            Fitness values / 适应度值
        """
        result = self._original_fun(x)
        self._fitness_record.extend(result.flatten().tolist())
        return result

    def get_fitness_record(self) -> List[float]:
        """
        Get the recorded fitness values.
        获取记录的适应度值。

        Returns / 返回:
            List of recorded fitness values / 记录的适应度值列表
        """
        return self._fitness_record.copy()

    def reset(self) -> List[float]:
        """
        Reset the environment and return initial state.
        重置环境并返回初始状态。

        Performs initial random sampling to find a starting solution,
        then computes the initial state vector.

        执行初始随机采样以找到起始解，然后计算初始状态向量。

        Returns:
            Initial state vector / 初始状态向量
        """
        # -------------------------------------------------------------------------
        # Initialize CMA-ES / 初始化 CMA-ES
        # -------------------------------------------------------------------------
        # Sample random population / 采样随机种群
        random_vectors = np.random.uniform(
            self.lb,
            self.ub,
            size=(INIT_POPULATION_SIZE, self.D)
        )

        # Initialize global covariance matrix / 初始化全局协方差矩阵
        self.global_C = np.eye(self.D)
        self.global_Xw = np.mean(random_vectors, axis=0)

        # Evaluate fitness / 评估适应度
        function_values = self.fun(random_vectors)

        # Find best solution / 找到最优解
        min_index = np.argmin(function_values)

        # Initialize global best / 初始化全局最优
        self.best = random_vectors[min_index]
        self.best_his = self.best.copy()

        # Initialize global best fitness / 初始化全局最优适应度
        self.g_best_fitness = function_values[min_index]
        self.origin_g_best_fitness = self.g_best_fitness
        self.g_best_fitness_his = self.g_best_fitness

        # Initialize other variables / 初始化其他变量
        self.fes = 0
        self.norm_factor = np.sqrt(((self.ub - self.lb) * 0.5)**2 * self.D)

        # Reset reward calculator / 重置奖励计算器
        self.reward_calculator.reset(self.g_best_fitness)

        logger.info(f"初始适应度 / Initial fitness: {self.g_best_fitness:.2e}")

        # -------------------------------------------------------------------------
        # Initialize Subgroup Features / 初始化子群特征
        # -------------------------------------------------------------------------
        self.correlation_coefficient = np.zeros(self.m)
        self.delta = np.zeros(self.m)
        self.variance = np.zeros(self.m)
        self.dmax = np.zeros(self.m)

        # -------------------------------------------------------------------------
        # Initialize Action History / 初始化动作历史
        # -------------------------------------------------------------------------
        self.action_num_count = np.zeros(3)
        self.action_reward_sum = np.zeros(3)
        self.action_best_sum = np.zeros(3)

        # -------------------------------------------------------------------------
        # Compute Initial State / 计算初始状态
        # -------------------------------------------------------------------------
        self.state = self.state_calculator.compute(
            global_Xw=self.global_Xw,
            global_C=self.global_C,
            best_solution=self.best,
            fes=self.fes,
            sigma=self.sigma,
            correlation_coefficient=self.correlation_coefficient,
            delta=self.delta,
            variance=self.variance,
            dmax=self.dmax,
            action_num_count=self.action_num_count,
            action_reward_sum=self.action_reward_sum,
            action_best_sum=self.action_best_sum
        )

        self.done = False
        self.reward = 0.0

        # gymnasium reset 返回 (obs, info) 元组
        info = {
            "question": self.question,
            "initial_fitness": float(self.g_best_fitness)
        }
        return self.state, info

    def _combine_solution(
        self,
        subspace_solution: npt.NDArray[np.float64],
        background_solution: npt.NDArray[np.float64],
        subspace_indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """
        Combine subspace solution with full solution.
        组合子空间解与完整解。

        Args:
            subspace_solution: Subspace solution vector / 子空间解向量
            background_solution: Background full solution / 背景完整解
            subspace_indices: Subspace dimension indices / 子空间维度索引

        Returns:
            Combined solution / 组合后的解
        """
        if subspace_indices is None:
            return subspace_solution
        combination = np.tile(background_solution, (len(subspace_solution), 1))
        combination[:, subspace_indices] = subspace_solution
        return combination

    def step(
        self,
        action: Union[int, npt.NDArray[np.int64]]
    ) -> Tuple[List[float], float, bool, Dict[str, any]]:
        """
        Execute one environment step with the given action.
        执行给定动作的环境步骤。

        Args:
            action: Action (0=MiVD, 1=MaVD, 2=RaVD) / 动作

        Returns:
            tuple: (state, reward, done, info) / (状态, 奖励, 完成, 信息)
                - state: Next state vector / 下一个状态向量
                - reward: Fitness improvement ratio / 适应度改善比例
                - done: Termination flag / 终止标志
                - info: Dictionary with additional info / 附加信息字典
        """
        # -------------------------------------------------------------------------
        # Parse Action / 解析动作
        # -------------------------------------------------------------------------
        if isinstance(action, np.ndarray):
            action_num = int(action.item())
        else:
            action_num = int(action)

        action_name = ACTION_NAMES.get(action_num, 'MiVD')

        logger.debug(f"执行动作 / Executing action: {action_name} (action_num={action_num})")

        # Store previous fitness / 存储之前的适应度
        self.g_best_fitness_his = self.g_best_fitness
        self.best_his = self.best.copy()

        # -------------------------------------------------------------------------
        # Get Subgroups / 获取子群
        # -------------------------------------------------------------------------
        grouping_func = _GROUPING_FUNCTIONS[action_name]
        groups = grouping_func(self.D, math.ceil(self.D / self.m), self.global_C)

        # -------------------------------------------------------------------------
        # Optimize Each Subgroup / 优化每个子群
        # -------------------------------------------------------------------------
        for i, dims in enumerate(groups):
            # Record keeping / 记录保存
            offspring_record: List[npt.NDArray[np.float64]] = []
            offspring_record_var: List[float] = []

            # Extract subspace parameters / 提取子空间参数
            sub_centroid = self.global_Xw[dims]
            sub_C = self.global_C[dims][:, dims]

            # Suppress CMA-ES warnings / 抑制 CMA-ES 警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cma.evolution_strategy._CMASolutionDict = cma.evolution_strategy._CMASolutionDict_empty

                # Configure CMA-ES options / 配置 CMA-ES 选项
                cma_options = {
                    'popsize': self.sub_popsize,
                    'bounds': [self.lb, self.ub],
                    'seed': self.seed,
                    'maxfevals': self.subFEs,
                    'verbose': -9 if not opts.output_init_cma_info else 0
                }

                # Initialize subspace CMA-ES / 初始化子空间 CMA-ES
                sub_es = cma.CMAEvolutionStrategy(sub_centroid, self.sigma, cma_options)

            # Set initial covariance matrix / 设置初始协方差矩阵
            sub_es.sm.C = sub_C.copy()
            sub_es.sm.update_now(-1)
            sub_es._updateBDfromSM()

            # Define subspace objective function / 定义子空间目标函数
            def subspace_objective(x_batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                combined = self._combine_solution(x_batch, self.best, dims)
                # cma-es expects a 1D list/array of scalar fitness values.
                # Brax fitness on a single sample may return shape (1,), so we scalarize here.
                values = [float(np.asarray(self.fun(x)).reshape(-1)[0]) for x in combined]
                return np.asarray(values, dtype=np.float64)

            # Run CMA-ES optimization / 运行 CMA-ES 优化
            while not sub_es.stop():
                X_batch = sub_es.ask()

                # Record offspring / 记录后代
                offspring_record.append(X_batch.copy())
                offspring_record_var.append(np.var(X_batch))

                # Evaluate and update / 评估并更新
                sub_es.tell(X_batch, subspace_objective(X_batch))

                # Update global best if improved / 如果改善则更新全局最优
                sub_best = sub_es.result[0]
                sub_best_fitness = sub_es.result[1]

                if sub_best_fitness < self.g_best_fitness:
                    self.g_best_fitness = sub_best_fitness
                    self.best[dims] = sub_best

                # Update global parameters / 更新全局参数
                self.global_Xw[dims] = sub_es.result[5]
                self.global_C[dims][:, dims] = sub_es.sm.C.copy()

            # Update sigma / 更新步长
            self.sigma = sub_es.sigma

            # Convert records to arrays / 转换记录为数组
            offspring_record = np.array(offspring_record)
            offspring_record_var = np.array(offspring_record_var)

            # -------------------------------------------------------------------------
            # Compute Subgroup Features / 计算子群特征
            # -------------------------------------------------------------------------
            # Correlation coefficient / 相关系数
            sub_correlation_matrix = np.corrcoef(sub_es.sm.C)
            upper_triangle_indices = np.triu_indices_from(sub_correlation_matrix, k=1)
            self.correlation_coefficient[i] = np.mean(
                sub_correlation_matrix[upper_triangle_indices]
            )

            # Delta (first-last difference) / 首尾差异
            if len(offspring_record) >= 2:
                delta = np.mean(
                    np.sum(offspring_record[-1] - offspring_record[0], axis=0) / self.sub_popsize
                ) / self.state_calculator.search_scope_half
            else:
                delta = 0.0
            self.delta[i] = delta

            # Variance / 方差
            variance = np.mean(offspring_record_var) / np.var([self.lb, self.ub])
            self.variance[i] = variance

            # Max Euclidean distance / 最大欧氏距离
            flattened_offspring = offspring_record.reshape(-1, offspring_record.shape[-1])
            distances = pdist(flattened_offspring, metric='euclidean')
            max_distance = np.max(distances) / self.norm_factor if len(distances) > 0 else 0.0
            self.dmax[i] = max_distance

        # -------------------------------------------------------------------------
        # Update FEs / 更新 FEs
        # -------------------------------------------------------------------------
        self.fes = self.fes + self.subFEs * self.m

        # -------------------------------------------------------------------------
        # Compute Reward / 计算奖励
        # -------------------------------------------------------------------------
        self.reward = self.reward_calculator.compute(
            previous_fitness=self.g_best_fitness_his,
            current_fitness=self.g_best_fitness
        )

        # Update action history / 更新动作历史
        self.action_num_count[action_num] += 1
        self.action_reward_sum[action_num] += self.reward
        self.action_best_sum[action_num] += np.linalg.norm(self.best - self.best_his)

        # Clip sigma for stability / 裁剪步长以保持稳定性
        sigma_clipped = np.clip(self.sigma, None, 100)

        # -------------------------------------------------------------------------
        # Compute State / 计算状态
        # -------------------------------------------------------------------------
        self.state = self.state_calculator.compute(
            global_Xw=self.global_Xw,
            global_C=self.global_C,
            best_solution=self.best,
            fes=self.fes,
            sigma=sigma_clipped,
            correlation_coefficient=self.correlation_coefficient,
            delta=self.delta,
            variance=self.variance,
            dmax=self.dmax,
            action_num_count=self.action_num_count,
            action_reward_sum=self.action_reward_sum,
            action_best_sum=self.action_best_sum
        )

        # -------------------------------------------------------------------------
        # Check Termination / 检查终止条件
        # -------------------------------------------------------------------------
        fes_remaining = self.MaxFES - self.fes
        if fes_remaining < self.subFEs * self.m:
            self.done = True
            import sys
            print(f"[DONE] Budget exhausted: fes={self.fes}, MaxFES={self.MaxFES}, fes_remaining={fes_remaining}", file=sys.stderr, flush=True)
        elif self.g_best_fitness <= FITNESS_THRESHOLD:
            self.done = True
            logger.info(f"收敛到最优 / Converged to optimum: fitness={self.g_best_fitness:.2e}")
        else:
            self.done = False

        # -------------------------------------------------------------------------
        # Build Info / 构建信息
        # -------------------------------------------------------------------------
        info = {
            "question": self.question,
            "Fes": self.fes,
            "gbest_val": float(self.g_best_fitness),
            "action": action_num,
            "fitness_record": self._fitness_record.copy()  # Include fitness record / 包含适应度记录
        }

        logger.debug(
            f"步骤完成 / Step complete: action={action_name}, "
            f"reward={self.reward:.4f}, fitness={self.g_best_fitness:.2e}"
        )

        # gymnasium step 返回 (obs, reward, terminated, truncated, info) 5 元组
        terminated = self.done  # 任务完成（达到最优或预算耗尽）
        truncated = False       # 未被截断
        return self.state, self.reward, terminated, truncated, info


# =============================================================================
# Backward Compatibility / 向后兼容
# =============================================================================
# Alias for backward compatibility / 向后兼容别名
cmaes = CMAESCCEnv
