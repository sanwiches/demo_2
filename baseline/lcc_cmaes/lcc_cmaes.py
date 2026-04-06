"""
LCC-CMAES: Learning-based Cooperative Coevolution CMA-ES
基于学习的协同进化 CMA-ES

A simplified optimizer wrapper for testing the trained LCC-CMAES algorithm.
训练好的 LCC-CMAES 算法的简洁优化器封装。

Paper / 论文:
    "Advancing CMA-ES with Learning-Based Cooperative Coevolution
    for Scalable Optimization", GECCO 2026

Usage / 使用方法:
    >>> from baseline.lcc_cmaes import LCC_CMAES
    >>> problem = {
    ...     'fitness_function': lambda x: sum(x**2),
    ...     'ndim_problem': 1000,
    ...     'upper_boundary': 100,
    ...     'lower_boundary': -100
    ... }
    >>> options = {
    ...     'model_path': 'path/to/model.pt',
    ...     'max_function_evaluations': 3000000,
    ...     'seed_rng': 2024
    ... }
    >>> optimizer = LCC_CMAES(problem, options)
    >>> results = optimizer.optimize()
    >>> print(f"Best fitness: {results['best_so_far_y']}")

Author: LCC-CMAES Project
Date: 2026-03-17
"""

# =============================================================================
# Imports / 导入
# =============================================================================
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Add current directory to path / 添加当前目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Local imports from env directory / 从 env 目录本地导入
from .env.agent.inference import InferenceAgent
from .env.optimizer.opt import cmaes
from .env.parallel import venvs


# =============================================================================
# Main Optimizer Class / 主优化器类
# =============================================================================

class LCC_CMAES:
    """
    PPO-guided Cooperative Coevolution CMA-ES Optimizer.
    PPO 引导的协同进化 CMA-ES 优化器。

    This optimizer uses a trained PPO agent to dynamically select decomposition
    strategies (MiVD, MaVD, RaVD) for large-scale optimization problems.
    该优化器使用训练好的 PPO 智能体动态选择大规模优化问题的分解策略。

    Parameters / 参数:
        problem (dict): Problem configuration / 问题配置
            - fitness_function: Objective function to minimize / 要最小化的目标函数
            - ndim_problem: Problem dimension / 问题维度
            - upper_boundary: Upper bound of search space / 搜索空间上界
            - lower_boundary: Lower bound of search space / 搜索空间下界
        options (dict): Optimizer options / 优化器选项
            - model_path (str): Path to trained PPO model / 训练好的 PPO 模型路径
            - max_function_evaluations (int): Max FEs budget / 最大函数评估预算
            - m (int): Number of subgroups / 子组数量 (default: 10)
            - sub_popsize (int): Subgroup population size / 子组种群大小 (default: 10)
            - subFEs (int): FEs per subspace optimization / 每个子空间优化的 FEs (default: 1000)
            - each_question_batch_num (int): Parallel instances / 并行实例数 (default: 4)
            - seed_rng (int): Random seed / 随机种子 (default: 2024)
            - device (str): Device for PPO model / PPO 模型设备 (default: 'cuda')
            - verbose (int): Logging verbosity / 日志详细度 (default: 10)
    """

    def __init__(self, problem: Dict, options: Dict):
        # -------------------------------------------------------------------------
        # Problem Setup / 问题设置
        # -------------------------------------------------------------------------
        self.fitness_function = problem.get('fitness_function')
        self.ndim_problem = problem.get('ndim_problem', 1000)
        self.upper_boundary = problem.get('upper_boundary', 100.0)
        self.lower_boundary = problem.get('lower_boundary', -100.0)

        # -------------------------------------------------------------------------
        # Optimizer Options / 优化器选项
        # -------------------------------------------------------------------------
        self.options = options
        # Use local model path if not provided / 如果未提供则使用本地模型路径
        if options.get('model_path') is None:
            _current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(_current_dir, 'model', 'epoch-9.pt')
        else:
            self.model_path = options['model_path']

        self.max_function_evaluations = int(options.get('max_function_evaluations', 3000000))
        self.m = options.get('m', 10)  # Number of subgroups / 子组数量
        self.sub_popsize = options.get('sub_popsize', 10)  # Subgroup population / 子组种群
        self.subFEs = options.get('subFEs', 50)  # FEs per subspace / 每个子空间的 FEs
        self.each_question_batch_num = options.get('each_question_batch_num', 1)

        # Random seeds / 随机种子
        self.seed_rng = options.get('seed_rng', 2024)
        self.rng = np.random.default_rng(self.seed_rng)

        # Device / 设备
        self.device = options.get('device', 'cuda')
        self.verbose = options.get('verbose', 10)

        # State dimension (can be specified directly, otherwise use default) / 状态维度（可直接指定，否则使用默认值）
        self.state_dim = options.get('state_dim', 58)

        # Problem ID for CEC2013 benchmark / CEC2013 基准测试的问题 ID
        self.problem_id = options.get('problem_id', 1)

        # -------------------------------------------------------------------------
        # Compute derived parameters / 计算派生参数
        # -------------------------------------------------------------------------
        # Number of action selections (ns) / 动作选择次数
        self.ns = int(self.max_function_evaluations // (self.m * self.subFEs))

        # -------------------------------------------------------------------------
        # State variables / 状态变量
        # -------------------------------------------------------------------------
        self.n_function_evaluations = 0
        self.runtime = 0
        self.best_so_far_y = np.inf
        self.best_so_far_x = None
        self.fitness_history = []
        self.fitness_record = []  # Store fitness record (cumulative from environment) / 存储适应度记录（来自环境的累积记录）

        # -------------------------------------------------------------------------
        # Initialize PPO Agent / 初始化 PPO 智能体
        # -------------------------------------------------------------------------
        self._init_ppo_agent()

    def _init_ppo_agent(self):
        """Initialize and load the trained PPO agent. / 初始化并加载训练好的 PPO 智能体。"""
        # Create a minimal options object for PPO / 为 PPO 创建最小配置对象
        class PPOOptions:
            def __init__(self, device, state_dim, action_dim, hidden_dim):
                self.device = device
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.hidden_dim = hidden_dim
                self.use_cuda = (device == 'cuda')
                self.test = True
                self.encoder_head_num = 4
                self.decoder_head_num = 4
                self.critic_head_num = 6
                self.embedding_dim = 16
                self.hidden_dim = 16
                self.n_encode_layers = 1
                self.normalization = 'layer'
                self.v_range = 6.0
                self.feature_num = 58
                self.state = [0.0 for _ in range(58)]

        # Use the state_dim from options (default is 58) / 使用选项中的 state_dim（默认为 58）
        state_dim = self.state_dim

        ppo_opts = PPOOptions(
            device=self.device,
            state_dim=state_dim,
            action_dim=3,  # MiVD, MaVD, RaVD
            hidden_dim=16
        )

        # Initialize agent / 初始化智能体
        self.agent = InferenceAgent(ppo_opts)
        self.agent.load(self.model_path)
        self.agent.eval()

        if self.verbose:
            print(f"[*] Loaded PPO model from: {self.model_path}")
            print(f"[*] State dimension: {state_dim}")

    def _create_env_fn(self, question_id: int):
        """Create environment function for a specific problem. / 为特定问题创建环境函数。"""
        def env_fn():
            # Create a wrapper that uses the fitness_function / 创建使用 fitness_function 的包装器
            return cmaes(question_id)
        return env_fn

    def optimize(self) -> Dict:
        """
        Run the optimization process. / 运行优化过程。

        Returns / 返回:
            dict: Optimization results / 优化结果
                - best_so_far_x: Best solution found / 找到的最优解
                - best_so_far_y: Best fitness value / 最优适应度值
                - n_function_evaluations: Total FEs used / 使用的总 FEs
                - runtime: Total runtime in seconds / 总运行时间（秒）
                - fitness_history: History of best fitness values / 最优适应度值历史
        """
        self.start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"LCC-CMAES Optimization")
            print(f"{'='*60}")
            print(f"Problem dimension: {self.ndim_problem}")
            print(f"Max function evaluations: {self.max_function_evaluations}")
            print(f"Number of subgroups (m): {self.m}")
            print(f"Subgroup population: {self.sub_popsize}")
            print(f"Action selections (ns): {self.ns}")
            print(f"{'='*60}\n")

        # -------------------------------------------------------------------------
        # Create parallel environments / 创建并行环境
        # -------------------------------------------------------------------------
        # Create environment factory that captures fitness_function / 创建捕获 fitness_function 的环境工厂
        def create_env(
            qid,
            fun,
            max_fes=self.max_function_evaluations,
            ndim=self.ndim_problem,
            lb=self.lower_boundary,
            ub=self.upper_boundary,
        ):
            # Use external problem metadata when a custom fitness function is provided.
            lower = float(np.min(np.asarray(lb)))
            upper = float(np.max(np.asarray(ub)))
            return cmaes(
                question=qid,
                fitness_function=fun,
                max_fes=max_fes,
                problem_dimension=int(ndim),
                lower_bound=lower,
                upper_bound=upper,
            )

        env_list = [lambda q=self.problem_id, f=self.fitness_function: create_env(q, f)
                    for _ in range(self.each_question_batch_num)]
        envs = venvs.SubprocVectorEnv(env_list)

        # Initialize state / 初始化状态
        state = envs.reset()
        state = torch.FloatTensor(state).to(self.device)
        state = torch.where(torch.isnan(state), torch.zeros_like(state), state)

        # -------------------------------------------------------------------------
        # Main optimization loop / 主优化循环
        # -------------------------------------------------------------------------
        for t in tqdm(
            range(self.ns),
            disable=(self.verbose == 0),
            desc='LCC-CMAES',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        ):
            # Select action using PPO agent / 使用 PPO 智能体选择动作
            action, _, _ = self.agent.actor(state)
            action_cpu = action.cpu()

            # Environment step / 环境步进
            state, _, is_end, info = envs.step(action_cpu)

            # Update best solution / 更新最优解
            for inf in info:
                current_fitness = inf.get('gbest_val', np.inf)
                if current_fitness < self.best_so_far_y:
                    self.best_so_far_y = current_fitness
                # Update fitness_record (environment keeps cumulative record) / 更新适应度记录（环境保持累积记录）
                if 'fitness_record' in inf:
                    self.fitness_record = inf['fitness_record']

            # Record fitness history / 记录适应度历史
            if len(info) > 0:
                avg_fitness = np.mean([inf.get('gbest_val', np.inf) for inf in info])
                self.fitness_history.append(avg_fitness)

            # Update state / 更新状态
            state = torch.FloatTensor(state).to(self.device)
            state = torch.where(torch.isnan(state), torch.zeros_like(state), state)

            # Log progress / 记录进度
            if self.verbose and (t % max(1, self.ns // 10) == 0):
                print(f"Step {t}/{self.ns} | Best fitness: {self.best_so_far_y:.6e}")

            # Check termination / 检查终止条件
            if all(is_end):
                break

        # Close environments / 关闭环境
        envs.close()

        # -------------------------------------------------------------------------
        # Collect results / 收集结果
        # -------------------------------------------------------------------------
        self.runtime = time.time() - self.start_time

        # Get final fitness record (environment keeps cumulative record) / 获取最终适应度记录（环境保持累积记录）
        merged_fitness_record = self.fitness_record if self.fitness_record else []

        results = {
            'best_so_far_x': self.best_so_far_x,
            'best_so_far_y': self.best_so_far_y,
            'n_function_evaluations': self.n_function_evaluations,
            'runtime': self.runtime,
            'fitness_history': self.fitness_history,
            'fitness_record': merged_fitness_record  # Include merged fitness record / 包含合并的适应度记录
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Optimization completed")
            print(f"Best fitness: {self.best_so_far_y:.6e}")
            print(f"Runtime: {self.runtime:.2f}s")
            print(f"{'='*60}\n")

        return results


# =============================================================================
# Convenience Functions / 便捷函数
# =============================================================================

def optimize_with_lcc_cmaes(
    fitness_function,
    ndim_problem: int = 1000,
    lower_boundary: float = -100.0,
    upper_boundary: float = 100.0,
    model_path: str = None,
    max_function_evaluations: int = 3000000,
    seed_rng: int = 2024,
    device: str = 'cuda',
    verbose: int = 1,
    **kwargs
) -> Dict:
    """
    Convenience function for quick optimization with LCC-CMAES.
    使用 LCC-CMAES 进行快速优化的便捷函数。

    Parameters / 参数:
        fitness_function: Objective function to minimize / 要最小化的目标函数
        ndim_problem: Problem dimension / 问题维度
        lower_boundary: Lower bound of search space / 搜索空间下界
        upper_boundary: Upper bound of search space / 搜索空间上界
        model_path: Path to trained PPO model / 训练好的 PPO 模型路径
        max_function_evaluations: Max FEs budget / 最大函数评估预算
        seed_rng: Random seed / 随机种子
        device: Device for PPO model / PPO 模型设备
        verbose: Logging verbosity / 日志详细度
        **kwargs: Additional options passed to optimizer / 传递给优化器的额外选项

    Returns / 返回:
        dict: Optimization results / 优化结果

    Example / 示例:
        >>> results = optimize_with_ppo_cmaes(
        ...     fitness_function=lambda x: sum(x**2),
        ...     ndim_problem=1000,
        ...     model_path='path/to/model.pt',
        ...     max_function_evaluations=100000
        ... )
        >>> print(f"Best: {results['best_so_far_y']}")
    """
    problem = {
        'fitness_function': fitness_function,
        'ndim_problem': ndim_problem,
        'lower_boundary': lower_boundary,
        'upper_boundary': upper_boundary
    }

    # Use local model path if not provided / 如果未提供则使用本地模型路径
    if model_path is None:
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(_current_dir, 'model', 'epoch-9.pt')

    options = {
        'model_path': model_path,
        'max_function_evaluations': max_function_evaluations,
        'seed_rng': seed_rng,
        'device': device,
        'verbose': verbose,
        **kwargs
    }

    optimizer = LCC_CMAES(problem, options)
    return optimizer.optimize()
