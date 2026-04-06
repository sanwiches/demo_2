"""
协同进化 CMA-ES 优化器模块 / Cooperative Coevolution CMA-ES Optimizer Module.

此模块实现了基于协同进化（CC）的 CMA-ES 算法，使用自适应策略选择
不同的子空间分解方法（MiVD、MaVD、RaVD）进行优化。
This module implements the Cooperative Coevolution (CC) CMA-ES algorithm,
using adaptive strategy to select different subspace decomposition methods
(MiVD, MaVD, RaVD) for optimization.

算法流程 / Algorithm Flow:
    1. 初始化种群并评估适应度 / Initialize population and evaluate fitness
    2. 使用自适应选择器选择分解策略 / Use adaptive selector to choose decomposition strategy
    3. 对每个子空间独立运行 CMA-ES / Run CMA-ES independently on each subspace
    4. 更新全局最优解和协方差矩阵 / Update global best solution and covariance matrix
    5. 根据改进比例更新自适应选择器 / Update adaptive selector based on improvement ratio

主要组件 / Main Components:
    - ccc_cmaes: 协同进化 CMA-ES 主函数 / Cooperative coevolution CMA-ES main function
    - _combine_solution: 组合子空间解与完整解 / Combine subspace solution with full solution
    - _optimize_subspace: 在子空间上运行 CMA-ES / Run CMA-ES on subspace
    - _initialize_population: 初始化种群 / Initialize population

使用示例 / Usage Example:
    >>> from env.optimizer.cc_cmaes.cc_cmaes import ccc_cmaes
    >>> import numpy as np
    >>>
    >>> def objective(x):
    ...     return np.sum(x**2)
    >>>
    >>> info = {
    ...     'dimension': 10,
    ...     'lower': -5.0,
    ...     'upper': 5.0
    ... }
    >>>
    >>> result = ccc_cmaes(
    ...     fun=objective,
    ...     info=info,
    ...     seed=42,
    ...     max_fes=10000,
    ...     verbose=True
    ... )
"""

import logging
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import cma
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from env.optimizer.cc_cmaes.adapter import ActionAdapter
from env.optimizer.cc_cmaes.mivd import mivd_groups
from env.optimizer.cc_cmaes.mavd import mavd_groups
from env.optimizer.cc_cmaes.ravd import ravd_groups

# ==========================================
# 配置常量 / Configuration Constants
# ==========================================

# 初始评估次数 / Initial evaluation count
INIT_FES = 50

# 默认参数 / Default parameters
DEFAULT_SUBSPACE_DIM = 50          # 子空间维度 / Subspace dimension
DEFAULT_SUB_FES = 500              # 每个子空间分配的评估次数 / FEs per subspace
DEFAULT_SAMPLE_SIZE = 10           # CMA-ES 采样大小 / CMA-ES sample size
DEFAULT_ADAPTER_LAYERS = 5         # 自适应选择器层数 / Adapter layers

# 动作名称映射 / Action name mapping
ACTION_NAMES = {
    'MiVD': 'mivd_groups',
    'MaVD': 'mavd_groups',
    'RaVD': 'ravd_groups'
}

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)

# 分组函数映射 / Grouping function mapping
_GROUPING_FUNCTIONS = {
    'MiVD': mivd_groups,
    'MaVD': mavd_groups,
    'RaVD': ravd_groups
}


# ==========================================
# 辅助函数 / Helper Functions
# ==========================================

def _combine_solution(
    subspace_solution: npt.NDArray[np.float64],
    background_solution: npt.NDArray[np.float64],
    subspace_indices: Optional[npt.NDArray[np.int64]]
) -> npt.NDArray[np.float64]:
    """
    组合子空间解与完整解 / Combine subspace solution with full solution.

    将子空间的解组合到完整解的指定位置，其他位置保持背景解不变。
    Combines subspace solution into specified positions of full solution,
    keeping other positions as background solution.

    Args:
        subspace_solution: 子空间解向量 / Subspace solution vector, shape (subspace_dim,)
        background_solution: 背景完整解 / Background full solution, shape (dimension,)
        subspace_indices: 子空间索引 / Subspace indices, None 表示返回子空间解 itself

    Returns:
        组合后的完整解 / Combined full solution, shape (dimension,)

    Example:
        >>> subspace = np.array([1.0, 2.0])
        >>> background = np.array([0.0, 0.0, 0.0, 0.0])
        >>> indices = np.array([0, 2])
        >>> result = _combine_solution(subspace, background, indices)
        >>> print(result)  # [1.0, 0.0, 2.0, 0.0]
    """
    if subspace_indices is None:
        return subspace_solution

    # 复制背景解并替换子空间位置 / Copy background and replace subspace positions
    combined = np.tile(background_solution, (len(subspace_solution), 1))
    combined[:, subspace_indices] = subspace_solution
    return combined


def _initialize_population(
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    population_size: int,
    seed: int
) -> npt.NDArray[np.float64]:
    """
    初始化种群 / Initialize population.

    在给定边界内均匀随机采样生成初始种群。
    Generates initial population by uniform random sampling within given bounds.

    Args:
        dimension: 问题维度 / Problem dimension
        lower_bound: 下界 / Lower bound
        upper_bound: 上界 / Upper bound
        population_size: 种群大小 / Population size
        seed: 随机种子 / Random seed

    Returns:
        初始种群矩阵 / Initial population matrix, shape (population_size, dimension)
    """
    rng = np.random.default_rng(seed)
    population = rng.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(population_size, dimension)
    )
    return population


def _optimize_subspace(
    objective_func: Callable[[npt.NDArray[np.float64]], float],
    subspace_indices: npt.NDArray[np.int64],
    current_solution: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64],
    sigma: float,
    bounds: Tuple[float, float],
    subspace_fes: int,
    sample_size: int,
    seed: int,
    verbose: bool
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """
    在子空间上运行 CMA-ES 优化 / Run CMA-ES optimization on subspace.

    Args:
        objective_func: 目标函数 / Objective function
        subspace_indices: 子空间维度索引 / Subspace dimension indices
        current_solution: 当前完整解 / Current full solution
        covariance: 当前协方差矩阵 / Current covariance matrix
        sigma: 当前步长 / Current step size
        bounds: (下界, 上界) / (lower_bound, upper_bound)
        subspace_fes: 子空间分配的评估次数 / FEs allocated for subspace
        sample_size: CMA-ES 采样大小 / CMA-ES sample size
        seed: 随机种子 / Random seed
        verbose: 是否输出 CMA-ES 信息 / Whether to output CMA-ES info

    Returns:
        (更新后的子空间解, 更新后的子空间协方差, 最终步长)
        (Updated subspace solution, Updated subspace covariance, Final step size)
    """
    lower_bound, upper_bound = bounds

    # 抑制 CMA-ES 警告 / Suppress CMA-ES warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cma.evolution_strategy._CMASolutionDict = cma.evolution_strategy._CMASolutionDict_empty

        # 配置 CMA-ES 选项 / Configure CMA-ES options
        cma_options = {
            'popsize': sample_size,
            'bounds': [lower_bound, upper_bound],
            'seed': seed,
            'maxfevals': subspace_fes,
            'verbose': -9 if not verbose else 0
        }

        # 初始化子空间 CMA-ES / Initialize subspace CMA-ES
        initial_mean = current_solution[subspace_indices]
        subspace_es = cma.CMAEvolutionStrategy(initial_mean, sigma, cma_options)

    # 设置初始协方差矩阵 / Set initial covariance matrix
    subspace_es.sm.C = covariance[np.ix_(subspace_indices, subspace_indices)].copy()
    subspace_es.sm.update_now(-1)
    subspace_es._updateBDfromSM()

    # 定义子空间目标函数 / Define subspace objective function
    def subspace_objective(x_batch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """评估子空间解的目标函数值 / Evaluate objective for subspace solutions."""
        combined = _combine_solution(x_batch, current_solution, subspace_indices)
        return np.array([objective_func(x) for x in combined])

    # 运行 CMA-ES 优化 / Run CMA-ES optimization
    iteration = 0
    with tqdm(total=subspace_fes, disable=not verbose, desc=f"  子空间优化 / Subspace {subspace_indices[0]}") as pbar:
        while not subspace_es.stop():
            x_batch = subspace_es.ask()
            fitness_values = subspace_objective(x_batch)
            subspace_es.tell(x_batch, fitness_values)
            pbar.update(len(x_batch))
            iteration += 1

    # 返回优化结果 / Return optimization results
    best_solution = subspace_es.result[0].copy()
    best_covariance = subspace_es.sm.C.copy()
    final_sigma = subspace_es.sigma

    logger.debug(
        f"子空间优化完成 / Subspace optimization complete: "
        f"indices={subspace_indices}, iterations={iteration}, sigma={final_sigma:.3e}"
    )

    return best_solution, best_covariance, final_sigma


# ==========================================
# 主函数 / Main Function
# ==========================================

def ccc_cmaes(
    fun: Callable[[npt.NDArray[np.float64]], float],
    info: Dict[str, any],
    seed: int,
    max_fes: int,
    verbose: bool = True
) -> Tuple[float, npt.NDArray[np.float64], List[int], List[float], List[int]]:
    """
    协同进化 CMA-ES 优化 / Cooperative Coevolution CMA-ES optimization.

    使用自适应策略选择子空间分解方法，在每个子空间上独立运行 CMA-ES，
    协同优化高维问题。
    Uses adaptive strategy to select subspace decomposition methods,
    runs CMA-ES independently on each subspace for cooperative optimization
    of high-dimensional problems.

    Args:
        fun: 目标函数 / Objective function to minimize
        info: 问题信息字典 / Problem info dictionary containing:
            - 'dimension': 问题维度 / Problem dimension
            - 'lower': 下界 / Lower bound
            - 'upper': 上界 / Upper bound
        seed: 随机种子 / Random seed
        max_fes: 最大函数评估次数 / Maximum function evaluations
        verbose: 是否输出详细信息 / Whether to output detailed information

    Returns:
        包含优化结果的元组 / Tuple containing optimization results:
            - 最优适应度 / Best fitness value
            - 最优个体 / Best individual solution
            - 动作记录 / Action record (0: MiVD, 1: MaVD, 2: RaVD)
            - 适应度历史 / Fitness history
            - 评估次数历史 / FEs history

    Raises:
        ValueError: 当 info 中缺少必需参数时 / When required parameters are missing in info
    """
    # ==========================================
    # 参数验证 / Parameter Validation
    # ==========================================
    required_keys = {'dimension', 'lower', 'upper'}
    missing_keys = required_keys - set(info.keys())
    if missing_keys:
        raise ValueError(f"info 中缺少必需参数 / Missing required parameters in info: {missing_keys}")

    dimension = info['dimension']
    lower_bound = info['lower']
    upper_bound = info['upper']

    # ==========================================
    # 初始化 / Initialization
    # ==========================================
    np.random.seed(seed)
    logger.info(f"开始协同进化 CMA-ES / Starting CC-CMA-ES: dimension={dimension}, max_fes={max_fes}")

    # 初始化种群 / Initialize population
    init_population = _initialize_population(
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        population_size=INIT_FES,
        seed=seed
    )

    # 评估初始种群 / Evaluate initial population
    init_fitness = np.array([fun(x) for x in init_population])
    best_fitness = np.min(init_fitness)
    best_individual = init_population[np.argmin(init_fitness)]

    # 初始化历史记录 / Initialize history records
    fitness_history: List[float] = [best_fitness]
    fes_history: List[int] = [INIT_FES]
    action_record: List[int] = []

    if verbose:
        print(f'初始最优适应度 / Initial best fitness: {best_fitness:.2e}')

    # ==========================================
    # 协同进化优化主循环 / CC Optimization Main Loop
    # ==========================================
    # 初始化参数 / Initialize parameters
    subspace_dim = DEFAULT_SUBSPACE_DIM
    subspace_fes = DEFAULT_SUB_FES
    sample_size = DEFAULT_SAMPLE_SIZE

    # 初始化自适应选择器 / Initialize adaptive selector
    adapter = ActionAdapter(
        actions=list(ACTION_NAMES.keys()),
        layers=DEFAULT_ADAPTER_LAYERS
    )

    # 初始化全局参数 / Initialize global parameters
    working_solution = best_individual.copy()
    covariance = np.eye(dimension)
    sigma = 0.5 * (upper_bound - lower_bound)

    sum_fes = INIT_FES
    iteration = 0

    while sum_fes < max_fes:
        iteration += 1
        old_fitness = best_fitness

        if verbose:
            print(f"\n=== 迭代 / Iteration {iteration} ===")
            print(f"当前适应度 / Current fitness: {best_fitness:.2e}")
            print(f"已用评估次数 / Used FEs: {sum_fes}/{max_fes}")

        # 选择分解策略 / Select decomposition strategy
        action = adapter.decide()
        action_num = list(ACTION_NAMES.keys()).index(action)
        action_record.append(action_num)

        if verbose:
            print(f"选择策略 / Selected strategy: {action}")

        # 生成分组 / Generate groups
        grouping_func = _GROUPING_FUNCTIONS[action]
        subgroups = grouping_func(dimension, subspace_dim, covariance)

        # 对每个子空间进行优化 / Optimize each subspace
        for subgroup_indices in subgroups:
            # 在子空间上运行 CMA-ES / Run CMA-ES on subspace
            subspace_solution, subspace_cov, subspace_sigma = _optimize_subspace(
                objective_func=fun,
                subspace_indices=subgroup_indices,
                current_solution=working_solution,
                covariance=covariance,
                sigma=sigma,
                bounds=(lower_bound, upper_bound),
                subspace_fes=subspace_fes,
                sample_size=sample_size,
                seed=seed + iteration,
                verbose=False
            )

            # 更新全局解 / Update global solution
            working_solution[subgroup_indices] = subspace_solution
            covariance[np.ix_(subgroup_indices, subgroup_indices)] = subspace_cov
            sigma = subspace_sigma

            # 评估并更新最优解 / Evaluate and update best solution
            current_fitness = fun(working_solution)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_individual = working_solution.copy()
                if verbose:
                    print(f"  更新最优解 / New best: {best_fitness:.2e}")

            fitness_history.append(best_fitness)
            sum_fes += subspace_fes
            fes_history.append(sum_fes)

        # 计算改进比例并更新选择器 / Calculate improvement ratio and update selector
        if old_fitness > 0:
            contribution_ratio = abs(old_fitness - best_fitness) / old_fitness
        else:
            contribution_ratio = abs(best_fitness - old_fitness)

        adapter.update(action, contribution_ratio)

        if verbose:
            print(f"改进比例 / Improvement ratio: {contribution_ratio:.4f}")

    # ==========================================
    # 返回结果 / Return Results
    # ==========================================
    logger.info(
        f"优化完成 / Optimization complete: "
        f"best_fitness={best_fitness:.2e}, total_iterations={iteration}"
    )

    return best_fitness, best_individual, action_record, fitness_history, fes_history
