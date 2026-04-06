"""
Shared Optimization Utilities / 共享优化工具模块

This module contains common optimization-related utilities used across
different baseline algorithms to reduce code duplication.

此模块包含不同基线算法中使用的通用优化相关工具，以减少代码重复。

Classes:
    FitnessRecorder: Records fitness values during optimization / 在优化过程中记录适应度值

Functions:
    combine_vectors: Combine vectors at specified locations / 在指定位置组合向量
    build_grouping_result: Build variable grouping for CC / 为 CC 构建变量分组
    parallel_optimization: Run optimization tasks in parallel / 并行运行优化任务
    optimization_task_cc: Single CC optimization task / 单次 CC 优化任务
    optimization_task_nda: Single NDA optimization task / 单次 NDA 优化任务

Author: WCC Project
Date: 2026-03-11
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import math
import time
from typing import List, Tuple, Dict, Any, Optional, Callable

import numpy as np

from baseline.CMAES.cmaes import CMAES
from baseline.FCMAES.fcmaes import FCMAES
from benchmark.aob import Benchmark
from benchmark.aob.utils import partition_p_and_s
from utils.config import PATHS


# =============================================================================
# Constants / 常量
# =============================================================================

# Default project paths / 默认项目路径 (use PATHS for consistency / 使用 PATHS 保持一致性)
DATA_DIR = PATHS.aob_data

# Default optimizer parameters / 默认优化器参数
DEFAULT_CC_SIGMA = 0.5
DEFAULT_NDA_SIGMA_RATIO = 0.3  # 30% of search range / 搜索范围的 30%
DEFAULT_VERBOSE = 1000
DEFAULT_EARLY_STOPPING = 1000


# =============================================================================
# Fitness Recorder / 适应度记录器
# =============================================================================

class FitnessRecorder:
    """
    Records fitness values during optimization / 在优化过程中记录适应度值

    This class wraps a fitness function and maintains a history of all
    fitness values returned by the function. It is callable and behaves
    like the original function while recording results.

    此类包装一个适应度函数并维护该函数返回的所有适应度值的历史记录。
    它是可调用的，行为类似于原始函数，同时记录结果。

    Attributes:
        fun (Callable): The wrapped fitness function / 被包装的适应度函数
        fitness_record (List[float]): History of fitness values / 适应度值历史记录

    Example:
        >>> recorder = FitnessRecorder(lambda x: [sum(x**2)])
        >>> result = recorder(np.array([1, 2, 3]))
        >>> print(recorder.fitness_record)  # [14.0]
    """

    def __init__(self, fun: Callable):
        """
        Initialize the fitness recorder / 初始化适应度记录器

        Args:
            fun: The fitness function to wrap / 要包装的适应度函数
                  Should return a list of fitness values / 应返回适应度值列表
        """
        self.fun = fun
        self.fitness_record: List[float] = []

    def __call__(self, x: np.ndarray) -> List[float]:
        """
        Evaluate the fitness function and record results / 评估适应度函数并记录结果

        Args:
            x: Input array to evaluate / 待评估的输入数组

        Returns:
            List of fitness values returned by the wrapped function /
            被包装函数返回的适应度值列表
        """
        fitness = self.fun(x)
        self.fitness_record.extend(fitness)
        return fitness

    def reset(self) -> None:
        """
        Clear the fitness record / 清空适应度记录

        Useful for reusing the same recorder for multiple runs.
        用于在多次运行中重用同一个记录器。
        """
        self.fitness_record = []

    def get_best_fitness(self) -> Optional[float]:
        """
        Get the best (minimum) fitness value recorded / 获取记录的最佳（最小）适应度值

        Returns:
            Best fitness value, or None if no records / 最佳适应度值，无记录时返回 None
        """
        if not self.fitness_record:
            return None
        return min(self.fitness_record)


# =============================================================================
# Vector Operations / 向量操作
# =============================================================================

def combine_vectors(
    small_vec: np.ndarray,
    background_vec: np.ndarray,
    location: Optional[List[int]]
) -> np.ndarray:
    """
    Combine a small vector with a background vector at specified locations /
    将小向量与背景向量在指定位置组合

    This function is used in Cooperative Coevolution (CC) to optimize subspaces.
    It creates a batch of full-dimensional vectors by inserting the small_vec
    into the background_vec at the specified indices.

    此函数用于协同进化（CC）中优化子空间。它通过将 small_vec 在指定索引处
    插入 background_vec 来创建一批全维向量。

    Args:
        small_vec: Small vector to insert, shape (n, d_sub) / 待插入的小向量，形状 (n, d_sub)
        background_vec: Background vector to copy, shape (d,) / 待复制的背景向量，形状 (d,)
        location: Indices where small_vec columns should be placed /
                 small_vec 列应放置的索引位置。If None, returns small_vec directly.
                 如果为 None，直接返回 small_vec。

    Returns:
        Combined array, shape (n, d) / 组合后的数组，形状 (n, d)

    Example:
        >>> small = np.array([[1, 2], [3, 4]])
        >>> background = np.array([0, 0, 0, 0, 0])
        >>> result = combine_vectors(small, background, [1, 3])
        >>> print(result)
        [[0 1 0 2 0]
         [0 3 0 4 0]]
    """
    if location is None:
        return small_vec
    combination = np.tile(background_vec, (len(small_vec), 1))
    combination[:, location] = small_vec
    return combination


# =============================================================================
# Grouping Utilities / 分组工具
# =============================================================================

def build_grouping_result(
    fun_id: int,
    grouping_mode: str,
    overlap_map: Optional[Dict[int, int]] = None,
    chunk_count: int = 20,
    dimension: int = 1000,
    data_dir: Path = DATA_DIR
) -> Optional[List[List[int]]]:
    """
    Build variable grouping for Cooperative Coevolution (CC) /
    为协同进化（CC）构建变量分组

    This function supports multiple grouping strategies:
    - 'none': No grouping (for NDA) / 无分组（用于 NDA）
    - 'equal_split': Split variables into equal chunks / 将变量等分为若干块
    - 'partition': Load predefined partition from files with optional overlap /
                  从文件加载预定义的分区，可选重叠

    此函数支持多种分组策略：
    - 'none': 无分组（用于 NDA）
    - 'equal_split': 将变量等分为若干块
    - 'partition': 从文件加载预定义的分区，可选重叠

    Args:
        fun_id: Function ID for loading partition files / 用于加载分区文件的函数 ID
        grouping_mode: Grouping strategy ('none', 'equal_split', or 'partition') /
                      分组策略
        overlap_map: Mapping from fun_id to overlap degree (required for 'partition') /
                    函数 ID 到重叠度的映射（'partition' 模式必需）
        chunk_count: Number of chunks for 'equal_split' mode / 'equal_split' 模式的块数
        dimension: Problem dimension / 问题维度
        data_dir: Directory containing partition data files / 包含分区数据文件的目录

    Returns:
        List of groups (each group is a list of indices), or None for 'none' mode /
        分组列表（每组是索引列表），'none' 模式返回 None

    Raises:
        ValueError: If grouping_mode is unsupported or required parameters are missing /
                    如果分组模式不支持或缺少必需参数

    Example:
        >>> # Equal split mode / 等分模式
        >>> groups = build_grouping_result(1, 'equal_split', chunk_count=4, dimension=10)
        >>> print(groups)  # [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]] (approximately)

        >>> # Partition mode / 分区模式
        >>> overlap_map = {1: 5, 2: 10}
        >>> groups = build_grouping_result(1, 'partition', overlap_map=overlap_map)
    """
    if grouping_mode == "none":
        return None

    if grouping_mode == "equal_split":
        return [chunk.tolist() for chunk in np.array_split(np.arange(dimension), chunk_count)]

    if grouping_mode == "partition":
        if overlap_map is None:
            raise ValueError("overlap_map is required when grouping_mode='partition'")
        overlap = overlap_map[fun_id]
        p_file_path = str(data_dir / f"F{fun_id}-p.txt")
        s_file_path = str(data_dir / f"F{fun_id}-s.txt")
        return partition_p_and_s(p_file_path, s_file_path, overlap=overlap)

    raise ValueError(f"Unsupported grouping_mode: {grouping_mode}. Use 'none', 'equal_split', or 'partition'.")


# =============================================================================
# CC Optimization Task / CC 优化任务
# =============================================================================

def optimization_task_cc(
    fun_id: int,
    func_name: str,
    nondep: bool,
    best_individual: np.ndarray,
    max_fes: int,
    grouping_result: List[List[int]],
    info: Dict[str, Any],
    reverse: bool = False,
    bench: Optional[Benchmark] = None,
    sigma: float = DEFAULT_CC_SIGMA,
    early_stopping: int = DEFAULT_EARLY_STOPPING
) -> Tuple[float, List[float]]:
    """
    Single Cooperative Coevolution (CC) optimization task / 单次协同进化优化任务

    CC optimizes high-dimensional problems by decomposing them into smaller subspaces.
    Each subspace is optimized iteratively while keeping other dimensions fixed.

    CC 通过将高维问题分解为更小的子空间来优化。每个子空间迭代优化，
    同时保持其他维度固定。

    Args:
        fun_id: Function ID / 函数 ID
        func_name: Function name / 函数名称
        nondep: Non-decomposition mode flag / 非分解模式标志
        best_individual: Current best solution [dimension] / 当前最优解
        max_fes: Maximum function evaluations / 最大函数评估次数
        grouping_result: List of variable groups / 变量分组列表
        info: Problem information dict with keys: 'lower', 'upper', 'dimension' /
              问题信息字典，包含键：'lower', 'upper', 'dimension'
        reverse: Whether to reverse group order / 是否反转分组顺序
        bench: Benchmark instance / 基准测试实例
        sigma: Initial step size for CMAES / CMAES 的初始步长
        early_stopping: Early stopping evaluations / 早停评估次数

    Returns:
        Tuple of (elapsed_time, fitness_record) / (耗时, 适应度记录) 的元组

    Algorithm:
        1. Split max_fes among all groups / 将 max_fes 分配给所有组
        2. For each group: optimize subspace using CMAES / 对每组：使用 CMAES 优化子空间
        3. Update best_individual with subspace results / 用子空间结果更新 best_individual
        4. Repeat until max_fes exhausted / 重复直到 max_fes 耗尽
    """
    time_start = time.time()
    fun = bench.get_function(func_name, fun_id, nondep)
    fun_recorder = FitnessRecorder(fun)
    sum_fes = 0

    # Optionally reverse group order / 可选：反转分组顺序
    if reverse:
        grouping_result = grouping_result[::-1]

    # Main CC loop: iterate through all groups until budget exhausted /
    # 主循环：遍历所有组直到预算耗尽
    while sum_fes < max_fes:
        sub_num = len(grouping_result)
        sub_fes = math.ceil((max_fes - sum_fes) / sub_num)

        # Optimize each subspace / 优化每个子空间
        for dims in grouping_result:
            # Create subspace objective function / 创建子空间目标函数
            obj_func = lambda x_batch: fun_recorder(combine_vectors(x_batch, best_individual, dims))

            # Configure CMAES for subspace / 为子空间配置 CMAES
            problem_cc = {
                "fitness_function": obj_func,
                "ndim_problem": len(dims),
                "lower_boundary": info["lower"] * np.ones((len(dims),)),
                "upper_boundary": info["upper"] * np.ones((len(dims),)),
            }
            options_cc = {
                "max_function_evaluations": sub_fes,
                "mean": (best_individual[dims],),
                "sigma": sigma,
                "is_restart": False,
                "verbose": DEFAULT_VERBOSE,
                "early_stopping_evaluations": early_stopping,
            }

            # Run CMAES on subspace / 在子空间上运行 CMAES
            optimizer_cc = CMAES(problem_cc, options_cc)
            results_cc = optimizer_cc.optimize()

            # Update solution with subspace results / 用子空间结果更新解
            best_individual[dims] = results_cc["best_so_far_x"].copy()
            sum_fes += results_cc["n_function_evaluations"]

    time_end = time.time()
    return (time_end - time_start), fun_recorder.fitness_record


# =============================================================================
# NDA Optimization Task / NDA 优化任务
# =============================================================================

def optimization_task_nda(
    fun_id: int,
    func_name: str,
    nondep: bool,
    best_individual: np.ndarray,
    max_fes: int,
    info: Dict[str, Any],
    bench: Benchmark,
    sigma_ratio: float = DEFAULT_NDA_SIGMA_RATIO
) -> Tuple[float, List[float]]:
    """
    Single Non-Decomposition Algorithm (NDA) optimization task / 单次非分解算法优化任务

    NDA optimizes the full problem dimension without decomposition, using FCMAES
    (Fully Competitive CMA-ES). This is suitable for non-separable problems.

    NDA 在完整问题维度上优化而不进行分解，使用 FCMAES（完全竞争 CMA-ES）。
    这适用于不可分离的问题。

    Args:
        fun_id: Function ID / 函数 ID
        func_name: Function name / 函数名称
        nondep: Non-decomposition mode flag / 非分解模式标志
        best_individual: Initial solution [dimension] / 初始解
        max_fes: Maximum function evaluations / 最大函数评估次数
        info: Problem information dict with keys: 'lower', 'upper', 'dimension' /
              问题信息字典，包含键：'lower', 'upper', 'dimension'
        bench: Benchmark instance / 基准测试实例
        sigma_ratio: Sigma as ratio of search range / Sigma 作为搜索范围的比例

    Returns:
        Tuple of (elapsed_time, fitness_record) / (耗时, 适应度记录) 的元组
    """
    time_start = time.time()
    fun = bench.get_function(func_name, fun_id, nondep)
    fun_recorder = FitnessRecorder(fun)

    # Calculate sigma as ratio of search range / 计算 sigma 为搜索范围的比例
    sigma = sigma_ratio * (info["upper"] - info["lower"])

    # Configure FCMAES for full problem / 为完整问题配置 FCMAES
    problem = {
        "fitness_function": fun_recorder,
        "ndim_problem": info["dimension"],
        "lower_boundary": info["lower"] * np.ones((info["dimension"],)),
        "upper_boundary": info["upper"] * np.ones((info["dimension"],)),
    }
    options = {
        "max_function_evaluations": max_fes,
        "mean": (best_individual,),
        "sigma": sigma,
        "is_restart": False,
        "verbose": DEFAULT_VERBOSE,
    }

    # Run FCMAES optimization / 运行 FCMAES 优化
    optimizer = FCMAES(problem, options)
    results = optimizer.optimize()

    time_end = time.time()
    return (time_end - time_start), fun_recorder.fitness_record


# =============================================================================
# Parallel Optimization / 并行优化
# =============================================================================

def parallel_optimization(
    fun_id: int,
    func_name: str,
    nondep: bool,
    best_individual: np.ndarray,
    max_fes: int,
    cycle_num: int,
    info: Dict[str, Any],
    bench: Benchmark,
    algorithm_type: str = "CC",
    grouping_result: Optional[List[List[int]]] = None,
    reverse: bool = False,
    **task_kwargs
) -> Tuple[float, List[List[float]]]:
    """
    Run optimization tasks in parallel (supports both CC and NDA) /
    并行运行优化任务（支持 CC 和 NDA）

    This function runs multiple independent optimization tasks in parallel using
    ProcessPoolExecutor and returns the average time and all fitness records.

    此函数使用 ProcessPoolExecutor 并行运行多个独立的优化任务，
    并返回平均时间和所有适应度记录。

    Args:
        fun_id: Function ID / 函数 ID
        func_name: Function name / 函数名称
        nondep: Non-decomposition mode flag / 非分解模式标志
        best_individual: Initial solution [dimension] / 初始解
        max_fes: Maximum function evaluations per task / 每个任务的最大函数评估次数
        cycle_num: Number of parallel runs / 并行运行次数
        info: Problem information / 问题信息
        bench: Benchmark instance / 基准测试实例
        algorithm_type: Algorithm type ('CC' or 'NDA') / 算法类型
        grouping_result: Variable grouping (for CC only) / 变量分组（仅 CC）
        reverse: Whether to reverse group order (for CC only) / 是否反转分组顺序（仅 CC）
        **task_kwargs: Additional keyword arguments passed to optimization task /
                      传递给优化任务的额外关键字参数

    Returns:
        Tuple of (average_time, fitness_records) / (平均时间, 适应度记录列表) 的元组
        - average_time: Average execution time across all runs / 所有运行的平均执行时间
        - fitness_records: List of fitness records from each run / 每次运行的适应度记录列表

    Raises:
        ValueError: If algorithm_type is not 'CC' or 'NDA' / 如果 algorithm_type 不是 'CC' 或 'NDA'

    Example:
        >>> avg_time, records = parallel_optimization(
        ...     fun_id=1, func_name="ackley", nondep=True,
        ...     best_individual=np.zeros(1000), max_fes=3e6,
        ...     cycle_num=10, info=info, bench=bench,
        ...     algorithm_type="CC", grouping_result=groups
        ... )
    """
    # Select optimization task function based on algorithm type / 根据算法类型选择优化任务函数
    if algorithm_type == "CC":
        task_func = optimization_task_cc
        task_kwargs.update({
            "grouping_result": grouping_result,
            "reverse": reverse,
            "bench": bench,
        })
    elif algorithm_type == "NDA":
        task_func = optimization_task_nda
        task_kwargs.update({
            "info": info,
            "bench": bench,
        })
    else:
        raise ValueError(f"Unsupported algorithm_type: {algorithm_type}. Use 'CC' or 'NDA'.")

    # Run tasks in parallel / 并行运行任务
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(cycle_num):
            futures.append(
                executor.submit(
                    task_func,
                    fun_id,
                    func_name,
                    nondep,
                    best_individual.copy(),
                    max_fes,
                    **task_kwargs
                )
            )

        # Collect results / 收集结果
        average_time = 0
        fitness_record = []
        for future in futures:
            result = future.result()
            fitness_record.append(result[1])
            average_time += result[0]

        return average_time / cycle_num, fitness_record
