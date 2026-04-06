"""
Utility Functions for CMA-ES PPO Training and Testing
CMA-ES PPO 训练和测试的工具函数

This module provides a comprehensive collection of utility functions for
data processing, logging, visualization, parallel execution, and PyTorch
model management for the CMA-ES PPO optimization framework.
本模块提供了 CMA-ES PPO 优化框架的数据处理、日志记录、可视化、
并行执行和 PyTorch 模型管理的综合工具函数集合。

Main Function Categories / 主要功能类别:
    - Data Processing: Subspace combination, monotonic transformation / 数据处理：子空间组合、单调变换
    - TensorBoard: Reading and logging TensorBoard events / TensorBoard：读取和记录 TensorBoard 事件
    - HDF5: Loading and saving HDF5 data files / HDF5：加载和保存 HDF5 数据文件
    - Parallel Execution: Multiprocessing task runner / 并行执行：多进程任务运行器
    - Logging: Fitness recording, result recording / 日志记录：适应度记录、结果记录
    - Visualization: Evaluation curve plotting / 可视化：评估曲线绘制
    - PyTorch Utils: Model loading, gradient clipping, device management / PyTorch 工具：模型加载、梯度裁剪、设备管理

Usage / 使用方法:
    Data processing / 数据处理:
        >>> from utils.utils import combine, make_monotonic_decreasing
        >>> combined = combine(sub_vec, global_vec, [0, 2, 4])

    TensorBoard logging / TensorBoard 日志:
        >>> from utils.utils import log_to_tensorboard
        >>> log_to_tensorboard(fitness_record, './logs', sample_rate=100)

    PyTorch utilities / PyTorch 工具:
        >>> from utils.utils import set_random_seed, clip_grad_norms
        >>> set_random_seed(42)
        >>> grad_norms, clipped = clip_grad_norms(optimizer.param_groups, max_norm=0.1)

Author: CMA-ES PPO Project
Date: 2026-03-15
"""

# =============================================================================
# Standard Library Imports / 标准库导入
# =============================================================================
import os
import math
import random
import time
import inspect
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =============================================================================
# Third-party Imports / 第三方导入
# =============================================================================
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Matplotlib for visualization / Matplotlib 用于可视化
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend / 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

# TensorBoard utilities / TensorBoard 工具
from tensorboard.util import tensor_util
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


# =============================================================================
# Data Processing Utilities / 数据处理工具
# =============================================================================

def combine(
    subspace_vec: np.ndarray,
    globalspace_vec: np.ndarray,
    subspace_index: Optional[List[int]]
) -> np.ndarray:
    """
    Map subspace vectors back to full space.
    将子空间的向量映射回全空间。

    This function combines subspace vectors with a global space vector by
    replacing elements at specified indices. If no indices are provided,
    returns the subspace vectors as-is.
    此函数通过在指定索引处替换元素，将子空间向量与全空间向量组合。
    如果没有提供索引，则按原样返回子空间向量。

    Args:
        subspace_vec (np.ndarray): Subspace vectors, shape (n_samples, n_subspace_dims) / 子空间向量，形状 (n_samples, n_subspace_dims)
        globalspace_vec (np.ndarray): Global space vector for filling, shape (n_global_dims,) / 用于填充的全空间向量，形状 (n_global_dims,)
        subspace_index (Optional[List[int]]): Indices in global space to replace. If None, returns subspace_vec. / 全空间中要替换的索引。如果为 None，返回 subspace_vec

    Returns:
        np.ndarray: Combined vectors, shape (n_samples, n_global_dims) / 组合后的向量，形状 (n_samples, n_global_dims)

    Example:
        >>> sub_vec = np.array([[1, 2], [3, 4]])
        >>> global_vec = np.array([0, 0, 0, 0])
        >>> index = [1, 3]
        >>> combine(sub_vec, global_vec, index)
        array([[0, 1, 0, 2],
               [0, 3, 0, 4]])
    """
    if subspace_index is None:
        return subspace_vec
    else:
        combination = np.tile(globalspace_vec, (len(subspace_vec), 1))
        combination[:, subspace_index] = subspace_vec
        return combination


def make_monotonic_decreasing(arr: List[float]) -> List[float]:
    """
    Ensure the input array is monotonically non-increasing.
    确保输入数组单调不增。

    This function modifies the array in-place to ensure each element is
    less than or equal to the previous one. Used for best-so-far fitness curves.
    此函数就地修改数组，确保每个元素小于或等于前一个元素。
    用于 best-so-far 适应度曲线。

    Args:
        arr (List[float]): Input 1D array (will be modified in-place) / 输入的一维数组（将被就地修改）

    Returns:
        List[float]: The modified array with monotonic non-increasing property / 具有单调不增性质的修改后的数组

    Example:
        >>> data = [5, 3, 4, 2, 1]
        >>> make_monotonic_decreasing(data)
        [5, 3, 3, 2, 1]
    """
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            arr[i + 1] = arr[i]
    return arr


# =============================================================================
# TensorBoard Reading Utilities / TensorBoard 读取工具
# =============================================================================

def read_data_from_tensorboard_file(
    path: str,
    main_tag: str = 'Optimizer/Cost'
) -> List[float]:
    """
    Read data from a single TensorBoard event file.
    从单个 TensorBoard 事件文件读取数据。

    This function reads scalar data from a TensorBoard event file or directory.
    It automatically detects whether the path is a file or directory and
    extracts the specified tag's values.
    此函数从 TensorBoard 事件文件或目录中读取标量数据。
    它自动检测路径是文件还是目录，并提取指定标签的值。

    Args:
        path (str): Path to event file or directory containing event files / 事件文件路径或包含事件文件的目录
        main_tag (str): Scalar tag to read (default: 'Optimizer/Cost') / 要读取的标量标签（默认：'Optimizer/Cost'）

    Returns:
        List[float]: List of scalar values from the specified tag / 来自指定标签的标量值列表

    Raises:
        FileNotFoundError: If the specified path does not exist / 如果指定的路径不存在

    Example:
        >>> values = read_data_from_tensorboard_file(
        ...     './logs/events.out.tfevents.1234567890.hostname',
        ...     main_tag='train/loss'
        ... )
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    # Determine the actual file path / 确定实际的文件路径
    file_path = path
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if 'tfevents' in f]
        if not files:
            print(f"Warning: No event files found in directory {path}")
            return []
        files.sort()
        file_path = os.path.join(path, files[0])

    print(f"Reading TensorBoard file: {file_path}")

    values = []

    try:
        loader = EventFileLoader(file_path)
        for event in loader.Load():
            for value in event.summary.value:
                if value.tag == main_tag:
                    if value.HasField('tensor'):
                        # PyTorch format: data in tensor field / PyTorch 格式：数据在 tensor 字段中
                        val = tensor_util.make_ndarray(value.tensor).item()
                        values.append(val)
                    elif value.HasField('simple_value'):
                        # Legacy format: data in simple_value field / 旧格式：数据在 simple_value 字段中
                        values.append(value.simple_value)
                    else:
                        continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    return values


def read_data_from_tensorboard_folder(
    summary_dir: str
) -> List[List[float]]:
    """
    Read data from all TensorBoard event files in a directory.
    从目录中的所有 TensorBoard 事件文件读取数据。

    This function finds all event files in the specified directory and
    reads data from each one, returning a list of runs.
    此函数在指定目录中查找所有事件文件，并从每个文件中读取数据，
    返回运行列表。

    Args:
        summary_dir (str): Directory containing TensorBoard event files / 包含 TensorBoard 事件文件的目录

    Returns:
        List[List[float]]: List of data from each run, format [[run1_data], [run2_data], ...] / 每次运行的数据列表，格式 [[run1_data], [run2_data], ...]

    Example:
        >>> data = read_data_from_tensorboard_folder('./logs/run_2024-01-01/Summary_All')
        >>> print(f"Loaded {len(data)} runs")
    """
    dataset = []

    if not os.path.exists(summary_dir):
        print(f"[Error] Directory not found: {summary_dir}")
        return []

    event_files = [f for f in os.listdir(summary_dir) if 'tfevents' in f]
    event_files.sort()

    print(f"Found {len(event_files)} event files in Summary_All.")

    for file_name in event_files:
        file_path = os.path.join(summary_dir, file_name)

        try:
            data = read_data_from_tensorboard_file(file_path)
            dataset.append(data)
        except Exception as e:
            print(f"  - Error reading {file_name}: {e}")

    return dataset


# =============================================================================
# HDF5 Data Utilities / HDF5 数据工具
# =============================================================================

def load_running_data(
    file_path: str
) -> Dict[str, np.ndarray]:
    """
    Load HDF5 file generated by running_data_record function.
    加载由 running_data_record 函数生成的 HDF5 文件。

    This function reads an HDF5 file containing optimization run data.
    It automatically handles directory paths and loads all datasets.
    此函数读取包含优化运行数据的 HDF5 文件。它自动处理目录路径
    并加载所有数据集。

    Args:
        file_path (str): Path to HDF5 file or directory containing 'running_data.h5' / HDF5 文件路径或包含 'running_data.h5' 的目录

    Returns:
        Dict[str, np.ndarray]: Dictionary with dataset names as keys and NumPy arrays as values / 以数据集名称为键、NumPy 数组为值的字典

    Raises:
        FileNotFoundError: If the HDF5 file is not found / 如果找不到 HDF5 文件

    Example:
        >>> data = load_running_data('./repository/data/baseline/mmes/cec2013lsgo/maxfes_1E6/f15/2024-01-01/running_data.h5')
        >>> print(data.keys())
    """
    # Path handling / 路径处理
    target_file = file_path

    if os.path.isdir(file_path):
        candidate = os.path.join(file_path, 'running_data.h5')
        if os.path.exists(candidate):
            target_file = candidate
        else:
            print(f"[Warning] Directory found but 'running_data.h5' is missing in: {file_path}")

    if not os.path.exists(target_file):
        raise FileNotFoundError(f"HDF5 file not found: {target_file}")

    print(f"Loading data from: {target_file} ...")

    loaded_data = {}

    try:
        with h5py.File(target_file, 'r') as f:
            for key in f.keys():
                data_matrix = f[key][:]
                loaded_data[key] = data_matrix
                print(f"  - Loaded '{key}': shape={data_matrix.shape}")
    except Exception as e:
        print(f"[Error] Failed to read HDF5: {e}")
        return {}

    return loaded_data


# =============================================================================
# Parallel Execution Utilities / 并行执行工具
# =============================================================================

def run_parallel_task(
    target_func: Callable,
    parallel_num: int,
    *args: Any,
    **kwargs: Any
) -> Tuple[List[Any], float]:
    """
    Generic parallel task execution wrapper.
    通用的并行任务执行包装器。

    This function executes a target function in parallel using multiple processes.
    It automatically handles parameter passing and collects results.
    此函数使用多进程并行执行目标函数。它自动处理参数传递并收集结果。

    Args:
        target_func (Callable): Function to execute in parallel / 要并行执行的函数
        parallel_num (int): Number of parallel executions / 并行执行次数
        *args: Positional arguments to pass to target_func / 传递给目标函数的位置参数
        **kwargs: Keyword arguments to pass to target_func / 传递给目标函数的关键字参数

    Returns:
        Tuple[List[Any], float]: (results_record, avg_time) - Results from each execution and average time / (结果记录, 平均时间) - 每次执行的结果和平均时间

    Note:
        The target function's last parameter is assumed to receive the loop index.
        The function should return a tuple of (time, result) for timing information.
        目标函数的最后一个参数假设接收循环索引。
        函数应返回 (time, result) 元组以提供计时信息。

    Example:
        >>> def sample_task(x, y, index):
        ...     time.sleep(1)
        ...     return (1.0, x + y + index)
        >>> results, avg_time = run_parallel_task(sample_task, 5, 10, 20)
        >>> print(f"Average Time: {avg_time}, Results: {results}")
    """
    # Analyze function signature to find index parameter / 分析函数签名以找到索引参数
    sig = inspect.signature(target_func)
    params = list(sig.parameters.keys())

    if not params:
        raise ValueError("目标函数没有参数，无法传递循环索引")

    index_param_name = params[-1]

    # Use spawn to avoid CUDA/fork conflicts / 使用 spawn 避免 CUDA/fork 冲突
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(mp_context=ctx) as executor:
        futures = []
        for i in range(parallel_num):
            current_kwargs = kwargs.copy()
            current_kwargs[index_param_name] = i
            futures.append(executor.submit(target_func, *args, **current_kwargs))

        total_time = 0
        results_record = []

        for future in futures:
            res = future.result()

            if isinstance(res, (tuple, list)) and len(res) >= 2:
                results_record.append(res[0])
                total_time += res[1]
            else:
                results_record.append(res)

        avg_time = total_time / parallel_num if parallel_num > 0 else 0

        return results_record, avg_time


# =============================================================================
# Recording Utilities / 记录工具
# =============================================================================

class FitnessRecorder:
    """
    Wrapper for recording objective function evaluation values.
    用于记录目标函数评估值的包装类。

    This class wraps an objective function and records all returned fitness
    values for tracking optimization progress.
    此类包装目标函数并记录所有返回的适应度值，用于跟踪优化进度。

    Attributes:
        fun (Callable): The wrapped objective function / 包装的目标函数
        fitness_record (List[float]): Recorded fitness values / 记录的适应度值

    Example:
        >>> def sample_function(x):
        ...     return x**2
        >>> recorder = FitnessRecorder(sample_function)
        >>> result = recorder(3)
        >>> print(recorder.fitness_record)  # [9]
    """

    def __init__(self, fun: Callable[[np.ndarray], List[float]]) -> None:
        """
        Initialize the fitness recorder.
        初始化适应度记录器。

        Args:
            fun (Callable): Objective function that returns a list of fitness values / 返回适应度值列表的目标函数
        """
        self.fun = fun
        self.fitness_record: List[float] = []

    def __call__(self, x: np.ndarray) -> List[float]:
        """
        Call the wrapped function and record results.
        调用包装的函数并记录结果。

        Args:
            x (np.ndarray): Input to the objective function / 目标函数的输入

        Returns:
            List[float]: Fitness values from the objective function / 来自目标函数的适应度值
        """
        fitness = self.fun(x)
        self.fitness_record.extend(fitness)
        return fitness


# Backward compatibility alias / 向后兼容别名
fun_record = FitnessRecorder


def log_to_tensorboard(
    fitness_record: List[List[float]],
    log_dir: str,
    sample_rate: int = 100,
    main_tag: str = 'Optimizer/Cost'
) -> None:
    """
    Log optimization convergence data to TensorBoard.
    将优化收敛数据记录到 TensorBoard。

    This function creates TensorBoard logs with individual runs, aggregated
    statistics, and standard deviation curves.
    此函数创建 TensorBoard 日志，包含单独运行、聚合统计和标准差曲线。

    Args:
        fitness_record (List[List[float]]): List of runs, each run is a list of fitness values / 运行列表，每次运行是适应度值列表
        log_dir (str): Root directory for logs (should include timestamp) / 日志的根目录（应包含时间戳）
        sample_rate (int): Record every Nth generation (default: 100) / 每隔 N 代记录一次（默认：100）
        main_tag (str): Tag name for TensorBoard (default: 'Optimizer/Cost') / TensorBoard 的标签名称（默认：'Optimizer/Cost'）

    Note:
        Creates three subdirectories:
        - Individual run logs in run_*/ / run_*/ 中的单独运行日志
        - Aggregated logs in Summary_All/ / Summary_All/ 中的聚合日志
        - Statistics in Summary_Stats/ / Summary_Stats/ 中的统计信息

    Example:
        >>> log_to_tensorboard(
        ...     fitness_record,
        ...     './logs/run_2024-01-01_12-00-00',
        ...     sample_rate=50,
        ...     main_tag='Optimizer/Cost'
        ... )
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    agg_dir = os.path.join(log_dir, 'Summary_All')
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)

    print(f"Starting TensorBoard logging to: {log_dir}")
    print(f"Total runs: {len(fitness_record)} | Sample rate: {sample_rate}")

    # Phase 1: Individual runs and aggregation / 阶段 1：单独运行和聚合
    for run_index, single_run_data in enumerate(fitness_record):
        individual_dir = os.path.join(log_dir, f'run_{run_index}')
        if not os.path.exists(individual_dir):
            os.makedirs(individual_dir)
        writer_individual = SummaryWriter(log_dir=individual_dir)
        writer_agg = SummaryWriter(log_dir=agg_dir)

        for fe, val in enumerate(single_run_data):
            if fe % sample_rate == 0:
                writer_individual.add_scalar(main_tag, val, global_step=fe)
                writer_agg.add_scalar(main_tag, val, global_step=fe)

        writer_individual.close()
        writer_agg.close()

    # Phase 2: Standard deviation curve / 阶段 2：标准差曲线
    if not fitness_record:
        print("Warning: fitness_record is empty.")
        return

    min_len = min(len(r) for r in fitness_record)
    records_array = np.array([r[:min_len] for r in fitness_record])
    std_curve = np.std(records_array, axis=0)

    stats_dir = os.path.join(log_dir, 'Summary_Stats')
    writer_stats = SummaryWriter(log_dir=stats_dir)
    std_tag = f"{main_tag}_Std"

    for fe in range(0, min_len):
        if fe % sample_rate == 0:
            writer_stats.add_scalar(std_tag, std_curve[fe], global_step=fe)

    writer_stats.close()

    print(f"Done. Logs ready at: {log_dir}")


def result_record(
    data: Dict[str, Any],
    output_path: str,
    record_FEs_list: List[int]
) -> None:
    """
    Record optimization algorithm evaluation values and runtime.
    记录优化算法的评估值和运行时间。

    This function writes algorithm performance statistics to a formatted text file,
    including mean fitness at specified evaluation points and final results.
    此函数将算法性能统计信息写入格式化的文本文件，
    包括指定评估点的平均适应度和最终结果。

    Args:
        data (Dict[str, Any]): Dictionary with algorithm names as keys. Each key should have
                               a corresponding '{name}_time' key for runtime data.
                               / 以算法名称为键的字典。每个键应有对应的 '{name}_time' 键用于运行时数据。
                               Format: {'Algorithm1': [[run1_data], [run2_data], ...], 'Algorithm1_time': [time1, time2, ...]}
        output_path (str): Directory path for saving results / 保存结果的目录路径
        record_FEs_list (List[int]): Evaluation points to record (e.g., [1000, 10000, 100000]) / 要记录的评估点（例如：[1000, 10000, 100000]）

    Note:
        Creates a 'result_record.txt' file with formatted output including:
        创建 'result_record.txt' 文件，包含格式化输出：
        - Mean fitness at each specified FE point / 每个指定 FE 点的平均适应度
        - Standard deviation at each point / 每个点的标准差
        - Final results / 最终结果
        - Average runtime / 平均运行时间

    Example:
        >>> data = {
        ...     'Algorithm1': [[...], [...], ...],
        ...     'Algorithm1_time': [12.5, 13.0, ...],
        ...     'Algorithm2': [[...], [...], ...],
        ...     'Algorithm2_time': [10.0, 11.5, ...]
        ... }
        >>> result_record(data, './results/', [1000, 10000, 100000])
    """
    # Preprocessing / 预处理
    record_FEs_list = sorted([int(x) for x in record_FEs_list])

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_file_path = os.path.join(output_path, "result_record.txt")

    HEADER_FMT = "{:<20}{:<20}{:<25}{:<25}{:<25}{:<25}\n"
    DATA_FMT = "{:<20}{:<20}{:<25.6f}{:<25.6e}{:<25.6f}{:<25.6e}\n"
    FINAL_FMT = "{:<15}{:<25}{:<25.6f}{:<25.6e}{:<25.6f}{:<25.6e}\n"
    SEPARATOR = "-" * 140 + "\n"

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            def dual_log(text: str) -> None:
                f.write(text)
                print(text, end='')

            dual_log(SEPARATOR)
            dual_log(HEADER_FMT.format(
                "Algorithm", "Record Point", "Mean Fitness", "Mean Sci", "Std Dev", "Std Sci"
            ))
            dual_log(SEPARATOR)

            for algorithm_name, runs_data in data.items():
                if algorithm_name.endswith("_time"):
                    continue

                try:
                    # Filter out empty runs / 过滤掉空的运行
                    valid_runs = [run for run in runs_data if run is not None and len(run) > 0]
                    if not valid_runs:
                        print(f"[Warning] No valid data for {algorithm_name}")
                        continue

                    max_len = max(len(run) for run in valid_runs)
                    n_runs = len(valid_runs)
                    padded_data = np.full((n_runs, max_len), np.nan)

                    for i, run in enumerate(valid_runs):
                        length = min(len(run), max_len)
                        run_array = np.array(run[:length], dtype=np.float64)
                        monotonic_run = np.minimum.accumulate(run_array)
                        padded_data[i, :length] = monotonic_run

                except Exception as e:
                    print(f"[Error] Processing data for {algorithm_name}: {e}")
                    continue

                # Check for valid data before computing statistics / 计算统计前检查有效数据
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_curve = np.nanmean(padded_data, axis=0)
                    std_curve = np.nanstd(padded_data, axis=0)

                    # Replace NaN in mean_curve with forward fill / 用前向填充替换 mean_curve 中的 NaN
                    if np.any(np.isnan(mean_curve)):
                        mask = np.isnan(mean_curve)
                        # Use forward fill / 使用前向填充
                        idx = np.where(~mask)[0]
                        if len(idx) > 0:
                            mean_curve[mask] = np.interp(
                                np.flatnonzero(mask),
                                np.flatnonzero(~mask),
                                mean_curve[~mask],
                                left=np.nan,
                                right=np.nan
                            )

                time_key = f"{algorithm_name}_time"
                avg_time = "N/A"
                if time_key in data:
                    times = data[time_key]
                    if isinstance(times, (list, np.ndarray)) and len(times) > 0:
                        avg_time = np.mean(times)
                    elif isinstance(times, (int, float)):
                        avg_time = times

                dual_log(f"Algorithm: {algorithm_name}\n")

                for fe_point in record_FEs_list:
                    idx = fe_point - 1 if fe_point > 0 else 0
                    fe_display = f"{fe_point:.1e}"

                    if idx < max_len:
                        fitness_val = mean_curve[idx]
                        std_val = std_curve[idx]

                        if np.isnan(fitness_val):
                            f.write(f"{'':<20}{fe_point:<20}{'N/A':<25}{'N/A':<25}{'N/A':<25}{'N/A':<25}\n")
                        else:
                            f.write(DATA_FMT.format("", fe_display, fitness_val, fitness_val, std_val, std_val))
                    else:
                        f.write(f"{'':<20}{fe_display:<20}{'Exceeded':<25}{'-':<25}{'-':<25}{'-':<25}\n")

                final_val = mean_curve[-1]
                final_std = std_curve[-1]
                scale_str = f"{max_len:.0e}".replace("+0", "").replace("+", "")
                final_label = f"Final({scale_str}|{max_len:,})"

                if not np.isnan(final_val):
                    dual_log(FINAL_FMT.format("", final_label, final_val, final_val, final_std, final_std))

                if avg_time != "N/A":
                    dual_log(f"{'':<15}{'Avg Time(s)':<25}{avg_time:<25.6f}\n")

                dual_log(SEPARATOR)

        print(f"Evaluation result records successfully saved to: {output_file_path}")

    except IOError as e:
        print(f"Error writing to file {output_file_path}: {e}")


def running_data_record(
    output_data: Dict[str, List[List[float]]],
    file_path: str
) -> None:
    """
    Save optimization run data as HDF5 format.
    将优化运行数据保存为 HDF5 格式。

    This function saves algorithm run data to an HDF5 file with gzip compression.
    It automatically aligns data from multiple runs and filters time keys.
    此函数将算法运行数据保存到使用 gzip 压缩的 HDF5 文件中。
    它自动对齐来自多次运行的数据并过滤时间键。

    Args:
        output_data (Dict[str, List[List[float]]]): Dictionary with algorithm names as keys
                                                   and lists of run data as values.
                                                   / 以算法名称为键、运行数据列表为值的字典。
                                                   Format: {'Algorithm1': [[run1_data], [run2_data], ...], ...}
        file_path (str): Directory path for saving the HDF5 file / 保存 HDF5 文件的目录路径

    Note:
        - Creates 'running_data.h5' in the specified directory / 在指定目录中创建 'running_data.h5'
        - Filters out keys containing '_time' / 过滤掉包含 '_time' 的键
        - Aligns runs to minimum length for matrix storage / 将运行对齐到最小长度以进行矩阵存储

    Example:
        >>> output_data = {
        ...     'Algorithm1': [[...], [...], ...],
        ...     'Algorithm2': [[...], [...], ...]
        ... }
        >>> running_data_record(output_data, './results/h5_data/')
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    h5_file = os.path.join(file_path, 'running_data.h5')

    print(f"Running data start saving to {file_path} ...")

    with h5py.File(h5_file, 'w') as f:
        for key, runs_list in output_data.items():
            if "_time" in key or not runs_list:
                continue

            try:
                min_len = min(len(r) for r in runs_list)

                if min_len == 0:
                    print(f"  [Skip] {key}: Runs are empty.")
                    continue

                data_matrix = np.array([r[:min_len] for r in runs_list], dtype=np.float64)
                f.create_dataset(key, data=data_matrix, compression='gzip')

                print(f"  - Saved '{key}': shape={data_matrix.shape}")

            except Exception as e:
                print(f"  [Error] Failed to save {key}: {e}")

    print(f"Running data saved to {file_path}.")


# =============================================================================
# Visualization Utilities / 可视化工具
# =============================================================================

def plot_evaluation_curve_best_so_far(
    data: Dict[str, List[List[float]]],
    output_path: str,
    maxfes: int,
    figsize: Tuple[float, float] = (3.5, 2.16),
    font_size: int = 8,
    log_scale: bool = True,
    show_variance: bool = True,
    eps: float = 1e-12
) -> None:
    """
    Plot Best-so-Far evaluation curves for different algorithms (Nature style).
    绘制不同算法的 Best-so-Far 评估曲线（Nature 风格）。

    This function creates publication-quality evaluation curves with Nature journal
    style formatting, including mean curves with variance bands.
    此函数创建具有 Nature 期刊风格格式的出版质量评估曲线，
    包括带有方差带的均值曲线。

    Args:
        data (Dict[str, List[List[float]]]): Dictionary with algorithm names as keys and run lists as values / 以算法名称为键、运行列表为值的字典
        output_path (str): Directory path for saving figures / 保存图形的目录路径
        maxfes (int): Maximum function evaluations for x-axis scaling / x 轴缩放的最大函数评估次数
        figsize (Tuple[float, float]): Figure size in inches (default: (3.5, 2.16) for IEEE double-column) / 图形大小（英寸，默认：(3.5, 2.16) 用于 IEEE 双栏）
        font_size (int): Base font size in points (default: 8) / 基础字体大小（点，默认：8）
        log_scale (bool): Use logarithmic y-axis (default: True) / 使用对数 y 轴（默认：True）
        show_variance (bool): Show variance bands (default: True) / 显示方差带（默认：True）
        eps (float): Small value to prevent log errors (default: 1e-12) / 防止 log 错误的小值（默认：1e-12）

    Note:
        Saves both PDF (vector) and PNG (raster) formats.
        The figure uses Nature journal formatting guidelines.
        保存 PDF（矢量）和 PNG（光栅）两种格式。
        图形使用 Nature 期刊格式指南。

    Example:
        >>> plot_evaluation_curve_best_so_far(
        ...     data={'Algorithm1': [[...], [...]], 'Algorithm2': [[...], [...]]},
        ...     output_path='./figures/',
        ...     maxfes=1000000
        ... )
    """
    # Global style settings (Nature style) / 全局样式设置（Nature 风格）
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size - 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'svg.fonttype': 'none',
    })

    # Nature classic color palette (NPG) / Nature 经典配色（NPG）
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']

    fig, ax = plt.subplots(figsize=figsize)
    color_idx = 0

    for algorithm, runs in data.items():
        if "_time" in algorithm or len(runs) == 0:
            continue

        # Data alignment / 数据对齐
        min_len = min(len(r) for r in runs)

        processed_runs = []
        for run in runs:
            r = np.array(run[:min_len])
            r = np.minimum.accumulate(r)  # Force monotonically decreasing / 强制单调递减
            processed_runs.append(r)

        run_matrix = np.array(processed_runs)
        run_matrix = np.clip(run_matrix, eps, None)  # Prevent log errors / 防止 log 错误

        # Generate x-axis / 生成 x 轴
        x_axis = np.arange(min_len)

        # Compute statistics / 计算统计量
        means = np.mean(run_matrix, axis=0)

        curr_color = colors[color_idx % len(colors)]

        # Plot main curve / 绘制主曲线
        ax.plot(x_axis, means, label=algorithm, color=curr_color, alpha=0.9)

        # Plot variance band (Log-Normal Error Band) / 绘制方差带（对数正态误差带）
        if show_variance and len(runs) > 1:
            log_vals = np.log10(run_matrix)
            std_log = np.std(log_vals, axis=0)
            factor = 10 ** std_log

            upper = means * factor
            lower = means / factor

            ax.fill_between(x_axis, lower, upper, color=curr_color, alpha=0.2, linewidth=0)

        color_idx += 1

    # Axis formatting / 坐标轴格式化
    if log_scale:
        ax.set_yscale("log")
        locmin = LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.get_offset_text().set_fontsize(font_size - 1)

    ax.set_xlabel("FEs")
    ax.set_ylabel("Objective Value")

    # Remove top and right spines / 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Enable light grid / 启用浅色网格
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    # Legend / 图例
    ax.legend(frameon=False, loc='best')

    plt.tight_layout()

    # Save figures / 保存图形
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    pdf_path = os.path.join(output_path, "evaluation_curves_best_so_far.pdf")
    png_path = os.path.join(output_path, "evaluation_curves_best_so_far.png")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=600)

    print(f"Figures saved to:\n  {pdf_path}\n  {png_path}")
    plt.close()


# =============================================================================
# PyTorch Utilities / PyTorch 工具
# =============================================================================

def torch_load_cpu(load_path: str) -> Any:
    """
    Load a PyTorch checkpoint on CPU to avoid CUDA memory issues.
    在 CPU 上加载 PyTorch 检查点以避免 CUDA 内存问题。

    This function loads a model checkpoint to CPU, which is useful when
    the saved model was trained on GPU but you want to load it on CPU
    or a different GPU configuration.
    此函数在 CPU 上加载模型检查点，这在保存的模型在 GPU 上训练
    但你想在 CPU 或不同的 GPU 配置上加载它时很有用。

    Args:
        load_path (str): Path to the checkpoint file / 检查点文件路径

    Returns:
        Any: Loaded checkpoint object / 加载的检查点对象

    Example:
        >>> checkpoint = torch_load_cpu('./models/epoch-10.pt')
        >>> model.load_state_dict(checkpoint['model_state'])
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)


def get_inner_model(model: nn.Module) -> nn.Module:
    """
    Get the inner model wrapped by DataParallel or DistributedDataParallel.
    获取被 DataParallel 或 DistributedDataParallel 包装的内部模型。

    When using PyTorch's DataParallel or DistributedDataParallel wrappers,
    this function extracts the actual model for parameter access.
    当使用 PyTorch 的 DataParallel 或 DistributedDataParallel 包装器时，
    此函数提取实际模型以访问参数。

    Args:
        model (nn.Module): Potentially wrapped model / 可能被包装的模型

    Returns:
        nn.Module: The inner unwrapped model / 内部未包装的模型

    Example:
        >>> model = MyModel()
        >>> parallel_model = nn.DataParallel(model)
        >>> inner = get_inner_model(parallel_model)
        >>> # inner is now the original model
    """
    return model.module if isinstance(model, (DataParallel, DDP)) else model


def move_to(var: Any, device: torch.device) -> Any:
    """
    Recursively move tensors or tensors in containers to a specified device.
    递归地将张量或容器中的张量移动到指定设备。

    This function handles individual tensors as well as dictionaries
    containing tensors.
    此函数处理单个张量以及包含张量的字典。

    Args:
        var (Any): Tensor or dictionary of tensors / 张量或张量字典
        device (torch.device): Target device / 目标设备

    Returns:
        Any: Tensor(s) moved to the specified device / 移动到指定设备的张量

    Example:
        >>> device = torch.device('cuda:0')
        >>> tensor = torch.randn(10, 10)
        >>> tensor = move_to(tensor, device)
        >>> # tensor is now on GPU
    """
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def move_to_cuda(var: Any, device: Union[int, str, torch.device]) -> Any:
    """
    Recursively move tensors or tensors in containers to CUDA.
    递归地将张量或容器中的张量移动到 CUDA。

    This function is a convenience wrapper for move_to with CUDA-specific
    handling. Use move_to for more general device handling.
    此函数是 move_to 的便捷包装器，具有 CUDA 特定的处理。
    对于更通用的设备处理，请使用 move_to。

    Args:
        var (Any): Tensor or dictionary of tensors / 张量或张量字典
        device (Union[int, str, torch.device]): CUDA device identifier / CUDA 设备标识符

    Returns:
        Any: Tensor(s) moved to CUDA / 移动到 CUDA 的张量

    Example:
        >>> tensor = torch.randn(10, 10)
        >>> tensor = move_to_cuda(tensor, 0)  # Move to cuda:0
    """
    if isinstance(var, dict):
        return {k: move_to_cuda(v, device) for k, v in var.items()}
    return var.cuda(device)


def clip_grad_norms(
    param_groups: List[Dict[str, Any]],
    max_norm: float = math.inf
) -> Tuple[List[float], List[float]]:
    """
    Clip gradients by norm and return norms before and after clipping.
    按范数裁剪梯度并返回裁剪前后的范数。

    This function clips gradients for all parameter groups to prevent
    exploding gradients during training.
    此函数裁剪所有参数组的梯度以防止训练期间梯度爆炸。

    Args:
        param_groups (List[Dict[str, Any]]): Parameter groups from optimizer / 来自优化器的参数组
        max_norm (float): Maximum norm for gradient clipping (default: inf = no clipping) / 梯度裁剪的最大范数（默认：inf = 不裁剪）

    Returns:
        Tuple[List[float], List[float]]: (grad_norms, grad_norms_clipped) - Gradient norms before and after clipping / (梯度范数, 裁剪后的梯度范数) - 裁剪前后的梯度范数

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> grad_norms, clipped = clip_grad_norms(optimizer.param_groups, max_norm=0.1)
        >>> print(f"Original: {grad_norms}, Clipped: {clipped}")
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def set_random_seed(seed: Optional[int] = None) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    为所有库的随机数生成器设置种子以确保可复现性。

    This function sets seeds for Python's random, NumPy, PyTorch (CPU and CUDA),
    ensuring reproducible results across different runs.
    此函数为 Python 的 random、NumPy、PyTorch（CPU 和 CUDA）设置种子，
    确保不同运行之间的可复现结果。

    Args:
        seed (Optional[int]): Random seed value. If None, uses random initialization. / 随机种子值。如果为 None，使用随机初始化。

    Note:
        For CUDA, additional settings like torch.backends.cudnn.deterministic = True
        may be needed for full reproducibility.
        对于 CUDA，可能需要额外的设置如 torch.backends.cudnn.deterministic = True
        以实现完全可复现性。

    Example:
        >>> set_random_seed(42)
        >>> # All random operations will now be reproducible
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        random.seed(None)
        np.random.seed(None)
        torch.manual_seed(int(time.time()))
        torch.cuda.manual_seed(int(time.time()))
        torch.cuda.manual_seed_all(int(time.time()))
