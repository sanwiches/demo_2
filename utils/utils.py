"""
General Utility Functions / 通用工具函数模块

This module provides common utility functions used throughout the WCC project,
including PyTorch model operations, gradient clipping, random seed setting,
and optimization-related utilities.

本模块提供 WCC 项目中使用的通用工具函数，包括 PyTorch 模型操作、梯度裁剪、
随机种子设置和优化相关工具。

Main Categories / 主要类别:
    - Model Operations: torch_load_cpu, get_inner_model, move_to
    - Gradient Management: clip_grad_norms
    - Randomness: set_random_seed
    - Data Processing: partition_p_and_s, farthest_pair_double_sweep
    - Optimization: FitnessRecorder

Author: WCC Project
Date: 2026-03-11
"""

import math
import os
import random
import time
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP


# =============================================================================
# Model Operations / 模型操作
# =============================================================================

# =======================================================================
# 函数：在 CPU 上加载 PyTorch 检查点
# =======================================================================
def torch_load_cpu(load_path: str) -> Any:
    """
    在 CPU 上加载 PyTorch 检查点以避免 CUDA 内存问题。

    主要功能包括：
    1. 将模型/状态字典首先加载到 CPU 上。
    2. 可以根据需要将模型移动到 GPU。
    3. 适用于加载大型模型或 CUDA 内存受限的情况。

    参数:
        load_path (str): 检查点文件路径。

    用法示例:
        checkpoint = torch_load_cpu('model.pt')
        model.load_state_dict(checkpoint['model_state'])

    结果:
        返回加载的检查点（模型状态字典、优化器状态等）。
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)


# =======================================================================
# 函数：从 DataParallel/DDP 包装器中提取内部模型
# =======================================================================
def get_inner_model(model: Union[DataParallel, DDP, torch.nn.Module]) -> torch.nn.Module:
    """
    从 DataParallel 或 DistributedDataParallel 包装器中提取内部模型。

    主要功能包括：
    1. 当使用 PyTorch 的 DataParallel 或 DDP 时，实际模型存储在 .module 属性中。
    2. 此函数将其解包以直接访问。

    参数:
        model: 包装或未包装的 PyTorch 模型。

    用法示例:
        model = MyModel()
        parallel_model = DataParallel(model)
        inner = get_inner_model(parallel_model)  # 返回 model

    结果:
        返回底层的 nn.Module。
    """
    if isinstance(model, (DataParallel, DDP)):
        return model.module
    return model


# =======================================================================
# 函数：将张量或张量集合移动到指定设备
# =======================================================================
def move_to(
    var: Union[torch.Tensor, Dict, List, Tuple],
    device: Union[str, torch.device]
) -> Union[torch.Tensor, Dict, List, Tuple]:
    """
    将张量或张量集合移动到指定设备。

    主要功能包括：
    1. 支持单个张量、字典、列表或元组的移动。
    2. 递归处理嵌套的数据结构。
    3. 保留数据结构的类型。

    参数:
        var: 要移动的张量或张量集合。
        device: 目标设备（如 'cpu', 'cuda', 或 torch.device 对象）。

    用法示例:
        tensor = torch.randn(3, 3)
        tensor_gpu = move_to(tensor, 'cuda')
        data = {'a': torch.randn(2), 'b': [torch.randn(3)]}
        data_gpu = move_to(data, 'cuda')

    结果:
        返回移动到目标设备后的张量或张量集合。
    """
    if isinstance(var, torch.Tensor):
        return var.to(device)
    elif isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, (list, tuple)):
        return type(var)(move_to(v, device) for v in var)
    return var


# =============================================================================
# Gradient Management / 梯度管理
# =============================================================================

# =======================================================================
# 函数：裁剪多个参数组的梯度范数
# =======================================================================
def clip_grad_norms(
    param_groups: List[Dict],
    max_norm: float = 1.0
) -> float:
    """
    裁剪多个参数组的梯度范数。

    主要功能包括：
    1. 遍历所有参数组，计算梯度范数。
    2. 对梯度进行裁剪，防止梯度爆炸。
    3. 返回总的梯度范数。

    参数:
        param_groups (list): 参数组列表，通常来自 optimizer.param_groups。
        max_norm (float): 最大梯度范数（默认 1.0）。

    用法示例:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # ... 计算梯度 ...
        grad_norm = clip_grad_norms(optimizer.param_groups, max_norm=1.0)

    结果:
        返回裁剪前的总梯度范数。
    """
    grad_norms = []
    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm(2))

    total_norm = math.sqrt(sum(g**2 for g in grad_norms))

    if max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for group in param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)

    return total_norm


# =============================================================================
# Randomness / 随机性控制
# =============================================================================

# =======================================================================
# 函数：设置随机种子以确保可重复性
# =======================================================================
def set_random_seed(seed: Optional[int] = None) -> None:
    """
    设置 Python、NumPy 和 PyTorch 的随机种子。

    主要功能包括：
    1. 设置 random、numpy 和 torch 的随机种子。
    2. 支持 CUDA 随机种子设置。
    3. 如果不提供种子，则使用基于时间的种子。

    参数:
        seed (int, optional): 随机种子值。如果为 None，则使用当前时间。

    用法示例:
        set_random_seed(42)  # 固定种子
        set_random_seed()    # 基于时间的种子

    注意:
        由于 CUDA 和某些 PyTorch 函数中的非确定性操作，设置种子不能保证完全可重复性。
        考虑使用 torch.use_deterministic_algorithms(True) 进行更严格的控制。
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


# =============================================================================
# Data Processing / 数据处理
# =============================================================================

# =======================================================================
# 函数：根据 p-file 和 s-file 划分维度索引
# =======================================================================
def partition_p_and_s(
    p_file_path: str,
    s_file_path: str,
    overlap: int = 0
) -> List[List[int]]:
    """
    根据 p 文件和 s 文件划分维度索引，支持可选重叠。

    主要功能包括：
    1. 读取 p 文件（包含维度索引）和 s 文件（包含子空间大小）。
    2. 将维度划分为子空间。
    3. 支持子空间之间的可选重叠。

    文件格式:
        - p-file: 逗号分隔的维度索引（如 "1,2,3,4,5"）
        - s-file: 空格分隔的子空间大小（如 "3 3 4"）

    参数:
        p_file_path (str): 包含维度索引的 p 文件路径。
        s_file_path (str): 包含子空间大小的 s 文件路径。
        overlap (int): 连续子空间之间的重叠维度数（默认 0）。

    用法示例:
        # p-file: "1,2,3,4,5,6,7,8,9,10"
        # s-file: "3 3 4"
        groups = partition_p_and_s('p.txt', 's.txt', overlap=1)
        # 返回: [[0,1,2], [2,3,4], [4,5,6,7]]（转换为 0-based 索引）

    结果:
        返回子空间列表，每个子空间是维度索引的列表。

    注意:
        函数将 p 文件中的从 1 开始的索引转换为从 0 开始的索引。
    """

    def _read_file(file_path: str) -> str:
        """读取文件内容并去除空白"""
        with open(file_path, "r") as file:
            return file.read().strip()

    def _parse_p_values(content: str) -> List[int]:
        """解析逗号分隔的维度索引"""
        return list(map(int, content.split(",")))

    def _parse_s_values(content: str) -> List[int]:
        """解析空格分隔的子空间大小"""
        return [int(float(x)) for x in content.split()]

    def _partition_with_overlap(
        p_values: List[int],
        s_values: List[int],
        overlap: int
    ) -> List[List[int]]:
        """根据 s 值和重叠划分 p 值"""
        partitioned = []
        start_idx = 0

        for size in s_values:
            # 对非第一个分区应用重叠
            if partitioned and overlap > 0:
                start_idx = start_idx - overlap
            end_idx = start_idx + size
            partitioned.append(p_values[start_idx:end_idx])
            start_idx = end_idx

        return partitioned

    # 读取和解析文件
    p_content = _read_file(p_file_path)
    s_content = _read_file(s_file_path)

    p_values = _parse_p_values(p_content)
    p_values = [x - 1 for x in p_values]  # 转换为 0-based
    s_values = _parse_s_values(s_content)

    return _partition_with_overlap(p_values, s_values, overlap)


# =======================================================================
# 函数：使用双重扫描算法估计最远点对距离
# =======================================================================
def farthest_pair_double_sweep(
    X: np.ndarray,
    n_starts: int = 64,
    seed: int = 42
) -> float:
    """
    使用双重扫描算法估计最远点对距离。

    主要功能包括：
    1. 通过反复找到距离随机起点最远的点来估计点云的直径。
    2. 比计算所有成对距离快得多。
    3. 提供近似结果。

    算法:
        1. 采样随机点 A
        2. 找到距离 A 最远的点 B
        3. 找到距离 B 最远的点 C
        4. 距离(B, C) 即为估计值
        5. 重复并保留最大值

    参数:
        X (np.ndarray): 输入点云，形状 (n, d)，n 为点数，d 为维度。
        n_starts (int): 近似的随机起点数量（默认 64）。
        seed (int): 用于可重复性的随机种子（默认 42）。

    用法示例:
        points = np.random.randn(1000, 10)
        diameter = farthest_pair_double_sweep(points)

    结果:
        返回 X 中任意两点之间的估计最大距离。
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_dist = -1.0

    def _farthest_from(idx: int) -> Tuple[int, float]:
        """返回距离 X[idx] 最远的点的索引和距离"""
        v = X - X[idx]
        d2 = np.einsum('ij,ij->i', v, v)
        j = int(np.argmax(d2))
        return j, float(np.sqrt(d2[j]))

    for _ in range(min(n_starts, n)):
        a = int(rng.integers(0, n))
        b, _ = _farthest_from(a)
        _, dist = _farthest_from(b)
        if dist > best_dist:
            best_dist = dist

    return best_dist


# =============================================================================
# Optimization Utilities / 优化工具
# =============================================================================

# =======================================================================
# 类：用于跟踪优化进度的适应度记录器
# =======================================================================
class FitnessRecorder:
    """
    用于跟踪优化进度的适应度记录器。

    此类包装目标函数并记录所有适应度评估和评估的个体。
    它处理 CC（子空间优化）和 NDA（全空间优化）两种模式。

    对于 CC 模式：通过将子空间解与背景全空间解组合来评估。
    对于 NDA 模式：直接评估全空间解。

    属性:
        fun: 包装的目标函数。
        best: 当前最优解（全空间）。
        dims: 子空间变量索引（仅 CC 模式）。
        info: 问题信息字典。
        is_cc: 是否为 CC 模式。
        fitness_record: 适应度值历史记录。
        individual_record: 评估的个体历史记录。

    参数:
        fun (callable): 要评估的目标函数。
        best (np.ndarray): 当前最优解（全空间）。
        dims (list, optional): 子问题的变量索引。NDA 模式为 None。
        info (dict): 问题信息字典，包含键：'lower', 'upper', 'dimension'。
        is_cc (bool): 是否为 CC 模式（子空间优化，默认 True）。

    用法示例:
        recorder = FitnessRecorder(func, best_x, dims=[0,1,2], info=..., is_cc=True)
        fitness = recorder(population)  # 评估并记录
        print(recorder.fitness_record)
    """

    def __init__(
        self,
        fun: callable,
        best: np.ndarray,
        dims: Optional[List[int]],
        info: Dict[str, Any],
        is_cc: bool = True
    ):
        self.fun = fun
        self.best = best
        self.dims = dims
        self.info = info
        self.is_cc = is_cc
        self.fitness_record: List[float] = []
        self.individual_record: List[np.ndarray] = []

    def _combine(
        self,
        sub_solution: np.ndarray,
        background: np.ndarray,
        dims: Optional[List[int]]
    ) -> np.ndarray:
        """
        将子空间解与全空间背景组合。

        参数:
            sub_solution: 子空间解，形状 (n, d_sub)。
            background: 背景全空间解，形状 (d,)。
            dims: 子空间变量索引。None 直接返回 sub_solution。

        结果:
            返回组合后的解，形状 (n, d)。
        """
        if dims is None:
            return sub_solution
        combined = np.tile(background, (len(sub_solution), 1))
        combined[:, dims] = sub_solution
        return combined

    def __call__(self, solutions_batch: np.ndarray) -> np.ndarray:
        """
        评估一批解的适应度。

        参数:
            solutions_batch: 要评估的解批次。

        结果:
            返回每个解的适应度值。
        """
        if self.is_cc:
            transformed = self._combine(solutions_batch, self.best, self.dims)
            self.individual_record.extend(transformed.copy())
            fitness = self.fun(transformed)
        else:
            self.individual_record.extend(solutions_batch.copy())
            fitness = self.fun(solutions_batch)

        self.fitness_record.extend(fitness.copy())
        return fitness


# 向后兼容别名
FunRecord = FitnessRecorder
