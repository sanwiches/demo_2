"""
最大方差分解模块 / Maximum Variance Decomposition Module.

此模块实现了基于协方差矩阵对角线方差的最大方差分组策略。
This module implements the maximum variance decomposition strategy based on
the diagonal variance of the covariance matrix.

该策略将协方差矩阵对角线元素按方差从大到小排序，然后按间隔采样分组。
This strategy sorts diagonal elements of covariance matrix by variance in
descending order, then groups by interval sampling.

主要组件 / Main Components:
    - mavd_groups: 最大方差分解分组函数 / Maximum variance decomposition grouping function

使用示例 / Usage Example:
    >>> import numpy as np
    >>> from env.optimizer.cc_cmaes.mavd import mavd_groups
    >>> C = np.eye(10)
    >>> groups = mavd_groups(dimension=10, subgroup_size=3, covariance=C)
    >>> print(len(groups))  # Number of subgroups
"""

from typing import List

import numpy as np
import numpy.typing as npt


def mavd_groups(dimension: int, subgroup_size: int, covariance: npt.NDArray[np.float64]) -> List[npt.NDArray[np.int64]]:
    """
    最大方差分解分组 / Maximum Variance Decomposition grouping.

    将维度按协方差矩阵对角线方差从大到小排序，然后按间隔采样分组。
    Sorts dimensions by diagonal variance of covariance matrix in descending order,
    then groups by interval sampling.

    分组策略：
    - 按方差降序排列维度 / Sort dimensions by variance in descending order
    - 使用间隔采样（stride）创建子组 / Use interval sampling (stride) to create subgroups
    - 每个子组包含间隔 subgroup_size 的维度 / Each subgroup contains dimensions spaced by subgroup_size

    Args:
        dimension: 问题总维度 / Total problem dimension (>= 1)
        subgroup_size: 采样间隔（子组内维度间隔）/ Sampling stride (interval between dimensions in subgroup)
        covariance: 协方差矩阵 / Covariance matrix, shape (dimension, dimension)

    Returns:
        子组索引列表 / List of subgroup index arrays

    Raises:
        ValueError: 当 dimension 或 subgroup_size 无效时 / When dimension or subgroup_size is invalid

    Example:
        >>> import numpy as np
        >>> C = np.diag([0.1, 0.5, 1.0, 2.0, 5.0])
        >>> groups = mavd_groups(dimension=5, subgroup_size=2, covariance=C)
        >>> # Highest variance dimensions distributed across groups
        >>> print(groups[0])  # [4, 2, 0] - dimensions with highest variance, sampled every 2
    """
    # 参数验证 / Parameter validation
    if dimension < 1:
        raise ValueError(f"维度必须 >= 1 / Dimension must be >= 1, got {dimension}")
    if subgroup_size < 1:
        raise ValueError(f"子组大小必须 >= 1 / Subgroup size must be >= 1, got {subgroup_size}")
    if covariance.shape != (dimension, dimension):
        raise ValueError(
            f"协方差矩阵形状不匹配 / Covariance matrix shape mismatch: "
            f"expected ({dimension}, {dimension}), got {covariance.shape}"
        )

    # 获取对角线方差并降序排序 / Get diagonal variances and sort in descending order
    diagonal_variances = np.diag(covariance)
    sorted_indices = np.argsort(diagonal_variances)[::-1]  # 降序 / Descending order

    # 计算子组数量 / Calculate number of subgroups
    num_subgroups = int(np.ceil(dimension / subgroup_size))

    subgroups: List[npt.NDArray[np.int64]] = []
    for i in range(num_subgroups):
        # 间隔采样 / Interval sampling
        subgroup = np.sort(sorted_indices[i:dimension:num_subgroups])
        subgroups.append(subgroup)

    return subgroups
