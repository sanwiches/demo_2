"""
最小方差分解模块 / Minimum Variance Decomposition Module.

此模块实现了基于协方差矩阵对角线方差的最小方差分组策略。
This module implements the minimum variance decomposition strategy based on
the diagonal variance of the covariance matrix.

该策略将协方差矩阵对角线元素按方差从小到大排序，然后将相邻维度分组。
This strategy sorts diagonal elements of covariance matrix by variance in
ascending order, then groups adjacent dimensions.

主要组件 / Main Components:
    - mivd_groups: 最小方差分解分组函数 / Minimum variance decomposition grouping function

使用示例 / Usage Example:
    >>> import numpy as np
    >>> from env.optimizer.cc_cmaes.mivd import mivd_groups
    >>> C = np.eye(10)
    >>> groups = mivd_groups(dimension=10, subgroup_size=3, covariance=C)
    >>> print(len(groups))  # Number of subgroups
"""

from typing import List

import numpy as np
import numpy.typing as npt


def mivd_groups(dimension: int, subgroup_size: int, covariance: npt.NDArray[np.float64]) -> List[npt.NDArray[np.int64]]:
    """
    最小方差分解分组 / Minimum Variance Decomposition grouping.

    将维度按协方差矩阵对角线方差从小到大排序，然后按顺序分组。
    Sorts dimensions by diagonal variance of covariance matrix in ascending order,
    then groups them sequentially.

    分组策略：
    - 按方差升序排列维度 / Sort dimensions by variance in ascending order
    - 每 subgroup_size 个维度分为一组 / Group every subgroup_size dimensions
    - 最后一组可能不足 subgroup_size 个 / Last group may have fewer than subgroup_size

    Args:
        dimension: 问题总维度 / Total problem dimension (>= 1)
        subgroup_size: 每个子组的维度数 / Dimension count per subgroup (>= 1)
        covariance: 协方差矩阵 / Covariance matrix, shape (dimension, dimension)

    Returns:
        子组索引列表 / List of subgroup index arrays

    Raises:
        ValueError: 当 dimension 或 subgroup_size 无效时 / When dimension or subgroup_size is invalid

    Example:
        >>> import numpy as np
        >>> C = np.diag([0.1, 0.5, 1.0, 2.0, 5.0])
        >>> groups = mivd_groups(dimension=5, subgroup_size=2, covariance=C)
        >>> # Lowest variance dimensions grouped first
        >>> print(groups[0])  # [0, 1] - dimensions with variance 0.1, 0.5
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

    # 获取对角线方差并排序 / Get diagonal variances and sort
    diagonal_variances = np.diag(covariance)
    sorted_indices = np.argsort(diagonal_variances)

    # 计算子组数量 / Calculate number of subgroups
    num_subgroups = int(np.ceil(dimension / subgroup_size))

    subgroups: List[npt.NDArray[np.int64]] = []
    for i in range(num_subgroups):
        start_idx = i * subgroup_size
        end_idx = min((i + 1) * subgroup_size, dimension)
        # 取排序后的索引并重新排序（保持原始顺序）/ Take sorted indices and re-sort
        subgroup = np.sort(sorted_indices[start_idx:end_idx])
        subgroups.append(subgroup)

    return subgroups
