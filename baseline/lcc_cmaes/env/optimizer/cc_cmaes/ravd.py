"""
随机方差分解模块 / Random Variance Decomposition Module.

此模块实现了随机维度分组策略。
This module implements the random dimension grouping strategy.

该策略将维度随机排列，然后按顺序分组，不依赖于协方差矩阵信息。
This strategy randomly permutes dimensions and groups them sequentially,
without depending on covariance matrix information.

主要组件 / Main Components:
    - ravd_groups: 随机方差分解分组函数 / Random variance decomposition grouping function

使用示例 / Usage Example:
    >>> import numpy as np
    >>> from env.optimizer.cc_cmaes.ravd import ravd_groups
    >>> groups = ravd_groups(dimension=10, subgroup_size=3, seed=42)
    >>> print(len(groups))  # Number of subgroups
"""

from typing import List, Optional

import numpy as np
import numpy.typing as npt


def ravd_groups(
    dimension: int,
    subgroup_size: int,
    covariance: npt.NDArray[np.float64],
    seed: Optional[int] = None
) -> List[npt.NDArray[np.int64]]:
    """
    随机方差分解分组 / Random Variance Decomposition grouping.

    将维度随机排列，然后按顺序分组。
    Randomly permutes dimensions and groups them sequentially.

    分组策略：
    - 随机打乱维度顺序 / Randomly shuffle dimension order
    - 每 subgroup_size 个维度分为一组 / Group every subgroup_size dimensions
    - 最后一组可能不足 subgroup_size 个 / Last group may have fewer than subgroup_size

    注意：covariance 参数未使用，仅为了与其他分组函数保持接口一致。
    Note: The covariance parameter is unused, only for interface consistency with
    other grouping functions.

    Args:
        dimension: 问题总维度 / Total problem dimension (>= 1)
        subgroup_size: 每个子组的维度数 / Dimension count per subgroup (>= 1)
        covariance: 协方差矩阵（未使用）/ Covariance matrix (unused)
        seed: 随机种子 / Random seed for reproducibility

    Returns:
        子组索引列表 / List of subgroup index arrays

    Raises:
        ValueError: 当 dimension 或 subgroup_size 无效时 / When dimension or subgroup_size is invalid

    Example:
        >>> from env.optimizer.cc_cmaes.ravd import ravd_groups
        >>> groups = ravd_groups(dimension=10, subgroup_size=3, seed=42)
        >>> # Dimensions are randomly assigned to groups
        >>> print(groups[0].shape)  # First group has 3 dimensions
    """
    # 参数验证 / Parameter validation
    if dimension < 1:
        raise ValueError(f"维度必须 >= 1 / Dimension must be >= 1, got {dimension}")
    if subgroup_size < 1:
        raise ValueError(f"子组大小必须 >= 1 / Subgroup size must be >= 1, got {subgroup_size}")

    # 随机排列维度 / Randomly permute dimensions
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(dimension)

    # 计算子组数量 / Calculate number of subgroups
    num_subgroups = int(np.ceil(dimension / subgroup_size))

    subgroups: List[npt.NDArray[np.int64]] = []
    for i in range(num_subgroups):
        start_idx = i * subgroup_size
        end_idx = min((i + 1) * subgroup_size, dimension)
        subgroup = permuted_indices[start_idx:end_idx]
        subgroups.append(subgroup)

    return subgroups
