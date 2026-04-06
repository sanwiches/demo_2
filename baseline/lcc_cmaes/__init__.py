"""
LCC-CMAES: Learning-based Cooperative Coevolution CMA-ES
基于学习的协同进化 CMA-ES

A simplified optimizer wrapper for testing the trained LCC-CMAES algorithm.
训练好的 LCC-CMAES 算法的简洁优化器封装。

Paper / 论文: "Advancing CMA-ES with Learning-Based Cooperative Coevolution for Scalable Optimization", GECCO 2026

Usage / 使用方法:
    >>> from baseline.lcc_cmaes import LCC_CMAES, optimize_with_lcc_cmaes
    >>>
    >>> # Method 1: Using the class directly / 方法1：直接使用类
    >>> optimizer = LCC_CMAES(problem, options)
    >>> results = optimizer.optimize()
    >>>
    >>> # Method 2: Using the convenience function / 方法2：使用便捷函数
    >>> results = optimize_with_lcc_cmaes(
    ...     fitness_function=lambda x: sum(x**2),
    ...     ndim_problem=1000,
    ...     model_path='path/to/model.pt'
    ... )
"""

from .lcc_cmaes import LCC_CMAES, optimize_with_lcc_cmaes

__all__ = ['LCC_CMAES', 'optimize_with_lcc_cmaes']
