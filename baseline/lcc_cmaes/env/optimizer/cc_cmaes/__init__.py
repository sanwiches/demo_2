"""
Cooperative Coevolution CMA-ES Package / 协同进化 CMA-ES 包

This package contains cooperative coevolution variants of CMA-ES with different
decomposition strategies.
本包包含具有不同分解策略的协同进化 CMA-ES 变体。

Modules / 模块:
    - adapter: Action adapter for strategy selection / 策略选择动作适配器
    - mivd: Multi-individual variable decomposition / 多个体变量分解
    - mavd: Multi-adaptive variable decomposition / 多自适应变量分解
    - ravd: Random adaptive variable decomposition / 随机自适应变量分解
    - cc_cmaes: Main CC-CMAES implementation / 主 CC-CMAES 实现

Usage / 使用方法:
    >>> from env.optimizer.cc_cmaes import ccc_cmaes, mivd_groups, mavd_groups, ravd_groups
"""

__all__ = []
