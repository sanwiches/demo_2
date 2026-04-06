"""
Shared utility modules for WCC project / WCC 项目共享工具模块

This package contains common utility functions and classes used across
different parts of the project to reduce code duplication.

此包包含项目中不同部分使用的通用工具函数和类，以减少代码重复。

Modules:
    optimization: Optimization-related utilities (FitnessRecorder, combine_vectors, etc.)
    config: Configuration utilities (paths, constants, config builder)

Usage:
    from utils import FitnessRecorder, parallel_optimization, PATHS
    from utils import get_cc_nondep_config, get_nda_nondep_config

使用方式：
    from utils import FitnessRecorder, parallel_optimization, PATHS
    from utils import get_cc_nondep_config, get_nda_nondep_config
"""

# Version / 版本
__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies / 延迟导入以避免循环依赖
# These are imported on-demand when accessed / 这些在访问时按需导入

def __getattr__(name: str):
    """Lazy import attributes to avoid circular dependencies / 延迟导入属性以避免循环依赖"""
    if name == "FitnessRecorder":
        from .optimization import FitnessRecorder
        return FitnessRecorder
    if name == "combine_vectors":
        from .optimization import combine_vectors
        return combine_vectors
    if name == "build_grouping_result":
        from .optimization import build_grouping_result
        return build_grouping_result
    if name == "parallel_optimization":
        from .optimization import parallel_optimization
        return parallel_optimization
    if name == "optimization_task_cc":
        from .optimization import optimization_task_cc
        return optimization_task_cc
    if name == "optimization_task_nda":
        from .optimization import optimization_task_nda
        return optimization_task_nda
    if name == "ProjectPaths":
        from .config import ProjectPaths
        return ProjectPaths
    if name == "OptimConfig":
        from .config import OptimConfig
        return OptimConfig
    if name == "FunctionSets":
        from .config import FunctionSets
        return FunctionSets
    if name == "ConfigBuilder":
        from .config import ConfigBuilder
        return ConfigBuilder
    if name == "PATHS":
        from .config import PATHS
        return PATHS
    if name == "get_cc_nondep_config":
        from .config import get_cc_nondep_config
        return get_cc_nondep_config
    if name == "get_cc_decomp_config":
        from .config import get_cc_decomp_config
        return get_cc_decomp_config
    if name == "get_nda_nondep_config":
        from .config import get_nda_nondep_config
        return get_nda_nondep_config
    if name == "get_nda_decomp_config":
        from .config import get_nda_decomp_config
        return get_nda_decomp_config
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Optimization / 优化
    "FitnessRecorder",
    "combine_vectors",
    "build_grouping_result",
    "parallel_optimization",
    "optimization_task_cc",
    "optimization_task_nda",
    # Configuration / 配置
    "ProjectPaths",
    "OptimConfig",
    "FunctionSets",
    "ConfigBuilder",
    "PATHS",
    "get_cc_nondep_config",
    "get_cc_decomp_config",
    "get_nda_nondep_config",
    "get_nda_decomp_config",
]
