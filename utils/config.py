"""
Configuration Management / 配置管理模块

This module provides centralized configuration management for the WCC project,
including paths, constants, and experiment settings.

此模块为 WCC 项目提供集中式配置管理，包括路径、常量和实验设置。

Author: WCC Project
Date: 2026-03-11
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import time


# =============================================================================
# Path Configuration / 路径配置
# =============================================================================

class ProjectPaths:
    """
    Centralized path management for the project / 项目的集中式路径管理

    All paths are resolved relative to the project root to ensure
    portability across different systems.

    所有路径都相对于项目根目录解析，以确保在不同系统间的可移植性。

    Attributes:
        root: Project root directory / 项目根目录
        env: Environment directory / 环境目录
        data: Data directory / 数据目录
        save: Save directory for results / 结果保存目录
        baseline: Baseline algorithms directory / 基线算法目录
        log: Log directory / 日志目录
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path configuration / 初始化路径配置

        Args:
            project_root: Custom project root path. If None, auto-detected. /
                         自定义项目根路径。如果为 None，则自动检测。
        """
        if project_root is None:
            # Auto-detect project root (this file's parent's parent) / 自动检测项目根目录
            self.root = Path(__file__).resolve().parent.parent
        else:
            self.root = Path(project_root).resolve()

        # Define all project paths / 定义所有项目路径
        self.env = self.root / "env"
        self.data = self.root / "data"
        self.benchmark = self.root / "benchmark"
        self.save = self.root / "WCC" / "save_dir"
        self.baseline = self.root / "baseline"
        self.utils = self.root / "utils"
        self.experiment = self.root / "experiment"
        self.log = self.save / "log"
        self.model = self.save / "ppo_model"
        self.test = self.save / "test"

        # Benchmark-specific paths / 基准测试特定路径
        self.ob_data = self.benchmark / "Overlapping_Benchmark" / "datafile"
        self.aob_data = self.benchmark / "aob" / "datafile"  # AOB (Overlapping Benchmark) data
        self.cec_data = self.benchmark / "cec2013lsgo" / "cdatafiles"

    def get_output_path(
        self,
        algorithm: str,
        nondep: bool,
        func_name: str,
        fun_id: int,
        prefix: str = "baseline_OB_nondep"
    ) -> str:
        """
        Generate standardized output path for experiment results /
        为实验结果生成标准化的输出路径

        Args:
            algorithm: Algorithm name (e.g., 'CC', 'FCMAES') / 算法名称
            nondep: Non-decomposition mode flag / 非分解模式标志
            func_name: Function name (e.g., 'ackley') / 函数名称
            fun_id: Function ID / 函数 ID
            prefix: Directory prefix / 目录前缀

        Returns:
            Output path string with timestamp / 带时间戳的输出路径字符串
        """
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        return f'WCC/save_dir/{prefix}_{nondep}/{algorithm}_3E6/{func_name}_F{fun_id}_{timestamp}/'

    def ensure_dir(self, *path_parts) -> Path:
        """
        Ensure a directory exists, create if it doesn't / 确保目录存在，不存在则创建

        Args:
            *path_parts: Path components relative to project root / 相对于项目根目录的路径组件

        Returns:
            Absolute path to the directory / 目录的绝对路径
        """
        path = self.root.joinpath(*path_parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global paths instance / 全局路径实例
PATHS = ProjectPaths()


# =============================================================================
# Optimization Constants / 优化常量
# =============================================================================

class OptimConfig:
    """
    Default optimization parameters / 默认优化参数

    Attributes:
        DEFAULT_MAX_FES: Default maximum function evaluations / 默认最大函数评估次数
        DEFAULT_CYCLE_NUM: Default number of parallel runs / 默认并行运行次数
        DEFAULT_CC_SIGMA: Default sigma for CMAES / CMAES 的默认 sigma
        DEFAULT_NDA_SIGMA_RATIO: Sigma ratio for NDA / NDA 的 sigma 比例
        DEFAULT_VERBOSE: Default verbose interval / 默认详细输出间隔
        DEFAULT_EARLY_STOPPING: Default early stopping evaluations / 默认早停评估次数
    """

    # Function evaluation budgets / 函数评估预算
    DEFAULT_MAX_FES = 3e6
    DEFAULT_INIT_FES = 1000

    # Parallel execution / 并行执行
    DEFAULT_CYCLE_NUM = 10
    DEFAULT_BATCH_SIZE = 4

    # CMAES parameters / CMAES 参数
    DEFAULT_CC_SIGMA = 0.5
    DEFAULT_NDA_SIGMA_RATIO = 0.3  # 30% of search range / 搜索范围的 30%

    # Logging and control / 日志和控制
    DEFAULT_VERBOSE = 1000
    DEFAULT_EARLY_STOPPING = 1000
    DEFAULT_SEED = 42

    # Recording checkpoints / 记录检查点
    RECORD_FES_LIST = [1.2e5, 2e5, 1e6, 2e6, 3e6]


# =============================================================================
# Function Lists / 函数列表
# =============================================================================

class FunctionSets:
    """
    Standard function sets for benchmark testing / 基准测试的标准函数集

    Attributes:
        OB_FUNCTIONS: Overlapping Benchmark function names / 重叠基准测试函数名称
        CEC_FUNCTIONS: CEC 2013 LSGO function IDs / CEC 2013 LSGO 函数 ID
    """

    # Overlapping Benchmark function names / 重叠基准测试函数名称
    OB_FUNCTIONS = ["ackley", "elliptic", "rastrigin", "schwefel"]

    # CEC 2013 function IDs / CEC 2013 函数 ID
    CEC_ALL = list(range(1, 16))  # F1-F15
    CEC_NONSEP = [1, 2, 3, 12, 15]  # Non-separable / 不可分离
    CEC_OVERLAP = [13, 14]  # With overlap / 有重叠
    CEC_SEP = [4, 5, 6, 7, 8, 9, 10, 11]  # Separable / 可分离

    @staticmethod
    def get_overlap_map() -> Dict[int, int]:
        """
        Get standard overlap degree mapping / 获取标准重叠度映射

        Returns:
            Dictionary mapping function IDs to overlap degrees /
            函数 ID 到重叠度的映射字典
        """
        return {2: 10, 3: 15, 4: 20}


# =============================================================================
# Configuration Builder / 配置构建器
# =============================================================================

class ConfigBuilder:
    """
    Builder pattern for creating experiment configurations /
    用于创建实验配置的构建器模式

    This class provides a fluent interface for building configuration
    dictionaries for experiments.

    此类提供了用于构建实验配置字典的流畅接口。

    Example:
        >>> config = (ConfigBuilder()
        ...     .algorithm("CC")
        ...     .nondep(True)
        ...     .functions(["ackley"])
        ...     .fun_ids([1])
        ...     .max_fes(3e6)
        ...     .build())
    """

    def __init__(self):
        """Initialize empty configuration / 初始化空配置"""
        self._config: Dict[str, Any] = {}

    def algorithm(self, algo_type: str, name: Optional[str] = None) -> "ConfigBuilder":
        """
        Set algorithm type and name / 设置算法类型和名称

        Args:
            algo_type: Algorithm type ('CC' or 'NDA') / 算法类型
            name: Algorithm name for output. If None, uses algo_type. /
                  用于输出的算法名称。如果为 None，使用 algo_type。
        """
        self._config["algorithm_type"] = algo_type
        self._config["algorithm_name"] = name if name is not None else algo_type
        return self

    def nondep(self, nondep: bool) -> "ConfigBuilder":
        """Set non-decomposition mode flag / 设置非分解模式标志"""
        self._config["nondep"] = nondep
        return self

    def functions(self, func_list: List[str]) -> "ConfigBuilder":
        """Set function name list / 设置函数名称列表"""
        self._config["func_name_list"] = func_list
        return self

    def fun_ids(self, id_list: List[int]) -> "ConfigBuilder":
        """Set function ID list / 设置函数 ID 列表"""
        self._config["fun_id_list"] = id_list
        return self

    def max_fes(self, max_fes: float) -> "ConfigBuilder":
        """Set maximum function evaluations / 设置最大函数评估次数"""
        self._config["max_fes"] = max_fes
        return self

    def cycles(self, cycle_num: int) -> "ConfigBuilder":
        """Set number of parallel runs / 设置并行运行次数"""
        self._config["cycle_num"] = cycle_num
        return self

    def grouping(
        self,
        mode: str,
        overlap_map: Optional[Dict[int, int]] = None,
        chunk_count: int = 20
    ) -> "ConfigBuilder":
        """
        Set variable grouping configuration / 设置变量分组配置

        Args:
            mode: Grouping mode ('none', 'equal_split', or 'partition') / 分组模式
            overlap_map: Overlap degree mapping (for 'partition') / 重叠度映射
            chunk_count: Number of chunks (for 'equal_split') / 块数量
        """
        self._config["grouping_mode"] = mode
        if mode == "partition" and overlap_map is not None:
            self._config["overlap_map"] = overlap_map
        if mode == "equal_split":
            self._config["chunk_count"] = chunk_count
        return self

    def custom(self, key: str, value: Any) -> "ConfigBuilder":
        """Add custom configuration parameter / 添加自定义配置参数"""
        self._config[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build and return the configuration dictionary / 构建并返回配置字典

        Returns:
            Configuration dictionary / 配置字典
        """
        return self._config.copy()


# =============================================================================
# Predefined Configurations / 预定义配置
# =============================================================================

def get_cc_nondep_config() -> Dict[str, Any]:
    """Get CC non-decomposition mode configuration / 获取 CC 非分解模式配置"""
    return (ConfigBuilder()
            .algorithm("CC")
            .nondep(True)
            .functions(FunctionSets.OB_FUNCTIONS)
            .fun_ids([1])
            .max_fes(OptimConfig.DEFAULT_MAX_FES)
            .cycles(OptimConfig.DEFAULT_CYCLE_NUM)
            .grouping("equal_split", chunk_count=20)
            .build())


def get_cc_decomp_config() -> Dict[str, Any]:
    """Get CC decomposition mode configuration / 获取 CC 分解模式配置"""
    return (ConfigBuilder()
            .algorithm("CC")
            .nondep(False)
            .functions(FunctionSets.OB_FUNCTIONS)
            .fun_ids([2, 3, 4])
            .max_fes(OptimConfig.DEFAULT_MAX_FES)
            .cycles(OptimConfig.DEFAULT_CYCLE_NUM)
            .grouping("partition", overlap_map=FunctionSets.get_overlap_map())
            .build())


def get_nda_nondep_config() -> Dict[str, Any]:
    """Get NDA non-decomposition mode configuration / 获取 NDA 非分解模式配置"""
    return (ConfigBuilder()
            .algorithm("NDA", "FCMAES")
            .nondep(True)
            .functions(FunctionSets.OB_FUNCTIONS)
            .fun_ids([1])
            .max_fes(OptimConfig.DEFAULT_MAX_FES)
            .cycles(OptimConfig.DEFAULT_CYCLE_NUM)
            .grouping("none")
            .build())


def get_nda_decomp_config() -> Dict[str, Any]:
    """Get NDA decomposition mode configuration / 获取 NDA 分解模式配置"""
    return (ConfigBuilder()
            .algorithm("NDA", "FCMAES")
            .nondep(False)
            .functions(FunctionSets.OB_FUNCTIONS)
            .fun_ids([2, 3, 4])
            .max_fes(OptimConfig.DEFAULT_MAX_FES)
            .cycles(OptimConfig.DEFAULT_CYCLE_NUM)
            .grouping("none")
            .build())
