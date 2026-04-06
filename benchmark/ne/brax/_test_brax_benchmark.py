"""
NE基准测试模块的测试脚本 / Test script for NE Benchmark module.

此脚本演示如何使用NE Benchmark类 / This script demonstrates how to use the NE Benchmark class
该类与cec2013lsgo具有相同的接口 / with the same interface as cec2013lsgo.

使用方法 / Usage:
    python _test_ne_benchmark.py                    # 自动检测设备 / Auto-detect device
    python _test_ne_benchmark.py --device cuda      # 指定CUDA设备 / Specify CUDA device
    python _test_ne_benchmark.py --device cpu       # 指定CPU设备 / Specify CPU device
"""

import argparse
import logging
from typing import Optional

import numpy as np
import torch
from .brax_benchmarks import Benchmark

# ============================================================================
# 配置常量 / Configuration Constants
# ============================================================================

# 默认显示的最大函数数量 / Maximum number of functions to display by default
DEFAULT_MAX_FUNCTIONS_DISPLAY = 70

# 默认测试种群大小 / Default test population size
DEFAULT_POPULATION_SIZE = 2

# 日志配置 / Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_device_info(device: str) -> None:
    """
    打印设备信息 / Print device information.

    Args:
        device: 设备名称 / Device name ('cuda' or 'cpu')
    """
    separator = "=" * 50
    logger.info(separator)
    logger.info(f"NE Benchmark测试 / NE Benchmark Test (设备/Device: {device})")
    logger.info(separator)
    logger.info(f"CUDA可用/CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA设备/CUDA Device: {torch.cuda.get_device_name(0)}")


def print_available_functions(benchmark: Benchmark,
                              max_display: int = DEFAULT_MAX_FUNCTIONS_DISPLAY) -> None:
    """
    打印可用的函数列表 / Print list of available functions.

    Args:
        benchmark: Benchmark实例 / Benchmark instance
        max_display: 最大显示数量 / Maximum number to display
    """
    num_functions = benchmark.get_num_functions()
    logger.info(f"\n函数总数/Total number of functions: {num_functions}")

    logger.info("\n可用函数/Available functions:")
    functions = benchmark.list_functions()

    # 显示前N个函数 / Display first N functions
    display_count = min(max_display, len(functions))
    for i, func in enumerate(functions[:display_count]):
        logger.info(
            f"  ID {func['id']:2d}: {func['env_name']:20s} "
            f"深度/depth={func['model_depth']} 维度/dim={func['dimension']}"
        )

    # 显示剩余数量 / Show remaining count
    remaining = num_functions - display_count
    if remaining > 0:
        logger.info(f"  ... 以及/and {remaining} 更多/more")


def test_function_info(benchmark: Benchmark, func_id: int) -> None:
    """
    测试获取函数信息 / Test getting function information.

    Args:
        benchmark: Benchmark实例 / Benchmark instance
        func_id: 函数ID / Function ID to test
    """
    separator = "\n" + "-" * 50
    logger.info(separator)
    logger.info(f"测试get_info() / Testing get_info() 函数/function {func_id}:")

    info = benchmark.get_info(func_id)
    logger.info(f"  维度/Dimension: {info['dimension']}")
    logger.info(f"  下界/Lower bound: {info['lower']}")
    logger.info(f"  上界/Upper bound: {info['upper']}")
    logger.info(f"  环境/Environment: {info['env_name']}")
    logger.info(f"  模型深度/Model depth: {info['model_depth']}")


def test_function_retrieval(benchmark: Benchmark, func_id: int) -> None:
    """
    测试函数检索 / Test function retrieval.

    Args:
        benchmark: Benchmark实例 / Benchmark instance
        func_id: 函数ID / Function ID to retrieve
    """
    separator = "\n" + "-" * 50
    logger.info(separator)
    logger.info(f"测试get_function() / Testing get_function() 函数/function {func_id}:")

    problem = benchmark.get_function(func_id)
    logger.info(f"  问题类/Problem class: {problem.__class__.__name__}")
    logger.info(f"  问题字符串/Problem string: {problem}")


def test_function_evaluation(benchmark: Benchmark,
                            func_id: int,
                            pop_size: int = DEFAULT_POPULATION_SIZE) -> bool:
    """
    测试函数评估 / Test function evaluation.

    Args:
        benchmark: Benchmark实例 / Benchmark instance
        func_id: 函数ID / Function ID to test
        pop_size: 种群大小 / Population size for testing

    Returns:
        bool: 评估是否成功 / Whether evaluation was successful
    """
    separator = "\n" + "-" * 50
    logger.info(separator)
    logger.info("测试函数评估 / Testing function evaluation (这可能需要一些时间/this may take a while...):")

    # 获取问题信息 / Get problem information
    info = benchmark.get_info(func_id)
    dim = info['dimension']
    lb = info['lower']
    ub = info['upper']

    # 创建随机测试种群 / Create random test population
    # 注意: 种群大小较小时测试更快，但可能不足以验证功能
    # Note: Smaller population is faster for testing but may not fully validate functionality
    x = np.random.uniform(lb, ub, size=(pop_size, dim))

    logger.info(f"  种群形状/Population shape: {x.shape}")

    try:
        problem = benchmark.get_function(func_id)
        fitness = problem(x)

        logger.info(f"  适应度形状/Fitness shape: {fitness.shape}")
        logger.info(f"  适应度值/Fitness values: {fitness}")
        logger.info("  ✓ 评估成功/Evaluation successful!")
        return True

    except Exception as e:
        logger.error(f"  ✗ 评估失败/Evaluation failed: {e}")
        return False


def test_ne_benchmark(device: Optional[str] = None) -> None:
    """
    测试NE Benchmark类 / Test the NE Benchmark class.

    此函数执行完整的基准测试流程，包括:
    - 设备信息显示 / Device information display
    - 函数列表显示 / Function list display
    - 函数信息获取 / Function information retrieval
    - 函数评估测试 / Function evaluation testing

    Args:
        device: 要使用的设备 / Device to use
               - 'cuda': 使用CUDA设备 / Use CUDA device
               - 'cpu': 使用CPU设备 / Use CPU device
               - None: 自动检测 / Auto-detect based on availability
    """
    # 检测CUDA可用性 / Check CUDA availability
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 打印设备信息 / Print device information
    print_device_info(device)

    # 初始化基准测试 / Initialize benchmark
    try:
        benchmark = Benchmark(device=device)
    except Exception as e:
        logger.error(f"基准初始化失败/Benchmark initialization failed: {e}")
        return

    # 显示可用函数 / Display available functions
    print_available_functions(benchmark)

    # 测试函数1: ant, depth=0 / Test function 1: ant, depth=0
    test_function_info(benchmark, 1)
    test_function_retrieval(benchmark, 1)
    test_function_evaluation(benchmark, 1)

    # 测试函数7: halfcheetah, depth=0 / Test function 7: halfcheetah, depth=0
    test_function_info(benchmark, 7)

    # 完成 / Completion
    separator = "\n" + "=" * 50
    logger.info(separator)
    logger.info("测试完成/Test completed!")
    logger.info(separator)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数 / Parse command line arguments.

    Returns:
        argparse.Namespace: 解析后的参数 / Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='测试NE基准 / Test NE Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例/Examples:
  %(prog)s                    自动检测设备 / Auto-detect device
  %(prog)s --device cuda      使用CUDA设备 / Use CUDA device
  %(prog)s --device cpu       使用CPU设备 / Use CPU device
        """
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='要使用的设备 / Device to use (cuda/cpu, 默认/default: 自动检测/auto-detect)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细输出 / Enable verbose output'
    )

    return parser.parse_args()


def main() -> None:
    """
    主入口函数 / Main entry point.
    """
    args = parse_arguments()

    # 根据verbose参数调整日志级别 / Adjust log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    test_ne_benchmark(device=args.device)


if __name__ == "__main__":
    main()
