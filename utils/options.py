"""
Command-Line Options Parser / 命令行选项解析器

This module provides argument parsing for the WCC (Within-Problem Cooperative Coevolution)
project. It centralizes all hyperparameters and configuration options for PPO training
and optimization algorithms.

本模块为 WCC（问题内协同进化）项目提供参数解析。它集中了 PPO 训练和优化算法的
所有超参数和配置选项。

Usage / 用法:
    from utils.options import get_options
    opts = get_options()  # Parse from sys.argv / 从 sys.argv 解析
    opts = get_options(['--lr_model', '1e-3'])  # Parse custom args / 解析自定义参数

Main Categories / 主要类别:
    - Backbone: CMAES optimization settings / CMAES 优化设置
    - Network: Actor/Critic network architecture / 演员网络/评论家网络架构
    - Training: PPO hyperparameters / PPO 超参数
    - Paths: Directory configurations / 目录配置
    - Logging: TensorBoard and checkpoint settings / TensorBoard 和检查点设置

Author: WCC Project
Date: 2026-03-11
"""

import os
import time
import argparse
from typing import List

import torch

from utils.config import PATHS


def _parse_resource_list(arg_string: str) -> List[int]:
    """
    Parse resource list from command line argument.
    从命令行参数解析资源列表。

    Args:
        arg_string: Comma-separated string (e.g., "50000,100000,200000")
                   逗号分隔的字符串

    Returns:
        List of integers / 整数列表

    Example:
        >>> _parse_resource_list("50000,100000,200000")
        [50000, 100000, 200000]
    """
    return [int(x.strip()) for x in arg_string.split(',')]


def get_options(args=None):
    """
    Parse and return configuration options.
    解析并返回配置选项。

    This function creates an argument parser, defines all command-line arguments,
    parses them, and performs post-processing to set derived paths and options.

    此函数创建参数解析器，定义所有命令行参数，解析它们，并执行后处理以设置
    派生路径和选项。

    Args:
        args: Optional list of arguments to parse. If None, uses sys.argv.
             可选的要解析的参数列表。如果为 None，使用 sys.argv。

    Returns:
        Namespace object containing all configuration options.
        包含所有配置选项的命名空间对象。
    """

    # =========================================================================
    # Initialize Parser / 初始化解析器
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="WCC: Within-Problem Cooperative Coevolution with PPO"
    )

    # =========================================================================
    # Backbone Optimization Settings / 主干优化设置
    # =========================================================================
    parser.add_argument(
        '--backbone',
        default='cmaes',
        choices=['cmaes'],
        help='Backbone optimization algorithm / 主干优化算法'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=10,
        help='Number of subgroups / 子组数量'
    )
    parser.add_argument(
        '--subspace_dim',
        type=int,
        default=100,
        help='Dimensionality of each subspace / 每个子空间的维度'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=2,
        help='Initial step size for CMAES / CMAES 初始步长'
    )
    parser.add_argument(
        '--sub_popsize',
        type=int,
        default=20,
        help='Population size of each subgroup / 每个子组的种群大小'
    )
    parser.add_argument(
        '--max_fes',
        type=float,
        default=1e6,
        help='Maximum number of function evaluations / 最大函数评估次数'
    )
    parser.add_argument(
        '--subFEs',
        type=int,
        default=1000,
        help='Minimum FEs for each subgroup iteration / 每个子组迭代的最小评估次数'
    )
    parser.add_argument(
        '--initFEs',
        type=int,
        default=1000,
        help='Number of FEs for initialization / 初始化的评估次数'
    )
    parser.add_argument(
        '--output_init_cma_info',
        type=bool,
        default=False,
        help='Output initial CMAES information / 输出初始 CMAES 信息'
    )

    # =========================================================================
    # Problem Settings / 问题设置
    # =========================================================================
    parser.add_argument(
        '--divide_method',
        default="CEC2013LSGO",
        choices=["BNS", "OB_nondep", "CEC2013LSGO"],
        help='Method to divide the problem set / 划分问题集的方法'
    )

    # =========================================================================
    # Action Space Settings / 动作空间设置
    # =========================================================================
    parser.add_argument(
        '--action_space',
        type=int,
        default=3,
        help='Number of resource allocation actions / 资源分配动作数量 (0/1/2)'
    )
    parser.add_argument(
        '--resource_list',
        type=_parse_resource_list,
        default="50000,100000,200000",
        help='Resource limits for each action level (comma-separated) / 每个动作级别的资源限制（逗号分隔）'
    )

    # =========================================================================
    # Rollout Settings / 滚动设置
    # =========================================================================
    parser.add_argument(
        '--one_problem_batch_size',
        type=int,
        default=4,
        help='Number of instances for each problem / 每个问题的实例数量'
    )
    parser.add_argument(
        '--per_eval_time',
        type=int,
        default=1,
        help='Number of evaluations for each instance / 每个实例的评估次数'
    )

    # =========================================================================
    # PPO Batch Settings / PPO 批次设置
    # =========================================================================
    parser.add_argument(
        '--each_question_batch_num',
        type=int,
        default=4,
        help='Number of parallel instances for each problem / 每个问题的并行实例数量'
    )

    # =========================================================================
    # Learning Rate Settings / 学习率设置
    # =========================================================================
    parser.add_argument(
        '--lr_critic',
        type=float,
        default=1e-4,
        help='Learning rate for critic network / 评论家网络学习率'
    )
    parser.add_argument(
        '--lr_model',
        type=float,
        default=1e-4,
        help='Learning rate for actor network / 演员网络学习率'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.99,
        help='Learning rate decay per epoch / 每轮学习率衰减'
    )

    # =========================================================================
    # State and Feature Dimensions / 状态和特征维度
    # =========================================================================
    parser.add_argument(
        '--state',
        default=[0.0 for _ in range(16)],
        help='Initial state placeholder (not directly used) / 初始状态占位符（不直接使用）'
    )
    parser.add_argument(
        '--feature_num_1',
        type=int,
        default=16,
        help='Input dimension for mode head (actor1) / 模式头输入维度'
    )
    parser.add_argument(
        '--feature_num_2',
        type=int,
        default=15,
        help='Input dimension for resource heads (actor2) / 资源头输入维度'
    )

    # =========================================================================
    # Device and Execution Settings / 设备和执行设置
    # =========================================================================
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training/testing / 用于训练/测试的设备'
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA (force CPU usage) / 禁用 CUDA（强制使用 CPU）'
    )
    parser.add_argument(
        '--no_DDP',
        action='store_true',
        help='Disable distributed data parallel / 禁用分布式数据并行'
    )
    parser.add_argument(
        '--test',
        type=int,
        default=0,
        help='Switch to test mode / 切换到测试模式'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility / 用于可重复性的随机种子'
    )

    # =========================================================================
    # Training Control / 训练控制
    # =========================================================================
    parser.add_argument(
        '--max_learning_step',
        type=int,
        default=10000,
        help='Maximum number of training steps / 最大训练步数'
    )
    parser.add_argument(
        '--update_best_model_epochs',
        type=int,
        default=1,
        help='Epoch interval for updating best model / 更新最佳模型的轮次间隔'
    )

    # =========================================================================
    # Network Architecture (Actor) / 网络架构（演员）
    # =========================================================================
    # Actor head 1: Mode selection (CC vs NDA) / 模式选择
    parser.add_argument(
        '--actor1_embedding_dim',
        type=int,
        default=128,
        help='Embedding dimension for mode head / 模式头嵌入维度'
    )
    parser.add_argument(
        '--actor1_hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension for mode head / 模式头隐藏层维度'
    )
    parser.add_argument(
        '--actor1_action_space',
        type=int,
        default=1,
        help='Action space for mode head (binary) / 模式头动作空间（二元）'
    )

    # Actor head 2: Resource allocation / 资源分配
    parser.add_argument(
        '--actor2_embedding_dim',
        type=int,
        default=160,
        help='Embedding dimension for resource heads / 资源头嵌入维度'
    )
    parser.add_argument(
        '--actor2_hidden_dim',
        type=int,
        default=320,
        help='Hidden dimension for resource heads / 资源头隐藏层维度'
    )
    parser.add_argument(
        '--actor2_action_space',
        type=int,
        default=3,
        help='Number of resource allocation levels / 资源分配级别数量'
    )

    # =========================================================================
    # Network Architecture (Critic) / 网络架构（评论家）
    # =========================================================================
    parser.add_argument(
        '--critic_head_num',
        type=int,
        default=6,
        help='Number of attention heads in critic (legacy, unused) / 评论家注意力头数（遗留，未使用）'
    )

    # =========================================================================
    # Legacy Network Parameters (Unused) / 遗留网络参数（未使用）
    # =========================================================================
    parser.add_argument(
        '--v_range',
        type=float,
        default=6.,
        help='Entropy control range (legacy, unused) / 熵控制范围（遗留，未使用）'
    )
    parser.add_argument(
        '--encoder_head_num',
        type=int,
        default=4,
        help='Encoder attention heads (legacy, unused) / 编码器注意力头数（遗留，未使用）'
    )
    parser.add_argument(
        '--decoder_head_num',
        type=int,
        default=4,
        help='Decoder attention heads (legacy, unused) / 解码器注意力头数（遗留，未使用）'
    )
    parser.add_argument(
        '--n_encode_layers',
        type=int,
        default=1,
        help='Number of encoder layers (legacy, unused) / 编码器层数（遗留，未使用）'
    )
    parser.add_argument(
        '--normalization',
        default='layer',
        choices=['layer', 'batch'],
        help='Normalization type (legacy, unused) / 归一化类型（遗留，未使用）'
    )

    # =========================================================================
    # PPO Hyperparameters / PPO 超参数
    # =========================================================================
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.999,
        help='Reward discount factor / 奖励折扣因子'
    )
    parser.add_argument(
        '--K_epochs',
        type=int,
        default=3,
        help='Number of PPO update epochs per batch / 每批 PPO 更新轮数'
    )
    parser.add_argument(
        '--eps_clip',
        type=float,
        default=0.2,
        help='PPO clipping parameter / PPO 裁剪参数'
    )
    parser.add_argument(
        '--n_step',
        type=int,
        default=5,
        help='N-step return estimation / N 步回报估计'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=0.1,
        help='Maximum L2 norm for gradient clipping / 梯度裁剪的最大 L2 范数'
    )

    # =========================================================================
    # Epoch and Training Loop / 轮次和训练循环
    # =========================================================================
    parser.add_argument(
        '--epoch_start',
        type=int,
        default=0,
        help='Starting epoch number (for resuming) / 起始轮次号（用于恢复）'
    )
    parser.add_argument(
        '--epoch_end',
        type=int,
        default=200,
        help='Maximum training epoch / 最大训练轮次'
    )
    parser.add_argument(
        '--epoch_size',
        type=int,
        default=200,
        help='Number of instances per epoch / 每轮实例数量'
    )
    parser.add_argument(
        '--T_train',
        type=int,
        default=2000,
        help='Number of training iterations / 训练迭代次数'
    )
    parser.add_argument(
        '--decision_interval',
        type=int,
        default=1,
        help='Action decision interval in generations / 动作决策间隔（代数）'
    )

    # =========================================================================
    # Inference and Validation / 推理和验证
    # =========================================================================
    parser.add_argument(
        '--eval_only',
        action='store_true',
        default=False,
        help='Switch to inference mode (no training) / 切换到推理模式（不训练）'
    )
    parser.add_argument(
        '--Max_Eval',
        type=int,
        default=200000,
        help='Max evaluations for inference / 推理的最大评估次数'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=1024,
        help='Number of validation instances / 验证实例数量'
    )
    parser.add_argument(
        '--inference_interval',
        type=int,
        default=3,
        help='Interval between inference runs / 推理运行间隔'
    )
    parser.add_argument(
        '--greedy_rollout',
        action='store_true',
        help='Use greedy action selection / 使用贪婪动作选择'
    )

    # =========================================================================
    # Path Configuration (using PATHS from config) / 路径配置（使用 config 中的 PATHS）
    # =========================================================================
    parser.add_argument(
        '--log_dir',
        default=str(PATHS.log),
        help=f'Directory for TensorBoard logs / TensorBoard 日志目录 (default: {PATHS.log})'
    )
    parser.add_argument(
        '--output_modal_dir',
        default=str(PATHS.model),
        help=f'Directory for saving models / 保存模型的目录 (default: {PATHS.model})'
    )
    parser.add_argument(
        '--output_data_dir',
        default=str(PATHS.save / "running_data"),
        help=f'Directory for output data / 输出数据目录 (default: {PATHS.save / "running_data"})'
    )
    parser.add_argument(
        '--output_rollout_dir',
        default=str(PATHS.save / "rollout_data"),
        help=f'Directory for rollout output / 滚动输出目录 (default: {PATHS.save / "rollout_data"})'
    )
    parser.add_argument(
        '--test_dir',
        default=str(PATHS.test),
        help=f'Directory for test output / 测试输出目录 (default: {PATHS.test})'
    )

    # =========================================================================
    # Run Identification / 运行标识
    # =========================================================================
    parser.add_argument(
        '--run_name',
        default='model',
        help='Base name to identify the run / 标识运行的基础名称'
    )
    parser.add_argument(
        '--RL_agent',
        default='ppo',
        choices=['ppo'],
        help='RL training algorithm / RL 训练算法'
    )

    # =========================================================================
    # Checkpoint and Model Loading / 检查点和模型加载
    # =========================================================================
    parser.add_argument(
        '--no_saving',
        action='store_true',
        help='Disable saving checkpoints / 禁用保存检查点'
    )
    parser.add_argument(
        '--checkpoint_epochs',
        type=int,
        default=1,
        help='Save checkpoint every N epochs (0 to disable) / 每N轮保存检查点（0为禁用）'
    )
    parser.add_argument(
        '--load_path',
        default=None,
        help='Path to load model parameters and optimizer state / 加载模型参数和优化器状态的路径'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Resume from checkpoint directory / 从检查点目录恢复'
    )
    parser.add_argument(
        '--load_path_for_test',
        default='save_dir/ppo_model/model_CEC2013LSGO_1.0e+06_10_20_0.0001_20250829T164119/epoch-29.pt',
        help='Path to load model for testing / 加载测试模型的路径'
    )

    # =========================================================================
    # Logging and Visualization / 日志和可视化
    # =========================================================================
    parser.add_argument(
        '--no_tb',
        action='store_true',
        help='Disable TensorBoard logging / 禁用 TensorBoard 日志'
    )
    parser.add_argument(
        '--no_progress_bar',
        action='store_true',
        help='Disable progress bar / 禁用进度条'
    )
    parser.add_argument(
        '--show_figs',
        action='store_true',
        help='Enable figure logging / 启用图表日志'
    )
    parser.add_argument(
        '--log_step',
        type=int,
        default=1,
        help='Log info every N gradient steps / 每N梯度步记录信息'
    )

    # =========================================================================
    # Debug and Development / 调试和开发
    # =========================================================================
    parser.add_argument(
        '--use_assert',
        action='store_true',
        help='Enable assertions / 启用断言'
    )
    parser.add_argument(
        '--dataset_path',
        default=None,
        help='Path to dataset (if loading from file) / 数据集路径（如果从文件加载）'
    )
    parser.add_argument(
        '--enable_analysis',
        action='store_true',
        default=False,
        help='Enable FeatureSE weight analysis during testing (default: disabled) / 在测试期间启用 FeatureSE 权重分析（默认：禁用）'
    )

    # =========================================================================
    # Parse Arguments / 解析参数
    # =========================================================================
    opts = parser.parse_args(args)

    # =========================================================================
    # Post-Processing / 后处理
    # =========================================================================

    # Derived calculation / 派生计算
    opts.ns = int(opts.max_fes // opts.subFEs)

    # Device configuration / 设备配置
    if opts.no_cuda or not torch.cuda.is_available():
        opts.device = 'cpu'
        opts.use_cuda = 0
    else:
        opts.use_cuda = 1

    # Distributed training / 分布式训练
    opts.world_size = 1
    opts.distributed = False
    # Uncomment to enable distributed training / 取消注释以启用分布式训练
    # opts.world_size = torch.cuda.device_count()
    # opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'

    # Generate run name with timestamp / 生成带时间戳的运行名称
    if opts.resume:
        opts.run_name = opts.resume.split('/')[-2]
    else:
        opts.run_name = "{}_{}_{}_{}_{}_{}_{}".format(
            opts.run_name,
            opts.divide_method,
            "{:.1e}".format(opts.max_fes),
            opts.m,
            opts.sub_popsize,
            opts.lr_model,
            time.strftime("%Y%m%dT%H%M%S")
        )

    # Build output directories / 构建输出目录
    opts.model_save_dir = os.path.join(
        opts.output_modal_dir,
        opts.run_name
    ) if not opts.no_saving else None

    opts.data_save_dir = os.path.join(
        opts.output_data_dir,
        opts.run_name
    ) if not opts.no_saving else None

    opts.rollout_save_dir = os.path.join(
        opts.output_rollout_dir,
        opts.run_name
    ) if not opts.no_saving else None

    opts.log_dir = os.path.join(
        opts.log_dir,
        opts.run_name
    )

    opts.test_dir = os.path.join(
        opts.test_dir,
        opts.run_name
    )

    return opts
