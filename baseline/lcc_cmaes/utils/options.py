"""
Command-Line Options Configuration for CMA-ES PPO
CMA-ES PPO 的命令行选项配置

This module provides a centralized configuration system for all training and testing
parameters of the CMA-ES optimization algorithm with PPO-based strategy selection.
本模块提供了基于 PPO 策略选择的 CMA-ES 优化算法的所有训练和测试参数的集中配置系统。

Usage / 使用方法:
    Default configuration / 默认配置:
        >>> opts = get_options()

    Custom configuration / 自定义配置:
        >>> opts = get_options(['--epoch_end', '100', '--lr_model', '0.001'])

    Training / 训练:
        python run.py --RL_agent ppo --epoch_end 200

    Testing / 测试:
        python run.py --RL_agent ppo --load_path_for_test repository/save_dir/ppo_model/model_name/epoch-9.pt

Author: CMA-ES PPO Project
Date: 2026-03-15
"""

# =============================================================================
# Standard Library Imports / 标准库导入
# =============================================================================
import os
import time
import argparse
from typing import Optional


# =============================================================================
# Configuration Function / 配置函数
# =============================================================================

def get_options(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parse and return command-line arguments for CMA-ES PPO training and testing.
    解析并返回 CMA-ES PPO 训练和测试的命令行参数。

    This function configures all hyperparameters, paths, and settings for the
    training pipeline, including CMA-ES parameters, PPO network architecture,
    optimization settings, and logging configurations.
    此函数配置训练管道的所有超参数、路径和设置，包括 CMA-ES 参数、
    PPO 网络架构、优化设置和日志配置。

    Args:
        args (Optional[list]): Argument list to parse (None for sys.argv). If provided,
                               these arguments are parsed instead of sys.argv.
                               / 要解析的参数列表（None 表示使用 sys.argv）。如果提供，
                               将解析这些参数而不是 sys.argv。

    Returns:
        argparse.Namespace: Parsed configuration options with dynamically computed
                           fields such as modal_save_dir, data_save_dir, etc.
                           / 解析后的配置选项，包含动态计算的字段如
                           modal_save_dir、data_save_dir 等。

    Example:
        >>> # Use default configuration / 使用默认配置
        >>> opts = get_options()
        >>> print(opts.lr_model)  # 0.0006
        >>> print(opts.modal_save_dir)  # repository/save_dir/ppo_model/model_...

        >>> # Override specific parameters / 覆盖特定参数
        >>> opts = get_options(['--epoch_end', '100', '--lr_model', '0.001'])
        >>> print(opts.epoch_end)  # 100
    """
    # =========================================================================
    # Initialize Argument Parser / 初始化参数解析器
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="LCC_CMAES: Large-scale Cooperative Coevolution with CMA-ES and PPO"
    )

    # =========================================================================
    # CMA-ES Algorithm Settings / CMA-ES 算法设置
    # =========================================================================
    parser.add_argument(
        '--backbone',
        default='cmaes',
        choices=['cmaes'],
        help='Backbone optimization algorithm / 基础优化算法'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=10,
        help='Number of subgroups for cooperative coevolution / 协同进化的子组数量'
    )
    parser.add_argument(
        '--sub_popsize',
        type=int,
        default=10,
        help='Population size of each subgroup / 每个子组的种群大小'
    )
    parser.add_argument(
        '--max_fes',
        type=int,
        default=3e6,
        help='Maximum number of function evaluations / 最大函数评估次数'
    )
    parser.add_argument(
        '--output_init_cma_info',
        type=bool,
        default=False,
        help='Output the initial CMA-ES information / 输出初始 CMA-ES 信息'
    )
    parser.add_argument(
        '--subFEs',
        type=int,
        default=50,
        help='Function evaluations allocated for each subspace optimization / 每个子空间优化分配的函数评估次数'
    )
    parser.add_argument(
        '--ns',
        type=int,
        default=0,
        help='Number of action selections for CMA-ES (computed automatically if 0) / CMA-ES 动作选择次数（若为0则自动计算）'
    )

    # =========================================================================
    # Cooperative Coevolution Settings / 协同进化设置
    # =========================================================================
    parser.add_argument(
        '--action_space',
        type=int,
        default=3,
        help='Number of decomposition strategy actions / 分解策略动作的数量'
    )

    # =========================================================================
    # Problem Settings / 问题设置
    # =========================================================================
    parser.add_argument(
        '--divide_method',
        default="CEC2013LSGO",
        choices=["CEC2013LSGO", "BNS"],
        help='Problem set division method / 问题集划分方法'
    )

    # =========================================================================
    # Rollout Settings / 采样设置
    # =========================================================================
    parser.add_argument(
        '--one_problem_batch_size',
        type=int,
        default=3,
        help='Number of instances for each problem during rollout / 采样期间每个问题的实例数量'
    )
    parser.add_argument(
        '--per_eval_time',
        type=int,
        default=1,
        help='Number of evaluations for each instance / 每个实例的评估次数'
    )

    # =========================================================================
    # PPO Training Settings / PPO 训练设置
    # =========================================================================
    parser.add_argument(
        '--each_question_batch_num',
        type=int,
        default=4,
        help='Number of parallel instances for each problem / 每个问题的并行实例数量'
    )
    parser.add_argument(
        '--lr_critic',
        type=float,
        default=6e-4,
        help='Learning rate for the critic network / Critic 网络的学习率'
    )
    parser.add_argument(
        '--lr_model',
        type=float,
        default=6e-4,
        help='Learning rate for the actor network / Actor 网络的学习率'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.99,
        help='Learning rate decay factor per epoch / 每轮的学习率衰减因子'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.999,
        help='Reward discount factor for future rewards / 未来奖励的折扣因子'
    )
    parser.add_argument(
        '--K_epochs',
        type=int,
        default=3,
        help='Number of PPO optimization epochs per update / 每次更新的 PPO 优化轮数'
    )
    parser.add_argument(
        '--eps_clip',
        type=float,
        default=0.2,
        help='PPO clipping parameter for ratio truncation / PPO 比率截断的裁剪参数'
    )
    parser.add_argument(
        '--n_step',
        type=int,
        default=5,
        help='N-step return for reward estimation / 奖励估计的 N 步回报'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=0.1,
        help='Maximum L2 norm for gradient clipping / 梯度裁剪的最大 L2 范数'
    )

    # =========================================================================
    # State and Feature Settings / 状态和特征设置
    # =========================================================================
    parser.add_argument(
        '--state',
        default=[0.0 for _ in range(18)],
        help='Initial state of actor (for actor instance creation only) / Actor 的初始状态（仅用于创建 actor 实例）'
    )
    parser.add_argument(
        '--feature_num',
        type=int,
        default=58,
        help='Number of features for each problem instance / 每个问题实例的特征数量'
    )

    # =========================================================================
    # Training Epoch Settings / 训练轮次设置
    # =========================================================================
    parser.add_argument(
        '--epoch_start',
        type=int,
        default=0,
        help='Starting epoch number (for resuming training) / 起始轮次（用于恢复训练）'
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
        help='Number of instances per epoch during training / 训练期间每轮的实例数量'
    )
    parser.add_argument(
        '--T_train',
        type=int,
        default=2000,
        help='Number of training iterations / 训练迭代次数'
    )
    parser.add_argument(
        '--max_learning_step',
        type=int,
        default=10000,
        help='Maximum number of learning steps / 最大学习步数'
    )
    parser.add_argument(
        '--update_best_model_epochs',
        type=int,
        default=1,
        help='Frequency of updating the best model / 更新最佳模型的频率（轮次）'
    )

    # =========================================================================
    # Network Architecture Settings / 网络架构设置
    # =========================================================================
    parser.add_argument(
        '--encoder_head_num',
        type=int,
        default=4,
        help='Number of attention heads in the encoder / 编码器中的注意力头数量'
    )
    parser.add_argument(
        '--decoder_head_num',
        type=int,
        default=4,
        help='Number of attention heads in the decoder / 解码器中的注意力头数量'
    )
    parser.add_argument(
        '--critic_head_num',
        type=int,
        default=6,
        help='Number of attention heads in the critic / Critic 中的注意力头数量'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=16,
        help='Dimension of input embeddings / 输入嵌入的维度'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=16,
        help='Dimension of hidden layers in encoder/decoder / 编码器/解码器隐藏层的维度'
    )
    parser.add_argument(
        '--n_encode_layers',
        type=int,
        default=1,
        help='Number of stacked layers in the encoder / 编码器堆叠层数'
    )
    parser.add_argument(
        '--normalization',
        default='layer',
        choices=['layer', 'batch'],
        help='Normalization type (layer or batch) / 归一化类型（layer 或 batch）'
    )
    parser.add_argument(
        '--v_range',
        type=float,
        default=6.0,
        help='Entropy control parameter / 熵控制参数'
    )

    # =========================================================================
    # Decision Settings / 决策设置
    # =========================================================================
    parser.add_argument(
        '--decision_interval',
        type=int,
        default=1,
        help='Make action decision every N generations / 每 N 代进行一次动作决策'
    )

    # =========================================================================
    # Paths Configuration / 路径配置
    # =========================================================================
    parser.add_argument(
        '--log_dir',
        default='repository/save_dir/log/',
        help='Directory for TensorBoard logs / TensorBoard 日志目录'
    )
    parser.add_argument(
        '--output_modal_dir',
        default='repository/save_dir/ppo_model/',
        help='Directory for saved model checkpoints / 保存模型检查点的目录'
    )
    parser.add_argument(
        '--output_data_dir',
        default='repository/save_dir/running_data/',
        help='Directory for running data / 运行数据目录'
    )
    parser.add_argument(
        '--output_rollout_dir',
        default='repository/save_dir/rollout_data/',
        help='Directory for rollout data / 采样数据目录'
    )

    # =========================================================================
    # Run Configuration / 运行配置
    # =========================================================================
    parser.add_argument(
        '--run_name',
        default='model',
        help='Base name to identify the run / 识别运行的名称'
    )
    parser.add_argument(
        '--RL_agent',
        default='ppo',
        choices=['ppo'],
        help='RL training algorithm / RL 训练算法'
    )

    # =========================================================================
    # Mode Settings / 模式设置
    # =========================================================================
    parser.add_argument(
        '--test',
        type=int,
        default=0,
        help='Switch to test mode (1 for test, 0 for training) / 切换到测试模式（1为测试，0为训练）'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        default=False,
        help='Switch to inference mode (no training) / 切换到推理模式（不训练）'
    )
    parser.add_argument(
        '--greedy_rollout',
        action='store_true',
        help='Use greedy strategy for rollout / 使用贪婪策略进行采样'
    )

    # =========================================================================
    # Model Loading and Resuming / 模型加载和恢复
    # =========================================================================
    parser.add_argument(
        '--load_path',
        default='baseline/lcc_cmaes/model/epoch-9.pt',
        help='Path to load model parameters and optimizer state from / 加载模型参数和优化器状态的路径'
    )
    parser.add_argument(
        '--load_path_for_test',
        default='baseline/lcc_cmaes/model/epoch-9.pt',
        help='Path to load model for testing / 加载测试模型的路径 '
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Resume from previous checkpoint file (path to epoch-X.pt file) / 从之前的检查点恢复（epoch-X.pt 文件路径）'
    )

    # =========================================================================
    # Hardware and Device Settings / 硬件和设备设置
    # =========================================================================
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Disable CUDA (use CPU only) / 禁用 CUDA（仅使用 CPU）'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training/testing / 用于训练/测试的设备'
    )

    # =========================================================================
    # Logging and Monitoring Settings / 日志和监控设置
    # =========================================================================
    parser.add_argument(
        '--no_tb',
        action='store_true',
        help='Disable TensorBoard logging / 禁用 TensorBoard 日志'
    )
    parser.add_argument(
        '--no_saving',
        action='store_true',
        help='Disable saving model checkpoints / 禁用保存模型检查点'
    )
    parser.add_argument(
        '--no_progress_bar',
        action='store_true',
        help='Disable progress bar display / 禁用进度条显示'
    )
    parser.add_argument(
        '--show_figs',
        action='store_true',
        help='Enable figure logging in TensorBoard / 在 TensorBoard 中启用图形记录'
    )
    parser.add_argument(
        '--log_step',
        type=int,
        default=1,
        help='Log information every N gradient steps / 每 N 梯度步记录一次信息'
    )
    parser.add_argument(
        '--checkpoint_epochs',
        type=int,
        default=1,
        help='Save checkpoint every N epochs (0 for no checkpoints) / 每 N 轮保存检查点（0表示不保存）'
    )

    # =========================================================================
    # Validation Settings / 验证设置
    # =========================================================================
    parser.add_argument(
        '--Max_Eval',
        type=int,
        default=3e6,
        help='Maximum function evaluations for inference / 推理的最大函数评估次数'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=1024,
        help='Number of instances for validation/inference / 验证/推理的实例数量'
    )
    parser.add_argument(
        '--inference_interval',
        type=int,
        default=3,
        help='Interval for inference during training / 训练期间的推理间隔'
    )

    # =========================================================================
    # Dataset Settings / 数据集设置
    # =========================================================================
    parser.add_argument(
        '--dataset_path',
        default=None,
        help='Path to custom dataset / 自定义数据集路径'
    )

    # =========================================================================
    # Miscellaneous Settings / 其他设置
    # =========================================================================
    parser.add_argument(
        '--seed',
        type=int,
        default=666,
        help='Random seed for reproducibility / 用于可复现性的随机种子'
    )
    parser.add_argument(
        '--use_assert',
        action='store_true',
        help='Enable assertion checks in code / 启用代码中的断言检查'
    )
    parser.add_argument(
        '--no_DDP',
        action='store_true',
        help='Disable distributed data parallel training / 禁用分布式数据并行训练'
    )

    # =========================================================================
    # Parse Arguments / 解析参数
    # =========================================================================
    opts = parser.parse_args(args)

    # =========================================================================
    # Distributed Training Configuration / 分布式训练配置
    # =========================================================================
    opts.world_size = 1
    opts.distributed = False
    # Uncomment for multi-GPU training / 取消注释以进行多 GPU 训练:
    # opts.world_size = torch.cuda.device_count()
    # opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'

    # =========================================================================
    # Device Configuration / 设备配置
    # =========================================================================
    opts.use_cuda = 1  # Set to 0 to disable CUDA / 设为 0 以禁用 CUDA

    # =========================================================================
    # Computed Parameters / 计算参数
    # =========================================================================
    # Compute number of selections / 计算选择次数
    opts.ns = opts.max_fes // (opts.m * opts.subFEs) if opts.ns == 0 else opts.ns

    # Generate run name with timestamp / 生成带时间戳的运行名称
    if not opts.resume:
        opts.run_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            opts.run_name,
            opts.divide_method,
            "{:.1e}".format(opts.max_fes),
            opts.m,
            opts.sub_popsize,
            opts.lr_model,
            opts.subFEs,
            time.strftime("%Y%m%dT%H%M%S")
        )
    else:
        # Extract run name from resume path / 从恢复路径提取运行名称
        opts.run_name = opts.resume.split('/')[-2]

    # =========================================================================
    # Compute Full Save Paths / 计算完整保存路径
    # =========================================================================
    if not opts.no_saving:
        opts.modal_save_dir = os.path.join(opts.output_modal_dir, opts.run_name)
        opts.data_save_dir = os.path.join(opts.output_data_dir, opts.run_name)
        opts.rollout_save_dir = os.path.join(opts.output_rollout_dir, opts.run_name)
    else:
        opts.modal_save_dir = None
        opts.data_save_dir = None
        opts.rollout_save_dir = None

    # Add timestamp to log directory / 添加时间戳到日志目录
    opts.log_dir = os.path.join(opts.log_dir, opts.run_name)

    return opts
