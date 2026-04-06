"""
TensorBoard Logger for PPO Training
PPO 训练的 TensorBoard 记录器

This module provides comprehensive logging utilities for monitoring PPO training,
including loss components, policy statistics, value function metrics, and gradients.

本模块提供用于监控 PPO 训练的综合记录工具，包括损失组件、策略统计、
价值函数指标和梯度。

Main Functions / 主要函数:
    log_train_step: Log metrics for each training step
    log_epoch_summary: Log summary at end of each epoch
    log_problem_results: Log results for each problem

Author: WCC Project
Date: 2026-03-11
"""

import math
from typing import Dict, Optional, Any

import torch
import numpy as np


def log_train_step(
    tb_logger,
    step: int,
    epoch: int,
    problem_id: int,
    # Loss components / 损失组件
    loss_total: float,
    loss_pi_mode: float,
    loss_pi_res: float,
    loss_vf: float,
    loss_ent: float,
    # Policy statistics / 策略统计
    lr_actor: float,
    lr_critic: float,
    entropy_mode: float,
    entropy_res: float,
    kl_mode: Optional[float] = None,
    kl_res: Optional[float] = None,
    # Value function statistics / 价值函数统计
    rewards_mean: float = 0.0,
    rewards_std: float = 0.0,
    advantages_mean: float = 0.0,
    advantages_std: float = 0.0,
    v_pred_mean: float = 0.0,
    # Gradient statistics / 梯度统计
    grad_norm_actor: float = 0.0,
    grad_norm_critic: float = 0.0,
    grad_norm_clipped_actor: float = 0.0,
    grad_norm_clipped_critic: float = 0.0,
    # Action statistics / 动作统计
    mode_selection_rate: float = 0.0,
    resource_dist: Optional[torch.Tensor] = None,
    # State statistics / 状态统计
    state_mean: float = 0.0,
    state_std: float = 0.0,
    nan_ratio: float = 0.0,
) -> None:
    """
    Log training metrics to TensorBoard.
    将训练指标记录到 TensorBoard。

    Args:
        tb_logger: TensorBoard logger instance / TensorBoard 记录器实例
        step: Global training step / 全局训练步数
        epoch: Current epoch / 当前 epoch
        problem_id: Problem function ID / 问题函数 ID

        Loss components / 损失组件:
            loss_total: Total loss / 总损失
            loss_pi_mode: Mode policy loss / 模式策略损失
            loss_pi_res: Resource policy loss / 资源策略损失
            loss_vf: Value function loss / 价值函数损失
            loss_ent: Entropy regularization loss / 熵正则化损失

        Policy statistics / 策略统计:
            lr_actor: Actor learning rate / 演员学习率
            lr_critic: Critic learning rate / 评论家学习率
            entropy_mode: Mode entropy / 模式熵
            entropy_res: Resource entropy / 资源熵
            kl_mode: Mode KL divergence / 模式 KL 散度
            kl_res: Resource KL divergence / 资源 KL 散度

        Value function statistics / 价值函数统计:
            rewards_mean: Mean reward / 平均奖励
            rewards_std: Reward standard deviation / 奖励标准差
            advantages_mean: Mean advantage / 平均优势
            advantages_std: Advantage standard deviation / 优势标准差
            v_pred_mean: Mean predicted value / 平均预测价值

        Gradient statistics / 梯度统计:
            grad_norm_actor: Actor gradient norm / 演员梯度范数
            grad_norm_critic: Critic gradient norm / 评论家梯度范数
            grad_norm_clipped_actor: Clipped actor gradient norm / 裁剪后的演员梯度范数
            grad_norm_clipped_critic: Clipped critic gradient norm / 裁剪后的评论家梯度范数

        Action statistics / 动作统计:
            mode_selection_rate: Rate of selecting NDA (mode=1) / 选择 NDA 的比例
            resource_dist: Resource action distribution [3] / 资源动作分布

        State statistics / 状态统计:
            state_mean: Mean state value / 平均状态值
            state_std: State standard deviation / 状态标准差
            nan_ratio: Ratio of NaN values in state / 状态中 NaN 的比例
    """
    if tb_logger is None:
        return

    # =========================================================================
    # Loss / 损失
    # =========================================================================
    tb_logger.log_value(f'loss/total', loss_total, step)
    tb_logger.log_value(f'loss/pi_mode', loss_pi_mode, step)
    tb_logger.log_value(f'loss/pi_resource', loss_pi_res, step)
    tb_logger.log_value(f'loss/value', loss_vf, step)
    tb_logger.log_value(f'loss/entropy', loss_ent, step)

    # =========================================================================
    # Learning Rate / 学习率
    # =========================================================================
    tb_logger.log_value(f'lr/actor', lr_actor, step)
    tb_logger.log_value(f'lr/critic', lr_critic, step)

    # =========================================================================
    # Policy Statistics / 策略统计
    # =========================================================================
    tb_logger.log_value(f'policy/entropy_mode', entropy_mode, step)
    tb_logger.log_value(f'policy/entropy_resource', entropy_res, step)

    if kl_mode is not None:
        tb_logger.log_value(f'policy/kl_mode', kl_mode, step)
    if kl_res is not None:
        tb_logger.log_value(f'policy/kl_resource', kl_res, step)

    tb_logger.log_value(f'policy/mode_selection_rate (NDA)', mode_selection_rate, step)

    if resource_dist is not None:
        for i, prob in enumerate(resource_dist):
            tb_logger.log_value(f'policy/resource_prob_{i}', prob.item(), step)

    # =========================================================================
    # Value Function / 价值函数
    # =========================================================================
    tb_logger.log_value(f'value/reward_mean', rewards_mean, step)
    tb_logger.log_value(f'value/reward_std', rewards_std, step)
    tb_logger.log_value(f'value/advantage_mean', advantages_mean, step)
    tb_logger.log_value(f'value/advantage_std', advantages_std, step)
    tb_logger.log_value(f'value/pred_mean', v_pred_mean, step)

    # =========================================================================
    # Gradients / 梯度
    # =========================================================================
    tb_logger.log_value(f'grad/actor', grad_norm_actor, step)
    tb_logger.log_value(f'grad/critic', grad_norm_critic, step)
    tb_logger.log_value(f'grad/clipped_actor', grad_norm_clipped_actor, step)
    tb_logger.log_value(f'grad/clipped_critic', grad_norm_clipped_critic, step)

    # =========================================================================
    # State / 状态
    # =========================================================================
    tb_logger.log_value(f'state/mean', state_mean, step)
    tb_logger.log_value(f'state/std', state_std, step)
    tb_logger.log_value(f'state/nan_ratio', nan_ratio, step)


def log_epoch_summary(
    tb_logger,
    epoch: int,
    step: int,
    # Performance metrics / 性能指标
    best_fitness: float,
    avg_fitness: float,
    baseline_fitness: float,
    improvement: float,
    # Training progress / 训练进度
    problems_completed: int,
    total_problems: int,
    # Timing / 计时
    epoch_time: float,
) -> None:
    """
    Log epoch-level summary to TensorBoard.
    记录 epoch 级别的摘要到 TensorBoard。

    Args:
        tb_logger: TensorBoard logger instance / TensorBoard 记录器实例
        epoch: Current epoch number / 当前 epoch 编号
        step: Current global step / 当前全局步数

        Performance metrics / 性能指标:
            best_fitness: Best fitness achieved this epoch / 本 epoch 达到的最佳适应度
            avg_fitness: Average fitness / 平均适应度
            baseline_fitness: Baseline fitness for comparison / 基线适应度用于比较
            improvement: Improvement over baseline / 相对于基线的改进

        Training progress / 训练进度:
            problems_completed: Number of problems completed / 完成的问题数量
            total_problems: Total number of problems / 总问题数量

        Timing / 计时:
            epoch_time: Time taken for this epoch / 本 epoch 耗时
    """
    if tb_logger is None:
        return

    # Performance / 性能
    tb_logger.log_value(f'epoch/best_fitness', best_fitness, step)
    tb_logger.log_value(f'epoch/avg_fitness', avg_fitness, step)
    tb_logger.log_value(f'epoch/baseline_diff', avg_fitness - baseline_fitness, step)
    tb_logger.log_value(f'epoch/improvement_ratio', improvement, step)

    # Progress / 进度
    tb_logger.log_value(f'epoch/problems_completed', problems_completed, step)
    tb_logger.log_value(f'epoch/completion_rate', problems_completed / total_problems, step)

    # Timing / 计时
    tb_logger.log_value(f'epoch/time', epoch_time, step)


def log_problem_results(
    tb_logger,
    epoch: int,
    problem_id: int,
    step: int,
    # Fitness results / 适应度结果
    final_fitness: float,
    best_fitness: float,
    baseline_fitness: float,
    # Action distribution / 动作分布
    mode_usage: Dict[int, int],  # {mode: count}
    resource_usage: Dict[int, int],  # {resource_level: count}
) -> None:
    """
    Log results for a single problem.
    记录单个问题的结果。

    Args:
        tb_logger: TensorBoard logger instance / TensorBoard 记录器实例
        epoch: Current epoch / 当前 epoch
        problem_id: Problem function ID / 问题函数 ID
        step: Current global step / 当前全局步数

        Fitness results / 适应度结果:
            final_fitness: Final fitness value / 最终适应度值
            best_fitness: Best fitness achieved / 达到的最佳适应度
            baseline_fitness: Baseline fitness / 基线适应度

        Action distribution / 动作分布:
            mode_usage: Count of each mode selected / 每种模式选择的计数
            resource_usage: Count of each resource level / 每种资源级别的计数
    """
    if tb_logger is None:
        return

    # Fitness / 适应度
    tb_logger.log_value(f'problem_{problem_id}/final_fitness', final_fitness, step)
    tb_logger.log_value(f'problem_{problem_id}/best_fitness', best_fitness, step)
    tb_logger.log_value(f'problem_{problem_id}/baseline_diff', best_fitness - baseline_fitness, step)

    # Action distribution / 动作分布
    total_actions = sum(mode_usage.values()) if mode_usage else 1
    tb_logger.log_value(f'problem_{problem_id}/mode_NDA_rate', mode_usage.get(1, 0) / total_actions, step)

    for res_level, count in resource_usage.items():
        tb_logger.log_value(f'problem_{problem_id}/resource_{res_level}_rate', count / total_actions, step)


def compute_statistics(
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    v_pred: torch.Tensor,
    state: torch.Tensor,
    mode_actions: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute statistics for logging.
    计算用于记录的统计量。

    Args:
        rewards: Reward tensor [T*B] / 奖励张量
        advantages: Advantage tensor [T*B] / 优势张量
        v_pred: Value prediction tensor [T*B] / 价值预测张量
        state: State tensor [T*B, F] / 状态张量
        mode_actions: Mode actions [T*B] / 模式动作

    Returns:
        Dictionary of computed statistics / 计算的统计量字典
    """
    stats = {}

    # Rewards / 奖励
    if rewards.numel() > 0:
        stats['rewards_mean'] = rewards.mean().item()
        stats['rewards_std'] = rewards.std().item() if rewards.numel() > 1 else 0.0

    # Advantages / 优势
    if advantages.numel() > 0:
        stats['advantages_mean'] = advantages.mean().item()
        stats['advantages_std'] = advantages.std().item() if advantages.numel() > 1 else 0.0

    # Value predictions / 价值预测
    if v_pred.numel() > 0:
        stats['v_pred_mean'] = v_pred.mean().item()

    # State / 状态
    if state.numel() > 0:
        stats['state_mean'] = state.mean().item()
        stats['state_std'] = state.std().item() if state.numel() > 1 else 0.0
        stats['nan_ratio'] = torch.isnan(state).float().mean().item()

    # Mode selection / 模式选择
    if mode_actions.numel() > 0:
        stats['mode_selection_rate'] = mode_actions.float().mean().item()

    return stats


def create_histogram_if_available(
    tb_logger,
    tag: str,
    values: torch.Tensor,
    step: int,
    bins: int = 50
) -> None:
    """
    Create histogram if TensorBoard logger supports it.
    如果 TensorBoard 记录器支持，则创建直方图。

    Args:
        tb_logger: TensorBoard logger instance / TensorBoard 记录器实例
        tag: Tag for the histogram / 直方图标签
        values: Values to histogram / 要直方图化的值
        step: Current step / 当前步数
        bins: Number of bins / 箱数
    """
    if tb_logger is None or values is None or values.numel() == 0:
        return

    try:
        # Check if logger supports histogram / 检查记录器是否支持直方图
        if hasattr(tb_logger, 'log_histogram'):
            tb_logger.log_histogram(tag, values.cpu().numpy(), step)
    except Exception:
        # Histogram logging not supported / 直方图记录不支持
        pass
