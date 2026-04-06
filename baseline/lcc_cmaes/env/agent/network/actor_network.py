"""
Actor 网络模块 / Actor Network Module.

定义用于强化学习的 Actor 策略网络。
Defines the Actor policy network for reinforcement learning.

该网络根据当前状态输出动作概率分布。
This network outputs action probability distributions based on current states.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from env.agent.network.common_network import MLP_for_actor

# Add parent directory to path for local utils / 添加父目录到路径以使用本地 utils
import sys
import os
_current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from utils.options import get_options

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)

# 获取配置 / Get configuration
opts = get_options()


class Actor(nn.Module):
    """
    Actor 策略网络 / Actor Policy Network.

    使用多层感知机将状态映射到动作概率分布。
    Uses MLP to map states to action probability distributions.

    Attributes:
        input_dim (int): 输入状态维度 / Input state dimension
        CC_method_net (MLP_for_actor): 主网络 / Main network

    Example:
        >>> actor = Actor()
        >>> state = torch.randn(10, 12)  # batch_size=10, feature_dim=12
        >>> action, log_prob, entropy = actor(state)
    """

    # ==========================================
    # 常量定义 / Constants
    # ==========================================
    DEFAULT_EMBEDDING_DIM = 64
    DEFAULT_HIDDEN_DIM = 64

    def __init__(self) -> None:
        """
        初始化 Actor 网络 / Initialize Actor Network.

        Raises:
            RuntimeError: 当配置无效时 / When configuration is invalid
        """
        super(Actor, self).__init__()

        self.input_dim = opts.feature_num
        # Use device from opts if available, otherwise auto-detect / 使用 opts 中的设备（如果有），否则自动检测
        if hasattr(opts, 'device'):
            self._device = torch.device(opts.device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 网络创建 / Network creation
        self.CC_method_net = MLP_for_actor(
            input_dim=self.input_dim,
            embedding_dim=self.DEFAULT_EMBEDDING_DIM,
            hidden_dim=self.DEFAULT_HIDDEN_DIM,
            output_dim=opts.action_space
        )

        logger.info(f"Actor 初始化完成 / Actor initialized: input_dim={self.input_dim}")

    def get_parameter_number(self) -> dict[str, int]:
        """
        获取模型参数数量 / Get model parameter count.

        Returns:
            包含总参数和可训练参数数量的字典 / Dict with total and trainable parameter counts:
                - 'Actor: Total' (int): 总参数数 / Total parameters
                - 'Trainable' (int): 可训练参数数 / Trainable parameters
        """
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Actor: Total': total_num, 'Trainable': trainable_num}

    def forward(
        self,
        state: torch.Tensor,
        fixed_action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 / Forward propagation.

        Args:
            state: 输入状态张量 / Input state tensor, shape (batch_size, input_dim)
            fixed_action: 固定动作（用于测试）/ Fixed action for testing (optional)

        Returns:
            包含动作、对数概率和熵的元组 / Tuple of (action, log_prob, entropy):
                - action (Tensor): 采样的动作 / Sampled action
                - ll (Tensor): 对数概率 / Log probability
                - entropy (Tensor): 策略熵 / Policy entropy
        """
        # 将状态移到设备 / Move state to device
        state = state.to(self._device)

        # 计算动作分数 / Compute action scores
        score = self.CC_method_net(state)

        # 计算动作概率 / Compute action probabilities
        action_prob = F.softmax(score, dim=-1)

        # 创建分类分布 / Create categorical distribution
        action_dist = torch.distributions.Categorical(action_prob)

        # 采样或使用固定动作 / Sample or use fixed action
        if fixed_action is not None:
            action = fixed_action.to(self._device)
        else:
            action = action_dist.sample()

        # 计算对数概率和熵 / Compute log probability and entropy
        ll = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action, ll, entropy


# ==========================================
# 测试代码 / Test Code
# ==========================================
if __name__ == '__main__':
    # 配置日志 / Configure logging
    logging.basicConfig(level=logging.INFO)

    # 创建 Actor 实例 / Create Actor instance
    actor = Actor()

    # 打印模型参数 / Print model parameters
    logger.info(f"模型参数 / Model parameters: {actor.get_parameter_number()}")

    # 测试数据 / Test data
    state = torch.tensor([
        [0.2, 0.9, 0.1, 0.5, 1.0, 1.0, 5.0, 2.0, 1.0, 5002, 500, 0.01],
        [0.2, 0.9, 0.1, 0.5, 10.0, 1.0, 5.0, 2.0, 10.0, 0.2, 50, 0.01]
    ])

    # 前向传播测试 / Forward propagation test
    action, log_prob, entropy = actor(state)

    logger.info(f"动作 / Action: {action}")
    logger.info(f"对数概率 / Log prob: {log_prob}")
    logger.info(f"熵 / Entropy: {entropy}")
