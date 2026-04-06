"""
自适应动作选择器模块 / Adaptive Action Selector Module.

此模块提供了一个基于多臂老虎机（Multi-Armed Bandit）的自适应动作选择器，
使用 softmax 策略动态选择最优动作。
This module provides an adaptive action selector based on Multi-Armed Bandit,
using softmax strategy to dynamically select optimal actions.

主要组件 / Main Components:
    - ActionAdapter: 自适应动作选择器类 / Adaptive action selector class

使用示例 / Usage Example:
    >>> from env.optimizer.cc_cmaes.adapter import ActionAdapter
    >>> adapter = ActionAdapter(actions=['MiVD', 'MaVD', 'RaVD'], layers=5)
    >>> action = adapter.decide()
    >>> adapter.update('MiVD', contribution_ratio=0.1)
"""

import logging
from typing import List, Union

import numpy as np
from scipy.special import softmax

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)


class ActionAdapter:
    """
    自适应动作选择器 / Adaptive Action Selector.

    基于历史表现动态选择动作，使用 softmax 策略进行概率选择。
    Dynamically selects actions based on historical performance,
    using softmax strategy for probabilistic selection.

    该选择器维护一个性能矩阵，每个动作有多层历史记录，
    通过 softmax 转换为选择概率。
    The selector maintains a performance matrix with multiple layers of
    historical records for each action, converted to selection probabilities via softmax.

    Attributes:
        actions (List[str]): 可选动作列表 / List of available actions
        _matrix (ndarray): 性能矩阵，形状 (n_actions, n_layers) / Performance matrix
        _pointer (int): 当前写入位置指针 / Current write pointer

    Example:
        >>> adapter = ActionAdapter(['A', 'B', 'C'], layers=5)
        >>> action = adapter.decide()
        >>> adapter.update(action, contribution_ratio=0.15)
    """

    # 类常量 / Class Constants
    DEFAULT_LAYERS = 5

    def __init__(self, actions: List[str], layers: int = DEFAULT_LAYERS) -> None:
        """
        初始化动作选择器 / Initialize action selector.

        Args:
            actions: 可选动作列表 / List of available actions
            layers: 历史记录层数 / Number of historical record layers

        Raises:
            ValueError: 当 actions 为空或 layers 小于 1 时 / When actions is empty or layers < 1
        """
        if not actions:
            raise ValueError("动作列表不能为空 / Actions list cannot be empty")
        if layers < 1:
            raise ValueError(f"层数必须 >= 1 / Layers must be >= 1, got {layers}")

        self.actions = actions
        self._matrix = np.ones((len(actions), layers), dtype=np.float64)
        self._pointer = 0

        logger.debug(f"初始化动作选择器 / Initialized adapter: actions={actions}, layers={layers}")

    def decide(self) -> str:
        """
        根据当前性能分布选择动作 / Select action based on current performance distribution.

        使用 softmax 策略将历史性能转换为选择概率，然后随机选择。
        Uses softmax strategy to convert historical performance to selection
        probabilities, then randomly selects.

        Returns:
            选择的动作名称 / Selected action name

        Raises:
            RuntimeError: 当概率分布无效时 / When probability distribution is invalid
        """
        # 计算每个动作的累积性能 / Calculate cumulative performance for each action
        performance_scores = np.sum(self._matrix, axis=1)
        probabilities = softmax(performance_scores)

        # 轮盘赌选择 / Roulette wheel selection
        random_value = np.random.random()
        cumulative_prob = 0.0

        for idx, prob in enumerate(probabilities):
            cumulative_prob += prob
            if random_value <= cumulative_prob:
                selected_action = self.actions[idx]
                logger.debug(f"选择动作 / Selected action: {selected_action} (prob={prob:.3f})")
                return selected_action

        # 兜底返回最后一个动作 / Fallback to last action
        logger.warning(f"概率选择失败，使用默认动作 / Prob selection failed, using default: {self.actions[-1]}")
        return self.actions[-1]

    def update(self, choice: str, value: Union[float, int]) -> None:
        """
        更新选中动作的性能记录 / Update performance record for selected action.

        将性能值写入当前指针位置，然后移动指针。
        Writes performance value to current pointer position, then advances pointer.

        Args:
            choice: 选择的动作 / Selected action
            value: 性能值（通常为改进比例）/ Performance value (usually improvement ratio)

        Raises:
            ValueError: 当 choice 不在 actions 中时 / When choice is not in actions
        """
        if choice not in self.actions:
            raise ValueError(f"未知动作 / Unknown action: {choice}. Available: {self.actions}")

        # 创建动作掩码 / Create action mask
        action_mask = np.array([action == choice for action in self.actions], dtype=bool)

        # 重置当前层 / Reset current layer
        self._matrix[:, self._pointer] = 1.0
        # 更新选中动作的性能值 / Update performance value for selected action
        self._matrix[action_mask, self._pointer] = float(value)

        # 移动指针 / Advance pointer
        self._pointer = (self._pointer + 1) % self._matrix.shape[1]

        logger.debug(
            f"更新动作性能 / Updated action: {choice}, value={value:.3f}, "
            f"layer={self._pointer - 1}"
        )
