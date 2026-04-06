"""
Agent Package / 智能体包

This package contains inference agents for LCC-CMAES optimization.
本包包含 LCC-CMAES 优化的推理智能体。

Modules / 模块:
    - inference: Inference agent for trained Actor network / 训练好的 Actor 网络的推理智能体
    - network: Neural network architectures (Actor) / 神经网络架构 (Actor)

Usage / 使用方法:
    >>> from env.agent.inference import InferenceAgent
    >>> from env.agent.network.actor_network import Actor
"""

__all__ = ['InferenceAgent']


# Export main inference agent / 导出主要推理智能体
from env.agent.inference import InferenceAgent
