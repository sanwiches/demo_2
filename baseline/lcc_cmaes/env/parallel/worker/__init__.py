"""
环境工作器包 / Environment Worker Package.

此包提供了多种环境工作器实现，支持不同的并行和分布式策略。
This package provides various environment worker implementations supporting
different parallel and distributed strategies.

工作器类型 / Worker Types:
    - EnvWorker: 所有工作器的抽象基类 / Abstract base class for all workers
    - DummyEnvWorker: 顺序执行的单进程工作器 / Sequential single-process worker
    - SubprocEnvWorker: 基于多进程的并行工作器 / Multiprocess-based parallel worker
    - RayEnvWorker: 基于 Ray 的分布式工作器 / Ray-based distributed worker

选择指南 / Selection Guide:
    - DummyEnvWorker: 调试、简单场景、无并行需求 / Debugging, simple scenarios, no parallelism
    - SubprocEnvWorker: 单机多核并行 / Multi-core parallelism on single machine
    - RayEnvWorker: 跨机器分布式 / Distributed across multiple machines

使用示例 / Usage Example:
    >>> from env.parallel.worker import DummyEnvWorker, SubprocEnvWorker
    >>> import gym
    >>>
    >>> # 单进程工作器 / Single-process worker
    >>> worker = DummyEnvWorker(env_fn=lambda: gym.make("CartPole-v1"))
    >>>
    >>> # 多进程工作器 / Multiprocess worker
    >>> worker = SubprocEnvWorker(
    ...     env_fn=lambda: gym.make("CartPole-v1"),
    ...     share_memory=True
    ... )
"""

from env.parallel.worker.base import EnvWorker
from env.parallel.worker.dummy import DummyEnvWorker
from env.parallel.worker.ray import RayEnvWorker
from env.parallel.worker.subproc import SubprocEnvWorker

__all__ = [
    "EnvWorker",
    "DummyEnvWorker",
    "SubprocEnvWorker",
    "RayEnvWorker",
]
