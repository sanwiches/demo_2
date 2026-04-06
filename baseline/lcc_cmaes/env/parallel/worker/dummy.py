"""
虚拟工作器模块 / Dummy Worker Module.

此模块提供了用于顺序向量环境的虚拟工作器实现。
This module provides a dummy worker implementation for sequential vector environments.

虚拟工作器在单个进程中顺序执行环境操作，适用于调试和简单场景。
The dummy worker executes environment operations sequentially in a single process,
suitable for debugging and simple scenarios.

主要组件 / Main Components:
    - DummyEnvWorker: 用于顺序执行的环境工作器 / Environment worker for sequential execution

使用示例 / Usage Example:
    >>> from env.parallel.worker.dummy import DummyEnvWorker
    >>> import gym
    >>> worker = DummyEnvWorker(env_fn=lambda: gym.make("CartPole-v1"))
    >>> obs = worker.reset()
    >>> obs, reward, done, info = worker.step(action)
    >>> worker.close()
"""

from typing import Any, Callable, List, Optional

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from env.parallel.worker.base import EnvWorker


class DummyEnvWorker(EnvWorker):
    """
    虚拟环境工作器 / Dummy Environment Worker.

    用于顺序向量环境的虚拟工作器，在主进程中直接执行环境操作。
    A dummy worker for sequential vector environments that executes
    environment operations directly in the main process.

    此工作器不使用多进程或分布式计算，适用于调试和单线程场景。
    This worker does not use multiprocessing or distributed computing,
    suitable for debugging and single-threaded scenarios.

    Attributes:
        env (gym.Env): 托管的环境实例 / Hosted environment instance
        result (Union[Tuple, ndarray]): 存储操作结果 / Stored operation result

    Example:
        >>> from env.parallel.worker.dummy import DummyEnvWorker
        >>> import gym
        >>> worker = DummyEnvWorker(env_fn=lambda: gym.make("CartPole-v1"))
        >>> observation = worker.reset()
        >>> obs, reward, done, info = worker.step(worker.action_space.sample())
        >>> worker.close()
    """

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        """
        初始化虚拟工作器 / Initialize dummy worker.

        Args:
            env_fn: 返回 gym 环境的可调用对象 / Callable that returns a gym environment
        """
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        """
        获取环境属性 / Get environment attribute.

        Args:
            key: 属性名称 / Attribute name

        Returns:
            属性值 / Attribute value
        """
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        """
        设置环境属性 / Set environment attribute.

        Args:
            key: 属性名称 / Attribute name
            value: 属性值 / Attribute value
        """
        setattr(self.env, key, value)

    def reset(self) -> npt.NDArray[np.float64]:
        """
        重置环境并返回初始观测 / Reset environment and return initial observation.

        Returns:
            初始观测 / Initial observation
        """
        obs, _ = self.env.reset()  # gymnasium reset 返回 (obs, info)
        return obs

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        """
        等待工作器完成 / Wait for workers to complete.

        顺序工作器总是立即可用，因此直接返回所有工作器。
        Sequential EnvWorker objects are always ready, so return all workers directly.

        Args:
            workers: 工作器列表 / List of workers
            wait_num: 等待数量（此参数被忽略）/ Number to wait (ignored)
            timeout: 超时时间（此参数被忽略）/ Timeout (ignored)

        Returns:
            所有工作器列表 / List of all workers
        """
        # 顺序工作器总是立即可用 / Sequential workers are always ready
        return workers

    def send(self, action: Optional[npt.NDArray[np.float64]]) -> None:
        """
        发送动作到环境 / Send action to environment.

        如果动作为 None，执行重置操作；否则执行步进操作。
        If action is None, performs reset; otherwise performs step.

        Args:
            action: 动作数组，None 表示重置 / Action array, None means reset
        """
        if action is None:
            obs, _ = self.env.reset()  # gymnasium reset 返回 (obs, info)
            self.result = obs
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)  # gymnasium step 返回 5 元组
            done = terminated or truncated
            self.result = (obs, reward, done, info)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        设置环境随机种子 / Set environment random seed.

        Args:
            seed: 随机种子值 / Random seed value

        Returns:
            种子列表 / List of seeds
        """
        super().seed(seed)
        return self.env.seed(seed)  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        """
        渲染环境 / Render the environment.

        Args:
            **kwargs: 渲染参数 / Rendering arguments

        Returns:
            渲染结果 / Rendering result
        """
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        """关闭环境 / Close the environment."""
        self.env.close()
