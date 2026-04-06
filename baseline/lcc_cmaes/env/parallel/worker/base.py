"""
环境工作器基类模块 / Environment Worker Base Module.

此模块定义了环境工作器的抽象基类，提供了所有工作器共享的接口和功能。
This module defines the abstract base class for environment workers, providing
shared interfaces and functionality for all worker implementations.

主要组件 / Main Components:
    - EnvWorker: 环境工作器抽象基类 / Abstract base class for environment workers

使用示例 / Usage Example:
    >>> from env.parallel.worker.base import EnvWorker
    >>> # 子类需要实现所有抽象方法 / Subclasses must implement all abstract methods
    >>> class MyWorker(EnvWorker):
    ...     def get_env_attr(self, key: str) -> Any:
    ...         pass
    ...     def set_env_attr(self, key: str, value: Any) -> None:
    ...         pass
    ...     def render(self, **kwargs: Any) -> Any:
    ...         pass
    ...     def close_env(self) -> None:
    ...         pass
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt


class EnvWorker(ABC):
    """
    环境工作器抽象基类 / Abstract Base Class for Environment Workers.

    此类定义了所有环境工作器必须实现的接口，支持同步和异步环境交互。
    This class defines the interface that all environment workers must implement,
    supporting both synchronous and asynchronous environment interactions.

    Attributes:
        _env_fn (Callable): 环境构造函数 / Environment constructor function
        is_closed (bool): 环境是否已关闭 / Whether the environment is closed
        result (Union[Tuple, ndarray]): 存储操作结果 / Stored operation result
        action_space (gym.Space): 环境的动作空间 / Environment action space
        is_reset (bool): 是否已重置 / Whether the environment has been reset

    Example:
        >>> worker = MyWorker(env_fn=lambda: gym.make("CartPole-v1"))
        >>> observation = worker.reset()
        >>> obs, reward, done, info = worker.step(action)
        >>> worker.close()
    """

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        """
        初始化环境工作器 / Initialize environment worker.

        Args:
            env_fn: 返回 gym 环境的可调用对象 / Callable that returns a gym environment
        """
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Union[
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                  npt.NDArray[np.bool_], dict],
            npt.NDArray[np.float64]
        ] = np.array([])  # type: ignore
        self.action_space = self.get_env_attr("action_space")  # noqa: B009
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        """
        获取环境属性 / Get environment attribute.

        Args:
            key: 属性名称 / Attribute name

        Returns:
            属性值 / Attribute value
        """
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        """
        设置环境属性 / Set environment attribute.

        Args:
            key: 属性名称 / Attribute name
            value: 属性值 / Attribute value
        """
        pass

    def send(self, action: Optional[npt.NDArray[np.float64]]) -> None:
        """
        向底层工作器发送动作信号 / Send action signal to low-level worker.

        当动作为 None 时，表示发送"重置"信号；否则表示"步进"信号。
        When action is None, it sends a "reset" signal; otherwise it sends a "step" signal.
        配对的"recv"函数返回值由不同的信号类型决定。
        The paired return value from "recv" is determined by the signal type.

        Args:
            action: 动作数组，None 表示重置 / Action array, None means reset
        """
        if hasattr(self, "send_action"):
            warnings.warn(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker.\n"
                "send_action 即将弃用。请为您的 EnvWorker 使用 send 和 recv。"
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)  # type: ignore

    def recv(
        self
    ) -> Union[
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
              npt.NDArray[np.bool_], dict],
        npt.NDArray[np.float64]
    ]:
        """
        从底层工作器接收结果 / Receive result from low-level worker.

        如果最后一次"send"发送的是 None 动作，仅返回单个观测值；
        否则返回元组 (观测, 奖励, 完成, 信息)。
        If the last "send" was a NULL action, returns a single observation;
        otherwise returns a tuple of (obs, reward, done, info).

        Returns:
            观测值或 (观测, 奖励, 完成, 信息) 元组 / Observation or tuple
        """
        if hasattr(self, "get_result"):
            warnings.warn(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker.\n"
                "get_result 即将弃用。请为您的 EnvWorker 使用 send 和 recv。"
            )
            if not self.is_reset:
                self.result = self.get_result()  # type: ignore
        return self.result

    def reset(self) -> npt.NDArray[np.float64]:
        """
        重置环境并返回初始观测 / Reset environment and return initial observation.

        Returns:
            初始观测 / Initial observation
        """
        self.send(None)
        return self.recv()  # type: ignore

    def step(
        self, action: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
               npt.NDArray[np.bool_], dict]:
        """
        执行环境动态的一个时间步 / Perform one timestep of the environment's dynamics.

        "send" 和 "recv" 在同步模拟中是耦合的，用户只需调用 "step" 函数。
        但它们可以在异步模拟中单独调用，即先调用 "send"，稍后调用 "recv"。
        "send" and "recv" are coupled in sync simulation, so users only call "step".
        They can be called separately in async simulation.

        Args:
            action: 要执行的动作 / Action to perform

        Returns:
            包含 (观测, 奖励, 完成, 信息) 的元组 / Tuple of (obs, reward, done, info)
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: List["EnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """
        等待指定数量的工作器完成 / Wait for a specified number of workers to complete.

        Args:
            workers: 工作器列表 / List of workers
            wait_num: 等待完成的工作器数量 / Number of workers to wait for
            timeout: 超时时间（秒）/ Timeout in seconds

        Returns:
            已完成的工作器列表 / List of completed workers

        Raises:
            NotImplementedError: 此方法应在子类中实现 / This method should be implemented in subclasses
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        """
        设置环境随机种子 / Set environment random seed.

        Args:
            seed: 随机种子值 / Random seed value

        Returns:
            种子列表或 None / List of seeds or None
        """
        return self.action_space.seed(seed)  # issue 299

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """
        渲染环境 / Render the environment.

        Args:
            **kwargs: 渲染参数 / Rendering arguments

        Returns:
            渲染结果 / Rendering result
        """
        pass

    @abstractmethod
    def close_env(self) -> None:
        """关闭环境 / Close the environment."""
        pass

    def close(self) -> None:
        """
        关闭工作器并清理资源 / Close the worker and clean up resources.

        如果工作器已关闭，则直接返回。
        If the worker is already closed, returns immediately.
        """
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()
