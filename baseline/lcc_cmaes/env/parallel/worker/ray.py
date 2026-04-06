"""
Ray 分布式工作器模块 / Ray Distributed Worker Module.

此模块提供了基于 Ray 框架的分布式环境工作器实现。
This module provides a distributed environment worker implementation based on Ray framework.

Ray 是一个分布式执行框架，可以轻松扩展到多台机器。
Ray is a distributed execution framework that can easily scale to multiple machines.

注意 / Note:
    使用此模块需要先安装 ray: pip install ray
    To use this module, install ray first: pip install ray

主要组件 / Main Components:
    - RayEnvWorker: 基于 Ray 的环境工作器 / Ray-based environment worker
    - _SetAttrWrapper: 提供 set_env_attr 和 get_env_attr 方法的包装器 /
                       Wrapper providing set_env_attr and get_env_attr methods

使用示例 / Usage Example:
    >>> import ray
    >>> from env.parallel.worker.ray import RayEnvWorker
    >>> ray.init()
    >>> worker = RayEnvWorker(env_fn=lambda: gym.make("CartPole-v1"))
    >>> obs = worker.reset()
    >>> obs, reward, done, info = worker.step(action)
    >>> worker.close()
    >>> ray.shutdown()
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from env.parallel.worker.base import EnvWorker

try:
    import ray
except ImportError:
    ray = None  # type: ignore


class _SetAttrWrapper(gym.Wrapper):
    """
    属性设置包装器 / Attribute Setting Wrapper.

    为环境提供 set_env_attr 和 get_env_attr 方法的 Gym 包装器。
    A Gym wrapper that provides set_env_attr and get_env_attr methods to environments.

    此包装器使通过 Ray 远程调用的环境能够动态设置和获取属性。
    This wrapper allows environments called remotely via Ray to dynamically
    set and get attributes.

    Example:
        >>> import gym
        >>> from env.parallel.worker.ray import _SetAttrWrapper
        >>> env = gym.make("CartPole-v1")
        >>> wrapped = _SetAttrWrapper(env)
        >>> wrapped.set_env_attr("custom_attr", 42)
        >>> value = wrapped.get_env_attr("custom_attr")  # Returns 42
    """

    def set_env_attr(self, key: str, value: Any) -> None:
        """
        设置环境属性 / Set environment attribute.

        Args:
            key: 属性名称 / Attribute name
            value: 属性值 / Attribute value
        """
        setattr(self.env, key, value)

    def get_env_attr(self, key: str) -> Any:
        """
        获取环境属性 / Get environment attribute.

        Args:
            key: 属性名称 / Attribute name

        Returns:
            属性值 / Attribute value
        """
        return getattr(self.env, key)


class RayEnvWorker(EnvWorker):
    """
    Ray 环境工作器 / Ray Environment Worker.

    用于 RayVectorEnv 的分布式工作器。
    A distributed worker used in RayVectorEnv.

    此工作器使用 Ray 框架实现环境的远程执行，支持分布式并行计算。
    This worker uses the Ray framework for remote environment execution,
    supporting distributed parallel computing.

    属性 / Attributes:
        env (ray.ActorHandle): 远程环境 Actor 句柄 / Remote environment actor handle
        result (ray.ObjectRef): 存储 Ray 对象引用 / Stores Ray object reference

    Example:
        >>> import ray
        >>> from env.parallel.worker.ray import RayEnvWorker
        >>> ray.init()
        >>> worker = RayEnvWorker(env_fn=lambda: gym.make("CartPole-v1"))
        >>> observation = worker.reset()
        >>> obs, reward, done, info = worker.step(worker.action_space.sample())
        >>> worker.close()
        >>> ray.shutdown()
    """

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        """
        初始化 Ray 工作器 / Initialize Ray worker.

        Args:
            env_fn: 返回 gym 环境的可调用对象 / Callable that returns a gym environment

        Raises:
            ImportError: 如果 Ray 未安装 / If Ray is not installed
        """
        if ray is None:
            raise ImportError(
                "Ray is not installed. Please install it via: pip install ray\n"
                "Ray 未安装。请通过以下命令安装: pip install ray"
            )
        # 使用 num_cpus=0 避免资源冲突 / Use num_cpus=0 to avoid resource conflicts
        self.env = ray.remote(_SetAttrWrapper).options(num_cpus=0).remote(env_fn())
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        """
        获取远程环境属性 / Get remote environment attribute.

        Args:
            key: 属性名称 / Attribute name

        Returns:
            属性值 / Attribute value
        """
        return ray.get(self.env.get_env_attr.remote(key))

    def set_env_attr(self, key: str, value: Any) -> None:
        """
        设置远程环境属性 / Set remote environment attribute.

        Args:
            key: 属性名称 / Attribute name
            value: 属性值 / Attribute value
        """
        ray.get(self.env.set_env_attr.remote(key, value))

    def reset(self) -> npt.NDArray[np.float64]:
        """
        重置远程环境并返回初始观测 / Reset remote environment and return initial observation.

        Returns:
            初始观测 / Initial observation
        """
        obs, _ = ray.get(self.env.reset.remote())  # gymnasium reset 返回 (obs, info)
        return obs

    @staticmethod
    def wait(  # type: ignore
        workers: List["RayEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None
    ) -> List["RayEnvWorker"]:
        """
        等待指定数量的 Ray 工作器完成 / Wait for a specified number of Ray workers to complete.

        Args:
            workers: Ray 工作器列表 / List of Ray workers
            wait_num: 等待完成的工作器数量 / Number of workers to wait for
            timeout: 超时时间（秒）/ Timeout in seconds

        Returns:
            已完成的工作器列表 / List of completed workers
        """
        results = [x.result for x in workers]
        ready_results, _ = ray.wait(results, num_returns=wait_num, timeout=timeout)
        return [workers[results.index(result)] for result in ready_results]

    def send(self, action: Optional[npt.NDArray[np.float64]]) -> None:
        """
        发送动作到远程环境 / Send action to remote environment.

        动作通过 Ray 远程调用异步发送。
        The action is sent asynchronously via Ray remote call.

        Args:
            action: 动作数组，None 表示重置 / Action array, None means reset
        """
        # self.result 实际上是一个句柄 / self.action is actually a handle
        if action is None:
            self.result = self.env.reset.remote()
        else:
            self.result = self.env.step.remote(action)

    def recv(
        self
    ) -> Union[
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
              npt.NDArray[np.bool_], dict],
        npt.NDArray[np.float64]
    ]:
        """
        从远程环境接收结果 / Receive result from remote environment.

        Returns:
            观测值或 (观测, 奖励, 完成, 信息) 元组 / Observation or tuple
        """
        result = ray.get(self.result)
        if isinstance(result, tuple) and len(result) == 5:
            # gymnasium step 返回 (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        elif isinstance(result, tuple) and len(result) == 2:
            # gymnasium reset 返回 (obs, info)
            return result[0]  # 只返回 obs
        return result

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        设置远程环境随机种子 / Set remote environment random seed.

        Args:
            seed: 随机种子值 / Random seed value

        Returns:
            种子列表 / List of seeds
        """
        super().seed(seed)
        return ray.get(self.env.seed.remote(seed))

    def render(self, **kwargs: Any) -> Any:
        """
        渲染远程环境 / Render remote environment.

        Args:
            **kwargs: 渲染参数 / Rendering arguments

        Returns:
            渲染结果 / Rendering result
        """
        return ray.get(self.env.render.remote(**kwargs))

    def close_env(self) -> None:
        """关闭远程环境 / Close remote environment."""
        ray.get(self.env.close.remote())
