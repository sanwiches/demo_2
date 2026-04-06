"""
Cloudpickle 包装器模块 / Cloudpickle Wrapper Module.

此模块提供了一个用于序列化 Python 对象的 cloudpickle 包装器。
This module provides a cloudpickle wrapper for serializing Python objects.

cloudpickle 能够序列化标准 pickle 无法处理的复杂对象（如 lambda 函数、
嵌套函数等），使其能够在进程间传递。
cloudpickle can serialize complex objects that standard pickle cannot handle
(such as lambda functions, nested functions, etc.), enabling inter-process transfer.

主要组件 / Main Components:
    - CloudpickleWrapper: Cloudpickle 序列化包装器 / Cloudpickle serialization wrapper

使用示例 / Usage Example:
    >>> from env.parallel.worker.CloudpickleWrapper import CloudpickleWrapper
    >>> # 包装 lambda 函数以便在进程间传递 / Wrap lambda for inter-process transfer
    >>> env_fn = lambda: gym.make("CartPole-v1")
    >>> wrapped = CloudpickleWrapper(env_fn)
    >>> pickled_data = wrapped.__getstate__()
    >>> unwrapped = CloudpickleWrapper(None)
    >>> unwrapped.__setstate__(pickled_data)
"""

import pickle
from typing import Any

import cloudpickle


class CloudpickleWrapper:
    """
    Cloudpickle 序列化包装器 / Cloudpickle Serialization Wrapper.

    用于子进程向量环境的 cloudpickle 包装器。
    A cloudpickle wrapper used in SubprocVectorEnv.

    此类使用 cloudpickle 序列化数据，支持序列化 lambda 函数、嵌套函数
    等标准 pickle 无法处理的对象。
    This class uses cloudpickle to serialize data, supporting serialization of
    lambda functions, nested functions, and other objects that standard pickle
    cannot handle.

    Attributes:
        data (Any): 包装的数据 / Wrapped data

    Example:
        >>> from env.parallel.worker.CloudpickleWrapper import CloudpickleWrapper
        >>> env_fn = lambda: gym.make("CartPole-v1")
        >>> wrapper = CloudpickleWrapper(env_fn)
        >>> # 可以跨进程传递 / Can be transferred across processes
        >>> import pickle
        >>> serialized = pickle.dumps(wrapper)
        >>> deserialized = pickle.loads(serialized)
        >>> env = deserialized.data()  # Reconstruct the environment
    """

    def __init__(self, data: Any) -> None:
        """
        初始化 Cloudpickle 包装器 / Initialize Cloudpickle wrapper.

        Args:
            data: 要包装的数据 / Data to wrap
        """
        self.data = data

    def __getstate__(self) -> bytes:
        """
        获取对象状态用于序列化 / Get object state for serialization.

        使用 cloudpickle 序列化包装的数据。
        Uses cloudpickle to serialize the wrapped data.

        Returns:
            序列化后的字节 / Serialized bytes
        """
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        """
        从序列化数据恢复对象状态 / Restore object state from serialized data.

        使用 cloudpickle 反序列化数据并设置到 data 属性。
        Uses cloudpickle to deserialize data and set to data attribute.

        Args:
            data: 序列化的字节数据 / Serialized byte data
        """
        self.data = cloudpickle.loads(data)
