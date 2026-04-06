"""
子进程工作器模块 / Subprocess Worker Module.

此模块提供了基于多进程的环境工作器实现，支持共享内存优化。
This module provides a multiprocessing-based environment worker implementation
with shared memory optimization.

子进程工作器通过管道或共享内存与主进程通信，实现真正的并行环境执行。
The subprocess worker communicates with the main process via pipes or shared memory,
achieving true parallel environment execution.

主要组件 / Main Components:
    - SubprocEnvWorker: 子进程环境工作器 / Subprocess environment worker
    - ShArray: 共享内存数组包装器 / Shared memory array wrapper
    - _worker: 子进程工作函数 / Subprocess worker function
    - _setup_buf: 设置共享内存缓冲区 / Setup shared memory buffer

使用示例 / Usage Example:
    >>> from env.parallel.worker.subproc import SubprocEnvWorker
    >>> import gymnasium as gym
    >>> worker = SubprocEnvWorker(
    ...     env_fn=lambda: gym.make("CartPole-v1"),
    ...     share_memory=False
    ... )
    >>> obs = worker.reset()
    >>> obs, reward, done, info = worker.step(action)
    >>> worker.close()
"""

import ctypes
import logging
import time
from collections import OrderedDict
from multiprocessing import Array, Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from env.parallel.worker.CloudpickleWrapper import CloudpickleWrapper
from env.parallel.worker.base import EnvWorker

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# NumPy 到 ctypes 类型映射 / NumPy to ctypes Type Mapping
# ==========================================

_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class ShArray:
    """
    共享内存数组包装器 / Shared Memory Array Wrapper.

    multiprocessing.Array 的包装器，支持 NumPy 数组操作。
    Wrapper of multiprocessing.Array with NumPy array operations support.

    此类提供了一种高效的方式在进程间共享 NumPy 数组数据。
    This class provides an efficient way to share NumPy array data between processes.

    Attributes:
        arr (multiprocessing.Array): 共享内存数组对象 / Shared memory array object
        dtype (np.generic): NumPy 数据类型 / NumPy data type
        shape (Tuple[int]): 数组形状 / Array shape

    Example:
        >>> from env.parallel.worker.subproc import ShArray
        >>> import numpy as np
        >>> shared_arr = ShArray(np.float64, (10, 10))
        >>> data = np.random.randn(10, 10)
        >>> shared_arr.save(data)
        >>> retrieved = shared_arr.get()
        >>> np.array_equal(data, retrieved)
        True
    """

    def __init__(self, dtype: np.generic, shape: Tuple[int, ...]) -> None:
        """
        初始化共享数组 / Initialize shared array.

        Args:
            dtype: NumPy 数据类型 / NumPy data type
            shape: 数组形状 / Array shape
        """
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: npt.NDArray[Any]) -> None:
        """
        保存 NumPy 数组到共享内存 / Save NumPy array to shared memory.

        Args:
            ndarray: 要保存的 NumPy 数组 / NumPy array to save

        Raises:
            AssertionError: 如果输入不是 NumPy 数组 / If input is not a NumPy array
        """
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(self.shape)
        np.copyto(dst_np, ndarray)

    def get(self) -> npt.NDArray[Any]:
        """
        从共享内存获取 NumPy 数组 / Get NumPy array from shared memory.

        Returns:
            共享内存中的 NumPy 数组 / NumPy array from shared memory
        """
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    """
    根据观测空间设置共享内存缓冲区 / Setup shared memory buffer based on observation space.

    Args:
        space: Gym 观测空间 / Gym observation space

    Returns:
        共享缓冲区结构 / Shared buffer structure (dict, tuple, or ShArray)
    """
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    """
    子进程工作函数 / Subprocess worker function.

    在子进程中运行，接收命令并执行环境操作。
    Runs in a subprocess, receives commands and executes environment operations.

    支持的命令 / Supported commands:
        - "step": 步进环境 (data=None 时重置) / Step environment (reset when data=None)
        - "close": 关闭环境 / Close environment
        - "render": 渲染环境 / Render environment
        - "seed": 设置随机种子 / Set random seed
        - "getattr": 获取环境属性 / Get environment attribute
        - "setattr": 设置环境属性 / Set environment attribute

    Args:
        parent: 父进程管道连接 / Parent pipe connection
        p: 子进程管道连接 / Child pipe connection
        env_fn_wrapper: 环境函数包装器 / Environment function wrapper
        obs_bufs: 观测缓冲区（可选）/ Observation buffers (optional)
    """

    def _encode_obs(
        obs: Union[dict, tuple, npt.NDArray[Any]],
        buffer: Union[dict, tuple, ShArray]
    ) -> None:
        """将观测编码到共享缓冲区 / Encode observation to shared buffer."""
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # 管道已关闭 / The pipe has been closed
                p.close()
                break
            if cmd == "step":
                if data is None:  # 重置 / Reset
                    obs, info = env.reset()
                else:
                    obs, reward, terminated, truncated, info = env.step(data)
                    done = terminated or truncated  # gymnasium 返回 terminated 和 truncated，合并为 done
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if data is None:
                    p.send(obs)
                else:
                    p.send((obs, reward, done, info))
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                p.send(env.seed(data) if hasattr(env, "seed") else None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env, data["key"], data["value"])
            else:
                p.close()
                raise NotImplementedError(f"未知命令 / Unknown command: {cmd}")
    except KeyboardInterrupt:
        p.close()


class SubprocEnvWorker(EnvWorker):
    """
    子进程环境工作器 / Subprocess Environment Worker.

    用于 SubprocVectorEnv 和 ShmemVectorEnv 的子进程工作器。
    Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv.

    此工作器在独立进程中运行环境，通过管道与主进程通信。
    可选使用共享内存优化数据传输。
    This worker runs an environment in a separate process and communicates
    with the main process via pipes. Optionally uses shared memory for
    optimized data transfer.

    Attributes:
        parent_remote (connection.Connection): 父进程管道 / Parent pipe
        share_memory (bool): 是否使用共享内存 / Whether to use shared memory
        buffer (Optional[Union[dict, tuple, ShArray]]): 共享缓冲区 / Shared buffer
        process (multiprocessing.Process): 子进程对象 / Subprocess object
        is_reset (bool): 是否已重置 / Whether reset

    Example:
        >>> from env.parallel.worker.subproc import SubprocEnvWorker
        >>> import gym
        >>> worker = SubprocEnvWorker(
        ...     env_fn=lambda: gym.make("CartPole-v1"),
        ...     share_memory=True
        ... )
        >>> obs = worker.reset()
        >>> obs, reward, done, info = worker.step(action)
        >>> worker.close()
    """

    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        """
        初始化子进程工作器 / Initialize subprocess worker.

        Args:
            env_fn: 返回 gym 环境的可调用对象 / Callable that returns a gym environment
            share_memory: 是否使用共享内存优化 / Whether to use shared memory optimization
        """
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            # 创建临时环境获取观测空间 / Create temp env to get observation space
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        self.is_reset = False
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        """
        获取环境属性 / Get environment attribute.

        Args:
            key: 属性名称 / Attribute name

        Returns:
            属性值 / Attribute value
        """
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        """
        设置环境属性 / Set environment attribute.

        Args:
            key: 属性名称 / Attribute name
            value: 属性值 / Attribute value
        """
        self.parent_remote.send(["setattr", {"key": key, "value": value}])

    def _decode_obs(self) -> Union[dict, tuple, npt.NDArray[Any]]:
        """
        从共享缓冲区解码观测 / Decode observation from shared buffer.

        Returns:
            解码后的观测 / Decoded observation
        """

        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]]
        ) -> Union[dict, tuple, npt.NDArray[Any]]:
            """递归解码缓冲区 / Recursively decode buffer."""
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError(f"不支持的缓冲区类型 / Unsupported buffer type: {type(buffer)}")

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        """
        等待指定数量的子进程工作器完成 / Wait for a specified number of subprocess workers to complete.

        Args:
            workers: 子进程工作器列表 / List of subprocess workers
            wait_num: 等待完成的工作器数量 / Number of workers to wait for
            timeout: 超时时间（秒）/ Timeout in seconds

        Returns:
            已完成的工作器列表 / List of completed workers
        """
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait 在列表为空时会挂起 / connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send(self, action: Optional[npt.NDArray[np.float64]]) -> None:
        """
        发送动作到子进程环境 / Send action to subprocess environment.

        Args:
            action: 动作数组，None 表示重置 / Action array, None means reset
        """
        self.parent_remote.send(["step", action])

    def recv(
        self
    ) -> Union[
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
              npt.NDArray[np.bool_], dict],
        npt.NDArray[np.float64]
    ]:
        """
        从子进程环境接收结果 / Receive result from subprocess environment.

        Returns:
            观测值或 (观测, 奖励, 完成, 信息) 元组 / Observation or tuple
        """
        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            obs, rew, done, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, rew, done, info
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        """
        设置环境随机种子 / Set environment random seed.

        Args:
            seed: 随机种子值 / Random seed value

        Returns:
            种子列表或 None / List of seeds or None
        """
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        return self.parent_remote.recv()

    def render(self, **kwargs: Any) -> Any:
        """
        渲染环境 / Render the environment.

        Args:
            **kwargs: 渲染参数 / Rendering arguments

        Returns:
            渲染结果 / Rendering result
        """
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        """关闭子进程环境 / Close subprocess environment."""
        try:
            self.parent_remote.send(["close", None])
            # mp 可能已被删除，可能引发 AttributeError / mp may be deleted, may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # 确保子进程已终止 / Ensure the subprocess is terminated
        self.process.terminate()
