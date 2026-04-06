# ==========================================
# JAX 设备配置 / JAX Device Configuration
# ==========================================
# 必须在任何 JAX 导入之前配置 / Must be configured before any JAX imports
import os

# 导入 JAX CUDA 插件（如果可用）/ Import JAX CUDA plugin (if available)
# 必须在 JAX 之前导入 / Must be imported before JAX
try:
    import jax_cuda12_plugin
except ImportError:
    pass  # CUDA 插件不可用 / CUDA plugin not available

# 导入 JAX / Import JAX
try:
    import jax
except ImportError:
    raise ImportError(
        "JAX is required but not installed. "
        "Please install it to use this benchmark.\n\n"
        "JAX 是必需依赖但未安装。请安装它以使用此基准测试。"
    )

# 运行时设备检测函数 / Runtime device detection function
def _detect_device() -> str:
    """检测可用的 JAX 设备 / Detect available JAX device"""
    try:
        if len(jax.devices('cuda')) > 0:
            return "cuda"
        else:
            return "cpu"
    except Exception:
        return "cpu"

# 禁用内存预分配以避免内存问题 / Disable pre-allocation to avoid memory issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

"""
MuJoCo Benchmark for LSGOplatform / MuJoCo 基准测试库

这是一个基于 mujoco_playground 的神经进化问题 benchmark 实现。
This is a neuroevolution benchmark implementation based on mujoco_playground.

【设计原则 / Design Principles】
- 与 mujoco_playground 的设计保持一致 / Consistent with mujoco_playground design
- 使用简单的 2 层 MLP 策略网络 / Use simple 2-layer MLP policy network
- 固定的评估参数 / Fixed evaluation parameters
- 支持 JAX 加速 / Support JAX acceleration
- 包含所有 54 个可用环境 / Include all 54 available environments

【评估协议 / Evaluation Protocol】
- 规范的 JAX random key 管理 / Standard JAX random key management
- 明确的 max_episode_length / Explicit max_episode_length
- 显式的 reward aggregation / Explicit reward aggregation
- 清晰的评估协议分层 / Clear evaluation protocol layering

【平台接口兼容 / Platform Interface Compatibility】
- 与 LSGOplatform 现有优化器 (MMES/CMAES) 完全兼容 / Fully compatible with LSGOplatform optimizers
- 输入 (D,) → 输出 (1,) / Input (D,) → Output (1,)
- 输入 (N, D) → 输出 (N,) / Input (N, D) → Output (N,)

【环境列表 / Environment List】
总共 54 个环境，分为以下类别：
- Classic Control (7): AcrobotSwingup, Cartpole*, PendulumSwingup
- Locomotion (11): SwimmerSwimmer6, CheetahRun, FishSwim, Hopper*, Walker*, Humanoid*
- Manipulation (15): Finger*, Reacher*, Panda*, Aloha*, Leap*, Aero*
- Humanoid/Robot (18): Apollo*, Barkour*, Berkeley*, G1*, Go1*, H1*, Op3*, Spot*, T1*
- Other (2): BallInCup, PointMass

【GPU 使用说明 / GPU Usage】
要使用 GPU 加速，请在导入 benchmark 之前按以下顺序导入 / To use GPU acceleration, import in the following order before importing benchmark:
    import jax_cuda12_plugin  # 首先导入 CUDA 插件 / First import CUDA plugin
    import jax                # 然后导入 JAX / Then import JAX
    from benchmark.mujoco.mujoco_benchmarks import Benchmark

如果 CUDA 插件未安装，benchmark 将自动回退到 CPU 模式 / If CUDA plugin is not installed, benchmark will automatically fall back to CPU mode.
"""

# ==========================================
# 第一阶段：Python 标准库导入 / Stage 1: Python Standard Library Imports
# ==========================================

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Callable, Union, Any, Protocol, List
import warnings

# ==========================================
# 第二阶段：第三方库导入 / Stage 2: Third-party Library Imports
# ==========================================

import numpy as np

# JAX 已在设备配置阶段导入 / JAX already imported during device configuration
# 重新导入用于使用 / Re-import for use
import jax
import jax.numpy as jnp
from jax import random as jrand

# 尝试导入 mujoco_playground（必需依赖） / Try importing mujoco_playground (required dependency)
try:
    from mujoco_playground import registry
except ImportError as e:
    raise ImportError(
        "mujoco_playground is required but not installed. "
        "Please install it to use this benchmark.\n\n"
        "mujoco_playground 是必需依赖但未安装。"
        "请安装它以使用此基准测试。"
    ) from e

# ==========================================
# 类型定义 / Type Definitions
# ==========================================

JAXKey = jax.Array  # PRNGKey type / PRNGKey 类型
JAXArray = jax.Array  # Generic JAX array type / 通用 JAX 数组类型

# 环境状态类型 / Environment state type
EnvState = Any  # mujoco_playground env state is Python object

ParamsDict = Dict[str, JAXArray]  # Network parameter dictionary type


class HasEnvProtocol(Protocol):
    """
    环境接口协议 / Environment Interface Protocol

    描述 mujoco_playground 环境对象应具备的接口契约。
    Describes the interface contract that mujoco_playground env objects should have.
    """
    def reset(self, key: JAXKey) -> EnvState:
        """重置环境，返回环境状态 / Reset environment, return env state"""
        ...

    def step(self, state: EnvState, action: JAXArray) -> EnvState:
        """环境步进，返回新环境状态 / Environment step, return new env state"""
        ...

    @property
    def action_size(self) -> int:
        """动作维度 / Action dimension"""
        ...


# ==========================================
# 常量 / Constants
# ==========================================

# MLP 隐藏层大小 / MLP hidden layer size
# 与 EvoX README Neuroevolution 示例保持一致（单隐藏层宽度=8）
# Aligned with EvoX README neuroevolution example (single hidden layer width=8)
HIDDEN_DIM = 8

# 权重初始化边界 / Weight initialization bounds
# 定义神经网络权重的初始化范围 / Defines initialization range for neural network weights
WEIGHT_LOWER_BOUND = -1.0
WEIGHT_UPPER_BOUND = 1.0

# 环境评估设置 / Environment evaluation settings
# 与 mujoco_playground 默认参数一致 / Consistent with mujoco_playground defaults
MAX_EPISODE_LENGTH = 100  # 每个 episode 的最大步数 / Maximum steps per episode
NUM_EPISODES = 3  # 用于评估的 episodes 数量 / Number of episodes for evaluation
RANDOM_SEED = 42  # 默认随机种子 / Default random seed

# ==========================================
# 环境配置 / Environment Configuration
# ==========================================

# 所有可用的 mujoco_playground 环境 / All available mujoco_playground environments
# 按类别分组 / Grouped by category
MUJOCO_ENVS = {
    # Classic Control (7)
    'AcrobotSwingup': 'Classic Control',
    'AcrobotSwingupSparse': 'Classic Control',
    'BallInCup': 'Classic Control',
    'CartpoleBalance': 'Classic Control',
    'CartpoleBalanceSparse': 'Classic Control',
    'CartpoleSwingup': 'Classic Control',
    'CartpoleSwingupSparse': 'Classic Control',
    'PendulumSwingup': 'Classic Control',
    'PointMass': 'Classic Control',

    # Locomotion (11)
    'SwimmerSwimmer6': 'Locomotion',
    'CheetahRun': 'Locomotion',
    'FishSwim': 'Locomotion',
    'HopperHop': 'Locomotion',
    'HopperStand': 'Locomotion',
    'WalkerRun': 'Locomotion',
    'WalkerStand': 'Locomotion',
    'WalkerWalk': 'Locomotion',
    'HumanoidStand': 'Locomotion',
    'HumanoidWalk': 'Locomotion',
    'HumanoidRun': 'Locomotion',

    # Manipulation (15)
    'FingerSpin': 'Manipulation',
    'FingerTurnEasy': 'Manipulation',
    'FingerTurnHard': 'Manipulation',
    'ReacherEasy': 'Manipulation',
    'ReacherHard': 'Manipulation',
    'PandaPickCube': 'Manipulation',
    'PandaPickCubeOrientation': 'Manipulation',
    'PandaPickCubeCartesian': 'Manipulation',
    'PandaOpenCabinet': 'Manipulation',
    'PandaRobotiqPushCube': 'Manipulation',
    'AlohaHandOver': 'Manipulation',
    'AlohaSinglePegInsertion': 'Manipulation',
    'LeapCubeReorient': 'Manipulation',
    'LeapCubeRotateZAxis': 'Manipulation',
    'AeroCubeRotateZAxis': 'Manipulation',

    # Humanoid/Robot (18)
    'ApolloJoystickFlatTerrain': 'Humanoid/Robot',
    'BarkourJoystick': 'Humanoid/Robot',
    'BerkeleyHumanoidJoystickFlatTerrain': 'Humanoid/Robot',
    'BerkeleyHumanoidJoystickRoughTerrain': 'Humanoid/Robot',
    'G1JoystickFlatTerrain': 'Humanoid/Robot',
    'G1JoystickRoughTerrain': 'Humanoid/Robot',
    'Go1JoystickFlatTerrain': 'Humanoid/Robot',
    'Go1JoystickRoughTerrain': 'Humanoid/Robot',
    'Go1Getup': 'Humanoid/Robot',
    'Go1Handstand': 'Humanoid/Robot',
    'Go1Footstand': 'Humanoid/Robot',
    'H1InplaceGaitTracking': 'Humanoid/Robot',
    'H1JoystickGaitTracking': 'Humanoid/Robot',
    'Op3Joystick': 'Humanoid/Robot',
    'SpotFlatTerrainJoystick': 'Humanoid/Robot',
    'SpotJoystickGaitTracking': 'Humanoid/Robot',
    'SpotGetup': 'Humanoid/Robot',
    'T1JoystickFlatTerrain': 'Humanoid/Robot',
    'T1JoystickRoughTerrain': 'Humanoid/Robot',
}

# 环境名称列表（按 ID 排序）/ Environment name list (sorted by ID)
ENV_NAMES = list(MUJOCO_ENVS.keys())


# ==========================================
# JAX 策略网络 / JAX Policy Network
# ==========================================

class JAXPolicyNetwork:
    """
    JAX 版本的策略网络（2层 MLP）/ JAX version of policy network (2-layer MLP)

    【网络结构 / Network Structure】
    - 输入层：obs_dim / Input layer: obs_dim
    - 隐藏层：HIDDEN_DIM (8) + Tanh / Hidden layer: HIDDEN_DIM (8) + Tanh
    - 输出层：action_dim + Tanh / Output layer: action_dim + Tanh

    【与 mujoco_playground 一致 / Consistent with mujoco_playground】
    使用简单的 2 层 MLP，不使用可变深度。
    Uses simple 2-layer MLP, no variable depth.

    【JAX 优化 / JAX Optimization】
    - 使用 jnp 替代 np / Use jnp instead of np
    - 纯函数式设计，支持 JIT 编译 / Pure functional design, supports JIT compilation
    - 前向传播可被 vmap / Forward propagation can be vmapped
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        """
        初始化网络结构 / Initialize network structure

        参数 / Args:
            obs_dim: 观测维度 / Observation dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension
            action_dim: 动作维度 / Action dimension
        """
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 参数总数：W1 + b1 + W2 + b2 / Total parameters: W1 + b1 + W2 + b2
        self.param_size = (
            obs_dim * hidden_dim + hidden_dim +
            hidden_dim * action_dim + action_dim
        )

    def unpack_params(self, flat_params: JAXArray) -> ParamsDict:
        """
        将扁平参数向量解包为网络参数字典 / Unpack flat parameter vector into network parameter dict

        参数 / Args:
            flat_params: 扁平的参数向量 / Flat parameter vector

        返回 / Returns:
            包含 'W1', 'b1', 'W2', 'b2' 的字典 / Dict containing 'W1', 'b1', 'W2', 'b2'
        """
        idx = 0

        # 第一层 / First layer: W1, b1
        W1_size = self.obs_dim * self.hidden_dim
        # 正确处理 1D 和批量输入 / Handle both 1D and batched inputs correctly
        if flat_params.ndim == 1:
            W1 = flat_params[idx:idx + W1_size].reshape((self.obs_dim, self.hidden_dim))
        else:
            batch_size = flat_params.shape[0]
            W1 = flat_params[:, idx:idx + W1_size].reshape((batch_size, self.obs_dim, self.hidden_dim))
        idx += W1_size

        b1 = flat_params[..., idx:idx + self.hidden_dim]
        idx += self.hidden_dim

        # 第二层 / Second layer: W2, b2
        W2_size = self.hidden_dim * self.action_dim
        if flat_params.ndim == 1:
            W2 = flat_params[idx:idx + W2_size].reshape((self.hidden_dim, self.action_dim))
        else:
            batch_size = flat_params.shape[0]
            W2 = flat_params[:, idx:idx + W2_size].reshape((batch_size, self.hidden_dim, self.action_dim))
        idx += W2_size

        b2 = flat_params[..., idx:idx + self.action_dim]

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def forward(self, params: ParamsDict, obs: JAXArray) -> JAXArray:
        """
        前向传播（JAX 纯函数）/ Forward propagation (JAX pure function)

        参数 / Args:
            params: 网络参数字典 / Network parameter dictionary
            obs: 观测，形状 (obs_dim,) 或 (N, obs_dim) / Observation, shape (obs_dim,) or (N, obs_dim)

        返回 / Returns:
            action: 动作，形状 (action_dim,) 或 (N, action_dim) / Action, shape (action_dim,) or (N, action_dim)
        """
        if obs.ndim == 1:
            obs = obs[None, :]  # (1, obs_dim)

        # 第一层：hidden = tanh(obs @ W1 + b1) / First layer
        hidden = jnp.tanh(jnp.dot(obs, params['W1']) + params['b1'])

        # 第二层：output = hidden @ W2 + b2 / Second layer
        output = jnp.dot(hidden, params['W2']) + params['b2']

        # 激活：action = tanh(output) / Activation
        action = jnp.tanh(output)

        # 如果输入是单个观测，压缩输出维度 / If input is single obs, squeeze output
        if action.shape[0] == 1:
            action = action[0]

        return action


# ==========================================
# 环境包装器 / Environment Wrapper
# ==========================================

class JAXEnvironmentWrapper:
    """
    JAX 环境包装器 / JAX Environment Wrapper

    【与 mujoco_playground 一致 / Consistent with mujoco_playground】
    使用 registry.load() 加载环境。
    Uses registry.load() to load environment.
    """

    def __init__(self, env_name: str) -> None:
        """
        初始化环境包装器 / Initialize environment wrapper

        参数 / Args:
            env_name: mujoco_playground 环境名称 / mujoco_playground environment name
        """
        self.env_name = env_name

        try:
            self.env: HasEnvProtocol = registry.load(env_name=env_name)
        except Exception as e:
            raise RuntimeError(
                f"无法加载环境 / Cannot load environment '{env_name}': {e}"
            ) from e

        # 使用临时 key 探测环境维度 / Use temp key to probe env dimensions
        _temp_key = jrand.PRNGKey(RANDOM_SEED)
        test_state = self.env.reset(_temp_key)

        # 提取观测值维度 / Extract observation dimension
        test_obs = _extract_obs(test_state.obs)
        self.obs_dim: int = int(test_obs.shape[-1])

        # 获取 action 维度 / Get action dimension
        if hasattr(self.env, 'action_size'):
            self.action_dim: int = int(self.env.action_size)
        elif hasattr(self.env, 'action_dim'):
            self.action_dim: int = int(self.env.action_dim)
        else:
            raise RuntimeError(
                f"环境 / Environment '{env_name}' 无法获取 action_dim / cannot get action_dim"
            )

    def reset(self, key: JAXKey) -> Tuple[JAXArray, EnvState]:
        """重置环境 / Reset environment"""
        state = self.env.reset(key)
        obs = _extract_obs(state.obs)
        return obs, state

    def step(
        self,
        state: EnvState,
        action: JAXArray
    ) -> Tuple[JAXArray, JAXArray, JAXArray, EnvState]:
        """环境步进 / Environment step"""
        jax_action = jnp.array(action)
        new_state = self.env.step(state, jax_action)

        obs = _extract_obs(new_state.obs)
        reward = jnp.asarray(new_state.reward, dtype=jnp.float32)
        done = jnp.asarray(new_state.done, dtype=jnp.bool_)

        return obs, reward, done, new_state

    def get_info(self) -> Dict[str, Union[str, int]]:
        """获取环境信息 / Get environment info"""
        return {
            'name': self.env_name,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
        }


# ==========================================
# Episode Rollout / Episode 执行
# ==========================================

def rollout_single_episode(
    params_dict: ParamsDict,
    initial_obs: JAXArray,
    initial_state: EnvState,
    max_episode_length: int,
    env_wrapper: JAXEnvironmentWrapper,
    policy: JAXPolicyNetwork
) -> float:
    """
    运行单个 episode / Run single episode

    使用 jax.lax.scan 实现高效的步进循环。
    Uses jax.lax.scan for efficient stepping loop.
    """
    def step_fn(carry, _):
        """单步函数 / Single step function"""
        obs, state, total_reward, done = carry

        def if_not_done(args):
            curr_obs, curr_state, curr_total, _ = args
            action = policy.forward(params_dict, curr_obs)
            new_obs, reward, new_done, new_state = env_wrapper.step(
                curr_state, action
            )
            return (new_obs, new_state, curr_total + reward, new_done)

        obs, state, total_reward, done = jax.lax.cond(
            done,
            lambda x: x,  # 如果 done，保持不变 / If done, keep unchanged
            if_not_done,
            (obs, state, total_reward, done)
        )

        return (obs, state, total_reward, done), None

    # 初始化 carry / Initialize carry
    initial_carry = (
        initial_obs,
        initial_state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(False, dtype=jnp.bool_)
    )

    (final_obs, final_state, final_reward, _), _ = jax.lax.scan(
        step_fn,
        initial_carry,
        None,
        length=max_episode_length
    )

    return final_reward


# ==========================================
# 辅助函数：处理环境观测 / Helper: Handle environment observations
# ==========================================

def _extract_obs(obs_data: Union[JAXArray, Dict[str, JAXArray]]) -> JAXArray:
    """
    从环境观测中提取数组 / Extract array from environment observation

    参数 / Args:
        obs_data: 观测数据，可能是 array 或 dict / Observation data, can be array or dict

    返回 / Returns:
        JAXArray: 观测数组 / Observation array
    """
    if isinstance(obs_data, dict):
        return jnp.array(obs_data.get('state', obs_data))
    return jnp.array(obs_data)


# ==========================================
# 批量评估 / Batch Evaluation
# ==========================================

def evaluate_population(
    params_batch: JAXArray,
    keys_batch: JAXArray,
    env_wrapper: JAXEnvironmentWrapper,
    policy: JAXPolicyNetwork,
    num_episodes: int,
    max_episode_length: int
) -> JAXArray:
    """
    批量评估种群（JAX 编译函数）/ Batch evaluate population (JAX compiled function)

    两层 vmap：外层并行 individuals，内层并行 episodes。
    Two-level vmap: outer for individuals, inner for episodes.
    """

    def run_one_episode(flat_params: JAXArray, key: JAXKey) -> float:
        """运行单个 episode / Run single episode"""
        params_dict = policy.unpack_params(flat_params)
        obs, state = env_wrapper.reset(key)
        reward = rollout_single_episode(
            params_dict, obs, state, max_episode_length,
            env_wrapper, policy
        )
        return reward

    def evaluate_one_individual(
        flat_params: JAXArray,
        individual_keys: JAXArray
    ) -> float:
        """评估单个个体 / Evaluate single individual"""
        # vmap 并行 episodes / vmap parallel episodes
        episode_rewards = jax.vmap(
            run_one_episode,
            in_axes=(None, 0)
        )(flat_params, individual_keys)
        # 使用 jnp.mean 聚合 / Aggregate using jnp.mean
        aggregated_reward = jnp.mean(episode_rewards)
        return -aggregated_reward  # 转换为最小化问题 / Convert to minimization

    # vmap 并行 individuals / vmap parallel individuals
    fitness = jax.vmap(evaluate_one_individual)(params_batch, keys_batch)

    return fitness


# ==========================================
# MuJoCo 基准问题基类 / MuJoCo Benchmark Base Class
# ==========================================

class MuJoCoBenchmarks(ABC):
    """
    MuJoCo 神经进化基准问题基类 / MuJoCo NeuroEvolution benchmark problem base class

    此类为所有 MuJoCo NE 基准问题提供基础功能。
    This class provides base functionality for all MuJoCo NE benchmark problems.

    【与 mujoco_playground 一致 / Consistent with mujoco_playground】
    - 使用固定的评估参数 / Uses fixed evaluation parameters
    - 使用 2 层 MLP 策略网络 / Uses 2-layer MLP policy network
    - 使用 registry.load() 加载环境 / Uses registry.load() for environment
    """

    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        初始化 MuJoCo 基准问题 / Initialize MuJoCo benchmark problem

        参数 / Args:
            random_seed: 随机种子 / Random seed
        """
        self.lb = WEIGHT_LOWER_BOUND
        self.ub = WEIGHT_UPPER_BOUND
        self.seed = random_seed
        self.num_episodes = NUM_EPISODES
        self.max_episode_length = MAX_EPISODE_LENGTH

        # 初始化内部 PRNGKey / Initialize internal PRNGKey
        self._jax_key: JAXKey = jrand.PRNGKey(random_seed)

        # JIT 编译函数（延迟编译）/ JIT compiled function (lazy compilation)
        self._compiled_evaluate: Optional[Callable[[JAXArray, JAXArray], JAXArray]] = None

    @abstractmethod
    def get_env_name(self) -> str:
        """返回环境名称 / Return environment name"""
        pass

    def _setup_model(self):
        """初始化环境包装器和策略网络 / Initialize env wrapper and policy network"""
        # 使用 __dict__ 直接检查避免触发 property / Use __dict__ to avoid triggering property
        if '_env_wrapper' in self.__dict__ and '_policy' in self.__dict__:
            return

        env_name = self.get_env_name()

        # 延迟加载环境包装器 / Lazy load environment wrapper
        if '_env_wrapper' not in self.__dict__:
            self._env_wrapper = JAXEnvironmentWrapper(env_name=env_name)

        # 获取环境维度 / Get environment dimensions
        self.obs_dim: int = self._env_wrapper.obs_dim
        self.action_dim: int = self._env_wrapper.action_dim

        # 创建策略网络（2层 MLP）/ Create policy network (2-layer MLP)
        self._policy = JAXPolicyNetwork(
            obs_dim=self.obs_dim,
            hidden_dim=HIDDEN_DIM,
            action_dim=self.action_dim
        )

        # 决策变量维度 / Decision variable dimension
        self.dim: int = self._policy.param_size

    @property
    def env_wrapper(self):
        """延迟加载环境包装器 / Lazy load environment wrapper"""
        if '_env_wrapper' not in self.__dict__:
            self._setup_model()
        return self._env_wrapper

    @property
    def policy(self):
        """延迟加载策略网络 / Lazy load policy network"""
        if '_policy' not in self.__dict__:
            self._setup_model()
        return self._policy

    def _compile_evaluation(self) -> Callable[[JAXArray, JAXArray], JAXArray]:
        """编译评估函数 / Compile evaluation function"""
        @jax.jit
        def compiled_fn(
            params_batch: JAXArray,
            keys_batch: JAXArray
        ) -> JAXArray:
            return evaluate_population(
                params_batch, keys_batch,
                self.env_wrapper, self.policy,
                self.num_episodes, self.max_episode_length
            )

        return compiled_fn

    def _generate_keys(self, batch_size: int) -> JAXArray:
        """生成批量评估所需的 PRNGKeys / Generate PRNGKeys for batch evaluation"""
        total_keys = batch_size * self.num_episodes
        all_keys = jrand.split(self._jax_key, total_keys + 1)

        # 第一个 key 作为新的主 key / First key as new master key
        self._jax_key = all_keys[0]

        # 剩余 keys 重塑 / Reshape remaining keys
        use_keys = all_keys[1:]
        keys = use_keys.reshape((batch_size, self.num_episodes, 2))

        return keys

    def __call__(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """使对象可调用 / Make object callable"""
        return self.compute(x)

    def compute(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """
        计算适应度值 / Compute fitness values

        【评估协议 / Evaluation Protocol】
        - 输入 (D,) → 输出 (1,) / Input (D,) → Output (1,)
        - 输入 (N, D) → 输出 (N,) / Input (N, D) → Output (N,)
        - 首次调用触发 JIT 编译 / First call triggers JIT compilation
        """
        # 输入检查 / Input validation
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)

        if x.ndim == 1:
            x = x[np.newaxis, :]

        N, D = x.shape

        # 设置模型（如果尚未设置）/ Setup model if not already done
        if '_env_wrapper' not in self.__dict__:
            self._setup_model()

        if D != self.dim:
            raise ValueError(
                f"输入维度 / Input dimension {D} 与问题维度 / doesn't match problem dimension {self.dim}"
            )

        if np.any(x < self.lb) or np.any(x > self.ub):
            warnings.warn(
                f"参数超出推荐范围 / Parameters outside recommended range [{self.lb}, {self.ub}]",
                UserWarning
            )

        # 延迟编译 / Lazy compilation
        if self._compiled_evaluate is None:
            self._compiled_evaluate = self._compile_evaluation()

        # JAX 侧：核心计算 / JAX side: Core computation
        x_jax = jnp.array(x)
        keys_jax = self._generate_keys(N)
        fitness_jax = self._compiled_evaluate(x_jax, keys_jax)

        # Python 侧：输出转换 / Python side: Output conversion
        fitness = np.array(fitness_jax, dtype=np.float64)

        return fitness

    def info(self) -> Dict[str, Union[float, int, Dict, str]]:
        """返回问题的元信息 / Return problem metadata"""
        # 确保模型已设置 / Ensure model is set up
        if '_env_wrapper' not in self.__dict__:
            self._setup_model()

        return {
            'dimension': self.dim,
            'lower': self.lb,
            'upper': self.ub,
            'env_name': self.get_env_name(),
            'category': MUJOCO_ENVS.get(self.get_env_name(), 'Unknown'),
            'num_episodes': self.num_episodes,
            'max_episode_length': self.max_episode_length,
            'env_info': self.env_wrapper.get_info()
        }


# ==========================================
# 工厂函数 / Factory Function
# ==========================================

def _create_mujoco_problem_class(env_name: str) -> type:
    """
    工厂函数：创建 MuJoCo 问题类 / Factory function to create MuJoCo problem classes.

    此函数动态创建一个继承自 MuJoCoBenchmarks 的问题类。
    Each function dynamically creates a problem class inheriting from MuJoCoBenchmarks.

    Args:
        env_name (str): mujoco_playground 环境名称 / mujoco_playground environment name

    Returns:
        type: MuJoCo 问题类 / MuJoCo problem class (继承自 MuJoCoBenchmarks / inherits from MuJoCoBenchmarks)

    Example:
        >>> SwimmerProblem = _create_mujoco_problem_class('SwimmerSwimmer6')
        >>> problem = SwimmerProblem()
        >>> problem.get_env_name()
        'SwimmerSwimmer6'
    """

    class MuJoCoProblem(MuJoCoBenchmarks):
        """动态生成的 MuJoCo 问题类 / Dynamically generated MuJoCo problem class."""

        def __init__(self, random_seed: int = RANDOM_SEED, device: Optional[str] = None):
            """
            初始化 MuJoCo 问题 / Initialize MuJoCo problem.

            Args:
                random_seed: 随机种子 / Random seed
                device: 设备（目前 JAX 自动处理，保留参数以保持一致性）/ Device (JAX handles automatically, kept for consistency)
            """
            super().__init__(random_seed=random_seed)
            self._env_name = env_name
            self._device = device if device is not None else _detect_device()
            # 立即设置模型 / Setup model immediately
            self._setup_model()

        def get_env_name(self) -> str:
            """返回环境名称 / Return environment name."""
            return self._env_name

        def __str__(self) -> str:
            """返回问题的字符串表示 / Return string representation of problem."""
            return f"MuJoCo_{self._env_name}"

    # 设置类的名称以便更好地调试 / Set class name for better debugging
    MuJoCoProblem.__name__ = f"MuJoCo_{env_name}"
    return MuJoCoProblem


# ==========================================
# 创建所有问题类 / Create All Problem Classes
# ==========================================

# 为每个环境创建问题类 / Create problem class for each environment
# 总共 54 个环境 × 1 个网络配置 = 54 个问题 / Total: 54 environments × 1 network config = 54 problems
_problem_classes: Dict[int, type] = {}
for func_id, env_name in enumerate(ENV_NAMES, start=1):
    cls = _create_mujoco_problem_class(env_name)
    _problem_classes[func_id] = cls


# ==========================================
# 基准测试入口点 / Benchmark Entry Point
# ==========================================

class Benchmark:
    """
    MuJoCo 神经进化基准测试类 / MuJoCo NeuroEvolution benchmark class

    提供与 cec2013lsgo 兼容的统一接口。
    Provides a unified interface compatible with cec2013lsgo.

    【问题数量 / Number of Problems】
    总共 54 个问题，涵盖所有 mujoco_playground 环境。
    Total: 54 problems covering all mujoco_playground environments.

    【问题分布 / Problem Distribution】
    - Classic Control: 9 problems (IDs 1-9)
    - Locomotion: 11 problems (IDs 10-20)
    - Manipulation: 15 problems (IDs 21-35)
    - Humanoid/Robot: 19 problems (IDs 36-54)

    【设备支持 / Device Support】
    - 自动检测并使用 GPU (CUDA) 如果可用 / Auto-detect and use GPU (CUDA) if available
    - 回退到 CPU 如果 GPU 不可用 / Fallback to CPU if GPU not available

    Attributes:
        _num_functions (int): 可用函数的总数 / Total number of available functions (54)
        _random_seed (int): 随机种子 / Random seed
        _device (str): 使用的设备 / Device being used ('cuda' or 'cpu')

    Example:
        >>> benchmark = Benchmark(random_seed=42)  # Auto-detect device
        >>> benchmark = Benchmark(random_seed=42, device='cuda')  # Force GPU
        >>> problem = benchmark.get_function(1)  # AcrobotSwingup
        >>> info = benchmark.get_info(1)
        >>> print(f"Environment: {info['env_name']}, Dimension: {info['dimension']}")
    """

    def __init__(self, random_seed: int = RANDOM_SEED, device: Optional[str] = None):
        """
        初始化 MuJoCo 基准测试 / Initialize MuJoCo Benchmark

        参数 / Args:
            random_seed: 随机种子 / Random seed (default: 42)
            device: 使用的设备 / Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self._num_functions = len(_problem_classes)  # 54 个问题 / 54 problems
        self._random_seed = random_seed
        self._device = device if device is not None else _detect_device()
        print(f"[MuJoCo 基准测试/MuJoCo Benchmark] 使用设备/Using device: {self._device}")
        print(f"[MuJoCo 基准测试/MuJoCo Benchmark] 使用种子/Using seed: {random_seed}")
        print(f"[MuJoCo 基准测试/MuJoCo Benchmark] 问题数量/Number of problems: {self._num_functions}")

    def get_function(self, func_id: int) -> MuJoCoBenchmarks:
        """
        获取指定问题的可调用对象 / Get specific problem by function ID

        【问题列表 / Problem List】
        - IDs 1-9: Classic Control (AcrobotSwingup, BallInCup, Cartpole*, PendulumSwingup, PointMass)
        - IDs 10-20: Locomotion (SwimmerSwimmer6, CheetahRun, FishSwim, Hopper*, Walker*, Humanoid*)
        - IDs 21-35: Manipulation (Finger*, Reacher*, Panda*, Aloha*, Leap*, Aero*)
        - IDs 36-53: Humanoid/Robot (Apollo*, Barkour*, Berkeley*, G1*, Go1*, H1*, Op3*, Spot*, T1*)
        - ID 54: Other

        参数 / Args:
            func_id: 函数 ID (1-54) / Function ID (1-54)

        返回 / Returns:
            问题类的实例 / Instance of problem class

        Raises:
            ValueError: 当 func_id 超出范围时 / When func_id is out of range

        Example:
            >>> benchmark = Benchmark()
            >>> problem = benchmark.get_function(1)  # AcrobotSwingup
            >>> fitness = problem(np.random.uniform(-1.0, 1.0, size=(10, problem.dim)))
        """
        if func_id < 1 or func_id > self._num_functions:
            raise ValueError(
                f"函数 ID / Function ID {func_id} 超出范围 / is out of range "
                f"[1, {self._num_functions}]"
            )

        return _problem_classes[func_id](random_seed=self._random_seed, device=self._device)

    def get_info(self, func_id: int) -> Dict[str, Union[float, int, Dict]]:
        """
        获取指定问题的信息 / Get information about specific problem

        参数 / Args:
            func_id: 函数 ID (1-54) / Function ID (1-54)

        返回 / Returns:
            包含问题元信息的字典 / Dict with problem metadata

        Raises:
            ValueError: 当 func_id 超出范围时 / When func_id is out of range

        Example:
            >>> benchmark = Benchmark()
            >>> info = benchmark.get_info(1)
            >>> print(f"Environment: {info['env_name']}, Dimension: {info['dimension']}")
        """
        if func_id < 1 or func_id > self._num_functions:
            raise ValueError(
                f"函数 ID / Function ID {func_id} 超出范围 / is out of range "
                f"[1, {self._num_functions}]"
            )

        problem = _problem_classes[func_id](random_seed=self._random_seed, device=self._device)
        return problem.info()

    def get_num_functions(self) -> int:
        """
        返回可用函数的总数 / Return total number of available functions

        返回 / Returns:
            int: 可用函数的数量 / Number of available functions (54)
        """
        return self._num_functions

    def list_functions(self) -> List[Dict[str, Union[int, str]]]:
        """
        列出所有可用的函数及其 ID / List all available functions with their IDs

        注意：
        - 此方法返回基本信息，不包括维度 / This method returns basic info, excluding dimension
        - 维度信息可通过 get_info() 获取 / Dimension info available via get_info()
        - 这样可以避免在导入时加载所有环境 / This avoids loading all environments during import

        返回 / Returns:
            list[dict]: 包含函数信息的字典列表 / List of dicts with function info
                每个字典包含 / Each dict contains:
                - 'id' (int): 函数 ID / Function ID
                - 'env_name' (str): 环境名称 / Environment name
                - 'category' (str): 类别 / Category
                - 'dimension' (int): 决策变量维度 / Decision variable dimension (None, use get_info() to get)

        Example:
            >>> benchmark = Benchmark()
            >>> functions = benchmark.list_functions()
            >>> for f in functions[:5]:
            ...     info = benchmark.get_info(f['id'])
            ...     print(f"ID {f['id']}: {f['env_name']}, category={f['category']}, dim={info['dimension']}")
        """
        result = []
        for func_id, env_name in zip(sorted(_problem_classes.keys()), ENV_NAMES):
            result.append({
                'id': func_id,
                'env_name': env_name,
                'category': MUJOCO_ENVS.get(env_name, 'Unknown'),
                'dimension': None,  # 使用 get_info() 获取维度 / Use get_info() to get dimension
            })
        return result
