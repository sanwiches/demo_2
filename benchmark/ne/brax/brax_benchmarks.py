import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import os
import warnings

# 抑制来自 brax.io.mjcf 的已知 Brax 维护警告 / Suppress only the known Brax maintenance warning from brax.io.mjcf.
# 注意：仅抑制此特定警告以避免掩盖其他问题 / Note: Only suppress this specific warning to avoid masking other issues
warnings.filterwarnings(
    "ignore",
    message=r"Brax System, piplines and environments are not actively being maintained.*",
    category=UserWarning,
    module=r"brax\.io\.mjcf",
)

# ==========================================
# 设备配置 / Device Configuration
# ==========================================

# 关键：在任何 JAX 相关导入之前配置 JAX / CRITICAL: Configure JAX BEFORE any JAX-related imports
# 这必须在模块级别、任何 JAX 操作之前完成 / This MUST happen at the module level before any JAX operations
# 注意：模块导入后更改设备可能无法正常工作 / Note: Changing device after module import may not work properly

if torch.cuda.is_available():
    os.environ["JAX_PLATFORMS"] = ""  # 自动检测，优先使用 CUDA / Auto-detect, prefer CUDA
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 禁用预分配以避免内存问题 / Disable pre-allocation to avoid memory issues
    DEFAULT_DEVICE = "cuda"

    # 立即导入 JAX 并配置使用 CUDA / Import JAX immediately and configure to use CUDA
    import jax

    # 获取 CUDA 设备并将其设置为默认 / Get CUDA device and set as default
    all_devices = jax.devices()
    cuda_devices = [d for d in all_devices if "cuda" in str(d).lower()]

    if cuda_devices:
        # 使用 jax.config 将 CUDA 设置为默认平台 / Use jax.config to set CUDA as default platform
        # 注意：jax.config 是属性，不是子模块 / Note: jax.config is an attribute, not a submodule
        jax.config.update("jax_platforms", "cuda")
else:
    os.environ["JAX_PLATFORMS"] = "cpu"
    DEFAULT_DEVICE = "cpu"
    import jax  # 为一致性导入 / Import for consistency


def get_default_device():
    """获取默认设备 / Get the default device (CUDA if available, else CPU)."""
    return DEFAULT_DEVICE


# ==========================================
# 常量 / Constants
# ==========================================

# MLP 隐藏层大小 / MLP hidden layer size
# 控制神经网络中隐藏层的神经元数量 / Controls the number of neurons in hidden layers
HIDDEN_DIM = 32

# 权重初始化边界 / Weight initialization bounds
# 定义神经网络权重的初始化范围 / Defines the initialization range for neural network weights
WEIGHT_LOWER_BOUND = -0.2
WEIGHT_UPPER_BOUND = 0.2

# 适应度变换 / Fitness transformation
# 将最大化（奖励）转换为最小化（适应度）/ Converts maximization (reward) to minimization (fitness)
# 更高的奖励对应更低的适应度值 / Higher rewards correspond to lower fitness values
FITNESS_SCALE = 0 # 缩放因子 / Scale factor  这里改为0，因为奖励已经是负数了，不需要再进行缩放 / Set to 0 because rewards are already negative, no need for scaling
PENALTY_VALUE = -1000  # 对无效奖励的惩罚值 / Penalty value for invalid rewards (-5 * 200)

# 环境评估设置 / Environment evaluation settings
MAX_EPISODE_LENGTH = 200  # 每个episode的最大步数 / Maximum steps per episode
NUM_EPISODES = 10  # 用于评估的episodes数量 / Number of episodes for evaluation


# ==========================================
# Neural Network Policy / 神经网络策略
# ==========================================

class MLP(nn.Module):
    """
    多层感知机神经网络 / Multi-Layer Perceptron neural network.

    用于神经进化 / Used for neuroevolution.
    """

    def __init__(self, state_dim, action_dim, hidden_layer_num):
        """
        初始化 MLP / Initialize MLP.

        Args:
            state_dim: 状态维度 / State dimension
            action_dim: 动作维度 / Action dimension
            hidden_layer_num: 隐藏层数量 / Number of hidden layers
        """
        super(MLP, self).__init__()
        self.networks = nn.ModuleList()
        self.networks.append(nn.Sequential(nn.Linear(state_dim, HIDDEN_DIM), nn.Tanh()))
        for _ in range(hidden_layer_num):
            self.networks.append(nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()))
        self.networks.append(nn.Linear(HIDDEN_DIM, action_dim))

    def forward(self, state):
        """前向传播 / Forward pass."""
        for layer in self.networks:
            state = layer(state)
        return torch.tanh(state)


# ==========================================
# 环境配置 / Environment Configuration
# ==========================================

# 支持的强化学习环境及其状态/动作维度 / Supported RL environments and their state/action dimensions
# state_dim: 观测空间的维度 / Dimension of observation space
# action_dim: 动作空间的维度 / Dimension of action space
ENVS = {
    'ant': {'state_dim': 27, 'action_dim': 8},
    'halfcheetah': {'state_dim': 17, 'action_dim': 6},
    'hopper': {'state_dim': 11, 'action_dim': 3},
    'humanoid': {'state_dim': 244, 'action_dim': 17},
    'humanoidstandup': {'state_dim': 244, 'action_dim': 17},
    'inverted_pendulum': {'state_dim': 4, 'action_dim': 1},
    'inverted_double_pendulum': {'state_dim': 8, 'action_dim': 1},
    'pusher': {'state_dim': 23, 'action_dim': 7},
    'reacher': {'state_dim': 11, 'action_dim': 2},
    'swimmer': {'state_dim': 8, 'action_dim': 2},
    'walker2d': {'state_dim': 17, 'action_dim': 6},
}

# 支持的模型深度（隐藏层数量）/ Supported model depths (number of hidden layers)
# 每个深度值对应不同复杂度的神经网络 / Each depth value corresponds to a different neural network complexity
MODEL_DEPTHS = [0, 1, 2, 3, 4, 5]


# ==========================================
# NE 基准问题基类 / Base Class for NE Problems
# ==========================================

class NEBenchmarks(ABC):
    """
    神经进化基准问题基类 / Base class for NeuroEvolution benchmark problems.

    此类为所有 NE 基准问题提供基础功能。
    This class provides base functionality for all NE benchmark problems.

    Attributes:
        lb (float): 权重下界 / Weight lower bound
        ub (float): 权重上界 / Weight upper bound
        device (str): 使用的设备 / Device to use ('cuda' or 'cpu')
        seed (int): 随机种子 / Random seed for reproducibility
        dim (int): 决策变量维度 / Dimension of decision variables
    """

    def __init__(self, device=None):
        """
        初始化 NE 基准问题 / Initialize NE benchmark problem.

        Args:
            device: 设备 ('cuda', 'cpu', 或 None 表示自动检测) / Device to use ('cuda', 'cpu', or None for auto-detect).
                    注意：模块导入后更改设备可能无法正常工作 / Note: Changing device after module import may not work properly
                    因为 JAX 在模块级别配置 / because JAX is configured at module level.

        Raises:
            None: 此构造函数不抛出异常 / This constructor does not raise exceptions
        """
        self.lb = WEIGHT_LOWER_BOUND
        self.ub = WEIGHT_UPPER_BOUND
        self.seed = 42  # 默认随机种子 / Default random seed
        self.evaluator = None  # Brax 评估器 / Brax evaluator
        self.adapter = None  # 参数适配器 / Parameter adapter
        self.init = False  # 初始化标志 / Initialization flag
        self.device = device if device is not None else DEFAULT_DEVICE

    @abstractmethod
    def get_env_name(self):
        """
        返回环境名称 / Return the environment name.
        """
        pass

    @abstractmethod
    def get_model_depth(self):
        """
        返回模型深度 / Return the model depth.
        """
        pass

    def _setup_model(self):
        """
        初始化神经网络模型并计算维度 / Initialize the neural network model and compute dimension.
        """
        env_name = self.get_env_name()
        model_depth = self.get_model_depth()

        state_dim = ENVS[env_name]['state_dim']
        action_dim = ENVS[env_name]['action_dim']

        self.torch_device = torch.device(self.device)
        self.nn_model = MLP(state_dim, action_dim, model_depth)
        self.nn_model.to(self.torch_device)
        self.dim = sum(p.numel() for p in self.nn_model.parameters())

    def info(self):
        """
        返回问题信息 / Return problem information.

        Returns:
            包含维度、边界、环境名称、模型深度的字典 / Dict with dimension, bounds, env_name, model_depth
        """
        return {
            'dimension': self.dim,
            'lower': self.lb,
            'upper': self.ub,
            'env_name': self.get_env_name(),
            'model_depth': self.get_model_depth(),
        }

    def __call__(self, x):
        """
        评估目标函数 / Evaluate the objective function.
        """
        return self.compute(x)

    def compute(self, x):
        """
        在环境中评估神经网络参数 / Evaluate neural network parameters in the environment.

        此方法是基准测试的核心，执行以下步骤：
        This method is the core of benchmarking, performing:
        1. 设置 JAX 设备上下文 / Setup JAX device context
        2. 初始化模型和适配器 / Initialize model and adapter
        3. 创建/更新评估器 / Create/update evaluator
        4. 验证输入维度 / Validate input dimensions
        5. 评估种群并返回适应度 / Evaluate population and return fitness

        Args:
            x: 待评估的种群 / Population to evaluate, shape (pop_size, dim)
               可以是 numpy 数组或 torch 张量 / Can be numpy array or torch tensor

        Returns:
            numpy.ndarray: 适应度值（最小化问题）/ Fitness values (minimization problem)
                          形状为 (pop_size,) / Shape is (pop_size,)

        Raises:
            ValueError: 当输入维度不匹配时 / When input dimensions don't match

        Note:
            - 适应度 = FITNESS_SCALE - 奖励 / Fitness = FITNESS_SCALE - reward
            - 更高的奖励 = 更低的适应度（最小化问题）/ Higher reward = lower fitness (minimization)
        """
        import jax

        # 在任何导入之前首先获取可用的 JAX 设备 / Get available JAX devices FIRST, before any imports
        available_devices = jax.devices()

        # 根据 self.device 偏好选择设备 / Select device based on self.device preference
        jax_device = None
        if self.device == "cuda":
            for dev in available_devices:
                if "cuda" in str(dev).lower() or dev.platform == "cuda":
                    jax_device = dev
                    break
        if jax_device is None:
            jax_device = available_devices[0]

        # 对所有 JAX 相关操作使用 JAX 设备上下文 / Use JAX device context for ALL JAX-related operations
        with jax.default_device(jax_device):
            from evox.utils import ParamsAndVector
            from evox.problems.neuroevolution.brax import BraxProblem

            # 如果尚未完成，则设置模型 / Setup model if not already done
            if not hasattr(self, 'nn_model'):
                self._setup_model()

            # 如果需要，设置适配器（在设备上下文内！）/ Setup adapter if needed (INSIDE device context!)
            if self.adapter is None:
                self.adapter = ParamsAndVector(dummy_model=self.nn_model)

        # 处理输入维度 / Handle input dimension
        # 确保 x 是 2D 的 / Ensure x is 2D (必须在计算 pop_size 之前!)
        x = np.atleast_2d(x)
        if x.shape[0] == 1 and x.shape[1] == self.dim:
            # 1D 输入 (dim,) -> 2D (1, dim) / Convert 1D to 2D
            pass
        elif x.shape[-1] == self.dim:
            # 正确的形状 / Correct shape
            pass
        elif x.shape[0] == self.dim:
            # 自动转置修复常见错误 / Auto-transpose to fix common mistake
            x = x.T
            print(f"[警告/Warning] 输入 x 已转置/Input x was transposed: 新形状/new shape = {x.shape}")
        else:
            raise ValueError(
                f"[维度错误/Dimension Error] 期望形状/Expected shape (batch_size, {self.dim}), 得到/got {x.shape}"
            )

        # 创建或更新评估器（在设备上下文内！）/ Create or update evaluator (INSIDE device context!)
        # 必须在维度处理之后计算 pop_size / Must compute pop_size AFTER dimension handling
        pop_size = x.shape[0]
        if self.evaluator is None or (self.init and pop_size != self.evaluator.pop_size):
            with jax.default_device(jax_device):
                self.evaluator = BraxProblem(
                    policy=self.nn_model,
                    env_name=self.get_env_name(),
                    max_episode_length=MAX_EPISODE_LENGTH,
                    num_episodes=NUM_EPISODES,
                    pop_size=pop_size,
                    seed=self.seed,
                    reduce_fn=torch.mean,
                    device=self.torch_device,
                )
            self.init = True

        # 评估 - 也在设备上下文内 / Evaluate - also within device context
        x_tensor = torch.tensor(x, device=self.torch_device).float()
        nn_population = self.adapter.batched_to_params(x_tensor)

        with jax.default_device(jax_device):
            rewards = self.evaluator.evaluate(nn_population)

        # 处理 NaN 和 Inf / Handle NaN and Inf
        # 无效奖励将被惩罚值替换 / Invalid rewards will be replaced with penalty value
        rewards[torch.isnan(rewards)] = PENALTY_VALUE
        rewards[torch.isinf(rewards)] = PENALTY_VALUE

        # 转换为最小化问题（更高的奖励 = 更低的适应度）/ Convert to minimization problem (higher reward = lower fitness)
        rewards = rewards.cpu().numpy()
        return FITNESS_SCALE - rewards


# ==========================================
# 生成的基准问题类 / Generated Problem Classes
# ==========================================

# 函数 ID 映射规则 / Function ID mapping rule:
#   id = env_index * 6 + depth + 1
# 其中 / where:
#   env_index: ant=0, halfcheetah=1, hopper=2, ...
#   depth: 0-5（模型深度/model depth）
#
# 示例 / Example:
#   - ant, depth=0  -> ID 1  (0*6 + 0 + 1)
#   - ant, depth=5  -> ID 6  (0*6 + 5 + 1)
#   - halfcheetah, depth=0 -> ID 7  (1*6 + 0 + 1)

ENV_NAMES = list(ENVS.keys())


def _create_ne_problem_class(env_name, model_depth):
    """
    工厂函数：创建 NE 问题类 / Factory function to create NE problem classes.

    此函数动态创建一个继承自 NEBenchmarks 的问题类，
    Each function dynamically creates a problem class inheriting from NEBenchmarks,
    用于特定的环境和模型深度组合。
    for a specific environment and model depth combination.

    Args:
        env_name (str): 环境名称 / Environment name (来自 ENVS 的键 / key from ENVS)
        model_depth (int): 模型深度 / Model depth (0-5，隐藏层数/number of hidden layers)

    Returns:
        type: NE 问题类 / NE problem class (继承自 NEBenchmarks / inherits from NEBenchmarks)

    Example:
        >>> AntDepth0 = _create_ne_problem_class('ant', 0)
        >>> problem = AntDepth0()
        >>> problem.get_env_name()
        'ant'
        >>> problem.get_model_depth()
        0
    """

    class NEProblem(NEBenchmarks):
        """动态生成的 NE 问题类 / Dynamically generated NE problem class."""

        def __init__(self, device=None):
            """
            初始化 NE 问题 / Initialize NE problem.

            Args:
                device: 计算设备 / Compute device ('cuda', 'cpu', or None)
            """
            super().__init__(device=device)
            self.env_name = env_name
            self.model_depth = model_depth
            self._setup_model()

        def get_env_name(self):
            """返回环境名称 / Return environment name."""
            return self.env_name

        def get_model_depth(self):
            """返回模型深度 / Return model depth."""
            return self.model_depth

        def __str__(self):
            """返回问题的字符串表示 / Return string representation of problem."""
            return f"NE_{env_name}_depth{model_depth}"

    # 设置类的名称以便更好地调试 / Set class name for better debugging
    NEProblem.__name__ = f"NE_{env_name.capitalize()}_Depth{model_depth}"
    return NEProblem


# 创建所有问题类 / Create all problem classes
# 这将生成所有环境 × 所有深度的组合 / This generates all combinations of environments × depths
# 总共 11 个环境 × 6 个深度 = 66 个问题 / Total: 11 environments × 6 depths = 66 problems
_problem_classes = {}
for env_idx, env_name in enumerate(ENV_NAMES):
    for depth in MODEL_DEPTHS:
        func_id = env_idx * 6 + depth + 1  # 1-indexed / 从 1 开始索引
        cls = _create_ne_problem_class(env_name, depth)
        _problem_classes[func_id] = cls


# ==========================================
# 基准测试入口点 / Benchmark Entry Point
# ==========================================

class Benchmark:
    """
    神经进化基准测试类 / Benchmark class for NeuroEvolution problems.

    此类提供与 cec2013lsgo 兼容的统一接口。
    This class provides a unified interface compatible with cec2013lsgo.

    此类是访问所有 NE 基准问题的主要入口点，
    This class is the main entry point for accessing all NE benchmark problems,
    支持动态获取问题信息和实例化。
    supporting dynamic problem information retrieval and instantiation.

    Attributes:
        _num_functions (int): 可用函数的总数 / Total number of available functions (66)
        _device (str): 默认计算设备 / Default compute device ('cuda' or 'cpu')

    Example:
        >>> benchmark = Benchmark(device='cuda')
        >>> problem = benchmark.get_function(1)  # ant, depth=0
        >>> info = benchmark.get_info(1)
        >>> print(f"Dimension: {info['dimension']}")
        >>> fitness = problem(np.random.uniform(-0.2, 0.2, size=(10, info['dimension'])))
    """

    def __init__(self, device=None):
        """
        初始化 NE 基准测试 / Initialize NE Benchmark.

        Args:
            device (str, optional): 使用的设备 / Device to use
                - 'cuda': 使用 NVIDIA GPU / Use NVIDIA GPU
                - 'cpu': 使用 CPU / Use CPU
                - None: 自动检测 / Auto-detect (default)
        """
        self._num_functions = len(_problem_classes)
        self._device = device if device is not None else DEFAULT_DEVICE
        print(f"[NE 基准测试/NE Benchmark] 使用设备/Using device: {self._device}")

    def get_function(self, func_id):
        """
        通过函数 ID 获取特定的 NE 问题 / Get a specific NE problem by function ID.

        每次调用都会创建一个新的问题实例，具有独立的模型和状态。
        Each call creates a new problem instance with independent model and state.

        函数 ID 映射 / Function ID mapping:
        - ant:                  IDs 1-6   (深度/depths 0-5)
        - halfcheetah:          IDs 7-12  (深度/depths 0-5)
        - hopper:               IDs 13-18 (深度/depths 0-5)
        - humanoid:             IDs 19-24 (深度/depths 0-5)
        - humanoidstandup:      IDs 25-30 (深度/depths 0-5)
        - inverted_pendulum:    IDs 31-36 (深度/depths 0-5)
        - inverted_double_pendulum: IDs 37-42 (深度/depths 0-5)
        - pusher:               IDs 43-48 (深度/depths 0-5)
        - reacher:              IDs 49-54 (深度/depths 0-5)
        - swimmer:              IDs 55-60 (深度/depths 0-5)
        - walker2d:             IDs 61-66 (深度/depths 0-5)

        Args:
            func_id (int): 函数 ID (1-66) / Function ID (1-66)

        Returns:
            NEBenchmarks: NE 问题类的实例 / An instance of the NE problem class

        Raises:
            ValueError: 当 func_id 超出范围时 / When func_id is out of range

        Example:
            >>> benchmark = Benchmark()
            >>> problem = benchmark.get_function(1)  # ant, depth=0
            >>> fitness = problem(np.random.uniform(-0.2, 0.2, size=(5, problem.dim)))
        """
        if func_id < 1 or func_id > self._num_functions:
            raise ValueError(
                f"函数 ID {func_id} 超出范围 / Function id {func_id} is out of range "
                f"[1, {self._num_functions}]"
            )

        return _problem_classes[func_id](device=self._device)

    def get_info(self, func_id):
        """
        获取特定 NE 问题的信息 / Get information about a specific NE problem.

        此方法创建问题实例并返回其元数据，无需实际评估。
        This method creates a problem instance and returns its metadata without evaluation.

        Args:
            func_id (int): 函数 ID (1-66) / Function ID (1-66)

        Returns:
            dict: 包含以下键的字典 / Dict with keys:
                - 'dimension' (int): 决策变量维度 / Dimension of decision variables
                - 'lower' (float): 权重下界 / Weight lower bound (-0.2)
                - 'upper' (float): 权重上界 / Weight upper bound (0.2)
                - 'env_name' (str): 环境名称 / Environment name
                - 'model_depth' (int): 模型深度 / Model depth (0-5)

        Raises:
            ValueError: 当 func_id 超出范围时 / When func_id is out of range

        Example:
            >>> benchmark = Benchmark()
            >>> info = benchmark.get_info(1)
            >>> print(f"Environment: {info['env_name']}, Dimension: {info['dimension']}")
            Environment: ant, Dimension: 1456
        """
        if func_id < 1 or func_id > self._num_functions:
            raise ValueError(
                f"函数 ID {func_id} 超出范围 / Function id {func_id} is out of range "
                f"[1, {self._num_functions}]"
            )

        problem = _problem_classes[func_id](device=self._device)
        return problem.info()

    def get_num_functions(self):
        """
        返回可用函数的总数 / Return the total number of available functions.

        Returns:
            int: 可用函数的数量 / Number of available functions (固定为 66 / fixed at 66)

        Note:
            总数 = 环境数量 × 深度数量 = 11 × 6 = 66
            Total = number of environments × number of depths = 11 × 6 = 66
        """
        return self._num_functions

    def list_functions(self):
        """
        列出所有可用的函数及其 ID / List all available functions with their IDs.

        此方法创建每个问题的临时实例以收集元数据。
        This method creates temporary instances of each problem to collect metadata.

        Returns:
            list[dict]: 包含函数信息的字典列表 / List of dicts with function info
                每个字典包含 / Each dict contains:
                - 'id' (int): 函数 ID / Function ID
                - 'env_name' (str): 环境名称 / Environment name
                - 'model_depth' (int): 模型深度 / Model depth
                - 'dimension' (int): 决策变量维度 / Decision variable dimension

        Example:
            >>> benchmark = Benchmark()
            >>> functions = benchmark.list_functions()
            >>> for f in functions[:5]:
            ...     print(f"ID {f['id']}: {f['env_name']}, depth={f['model_depth']}, dim={f['dimension']}")
        """
        result = []
        for func_id, cls in sorted(_problem_classes.items()):
            problem = cls(device=self._device)
            result.append({
                'id': func_id,
                'env_name': problem.get_env_name(),
                'model_depth': problem.get_model_depth(),
                'dimension': problem.dim,
            })
        return result
