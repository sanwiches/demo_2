# Brax NeuroEvolution Benchmark Library / Brax 神经进化基准测试库

A comprehensive benchmark suite for NeuroEvolution (NE) algorithms using Brax reinforcement learning environments.

使用 Brax 强化学习环境的神经进化算法综合基准测试套件。

## Background / 背景

NeuroEvolution combines evolutionary algorithms with neural networks, optimizing neural network weights through evolutionary strategies rather than gradient-based methods like backpropagation. This library provides a standardized benchmark suite to evaluate NE algorithms across various control tasks with varying problem complexities.

神经进化将进化算法与神经网络结合，通过进化策略（而非反向传播等基于梯度的方法）来优化神经网络权重。本库提供了一个标准化的基准测试套件，用于在各种控制任务上评估神经进化算法。

The benchmark uses **Brax** - a differentiable physics engine for training and evaluating reinforcement learning agents. Each problem requires evolving a neural network policy to control a specific MuJoCo-like environment.

本基准测试使用 **Brax** - 一个可微分物理引擎，用于训练和评估强化学习智能体。每个问题需要进化一个神经网络策略来控制特定的类 MuJoCo 环境。

## Installation Requirements / 安装要求

```bash
# Basic installation / 基础安装
pip install torch numpy evox

# For CUDA support / CUDA 支持
# Install PyTorch with CUDA from: https://pytorch.org/get-started/locally/
```

## Quick Start / 快速开始

```python
from benchmark.ne.brax import Benchmark

# Initialize the benchmark / 初始化基准测试
benchmark = Benchmark()

# Get a specific problem (e.g., Ant environment with depth 0) / 获取特定问题
problem = benchmark.get_function(1)

# Get problem information / 获取问题信息
info = benchmark.get_info(1)
print(f"Dimension: {info['dimension']}")
print(f"Environment: {info['env_name']}")
print(f"Model Depth: {info['model_depth']}")

# Evaluate a population of solutions / 评估一组解
import numpy as np
population = np.random.uniform(low=-0.2, high=0.2, size=(100, info['dimension']))
fitness = problem(population)
```

## Problem Suite / 问题集

The benchmark suite consists of **66 test problems**, generated from the Cartesian product of:

基准测试套件包含 **66 个测试问题**，由以下笛卡尔积生成：

- **11 Brax Environments** (control tasks) / **11 个 Brax 环境**（控制任务）
- **6 Model Depths** (0 to 5 hidden layers) / **6 种模型深度**（0-5 个隐藏层）

### Environment List / 环境列表

| ID Range | Environment / 环境 | State Dim | Action Dim | Description / 描述 |
|----------|-------------------|-----------|------------|---------------------|
| 1-6 | ant | 27 | 8 | Quadruped robot walking / 四足机器人行走 |
| 7-12 | halfcheetah | 17 | 6 | 2D bipedal running / 双足机器人跑动 |
| 13-18 | hopper | 11 | 3 | One-legged hopping / 单脚跳跃 |
| 19-24 | humanoid | 244 | 17 | 3D humanoid walking / 人形机器人行走 |
| 25-30 | humanoidstandup | 244 | 17 | Humanoid stand up / 人形机器人站立 |
| 31-36 | inverted_pendulum | 4 | 1 | Cart-pole balancing / 倒立摆平衡 |
| 37-42 | inverted_double_pendulum | 8 | 1 | Double pendulum / 双摆平衡 |
| 43-48 | pusher | 23 | 7 | Robotic arm pushing / 机械臂推动 |
| 49-54 | reacher | 11 | 2 | Arm reaching / 机械臂到达 |
| 55-60 | swimmer | 8 | 2 | Snake-like swimming / 蛇形游动 |
| 61-66 | walker2d | 17 | 6 | Bipedal walking / 双足行走 |

**Formula / 公式:** `function_id = env_index * 6 + depth + 1`

### Complete Function Dimensions / 完整函数维度

| ID | Environment | Depth | Dimension | ID | Environment | Depth | Dimension |
|----|-------------|-------|-----------|----|-------------|-------|-----------|
| 1 | ant | 0 | 1160 | 34 | inverted_double_pendulum | 3 | 3489 |
| 2 | ant | 1 | 2216 | 35 | inverted_double_pendulum | 4 | 4545 |
| 3 | ant | 2 | 3272 | 36 | inverted_double_pendulum | 5 | 5601 |
| 4 | ant | 3 | 4328 | 37 | pusher | 0 | 999 |
| 5 | ant | 4 | 5384 | 38 | pusher | 1 | 2055 |
| 6 | ant | 5 | 6440 | 39 | pusher | 2 | 3111 |
| 7 | halfcheetah | 0 | 774 | 40 | pusher | 3 | 4167 |
| 8 | halfcheetah | 1 | 1830 | 41 | pusher | 4 | 5223 |
| 9 | halfcheetah | 2 | 2886 | 42 | pusher | 5 | 6279 |
| 10 | halfcheetah | 3 | 3942 | 43 | reacher | 0 | 450 |
| 11 | halfcheetah | 4 | 4998 | 44 | reacher | 1 | 1506 |
| 12 | halfcheetah | 5 | 6054 | 45 | reacher | 2 | 2562 |
| 13 | hopper | 0 | 483 | 46 | reacher | 3 | 3618 |
| 14 | hopper | 1 | 1539 | 47 | reacher | 4 | 4674 |
| 15 | hopper | 2 | 2595 | 48 | reacher | 5 | 5730 |
| 16 | hopper | 3 | 3651 | 49 | swimmer | 0 | 354 |
| 17 | hopper | 4 | 4707 | 50 | swimmer | 1 | 1410 |
| 18 | hopper | 5 | 5763 | 51 | swimmer | 2 | 2466 |
| 19 | humanoid | 0 | 8401 | 52 | swimmer | 3 | 3522 |
| 20 | humanoid | 1 | 9457 | 53 | swimmer | 4 | 4578 |
| 21 | humanoid | 2 | 10513 | 54 | swimmer | 5 | 5634 |
| 22 | humanoid | 3 | 11569 | 55 | walker2d | 0 | 774 |
| 23 | humanoid | 4 | 12625 | 56 | walker2d | 1 | 1830 |
| 24 | humanoid | 5 | 13681 | 57 | walker2d | 2 | 2886 |
| 25 | humanoidstandup | 0 | 8401 | 58 | walker2d | 3 | 3942 |
| 26 | humanoidstandup | 1 | 9457 | 59 | walker2d | 4 | 4998 |
| 27 | humanoidstandup | 2 | 10513 | 60 | walker2d | 5 | 6054 |
| 28 | humanoidstandup | 3 | 11569 | 61 | inverted_pendulum | 0 | 193 |
| 29 | humanoidstandup | 4 | 12625 | 62 | inverted_pendulum | 1 | 1249 |
| 30 | humanoidstandup | 5 | 13681 | 63 | inverted_pendulum | 2 | 2305 |
| 31 | inverted_pendulum | 0 | 193 | 64 | inverted_pendulum | 3 | 3361 |
| 32 | inverted_pendulum | 1 | 1249 | 65 | inverted_pendulum | 4 | 4417 |
| 33 | inverted_pendulum | 2 | 2305 | 66 | inverted_pendulum | 5 | 5473 |

## Neural Network Architecture / 神经网络架构

All problems use an MLP (Multi-Layer Perceptron) policy with:

所有问题使用 MLP（多层感知机）策略：

- **Input / 输入:** State dimension (varies by environment) / 状态维度（因环境而异）
- **Hidden Layers / 隐藏层:**
  - First layer: Linear(state_dim → 32) + Tanh
  - Middle layers: depth × [Linear(32 → 32) + Tanh]
  - Total layers = depth + 2
- **Output / 输出:** Action dimension (varies by environment) + Tanh

**Parameter Count Formula / 参数数量公式:**
```
params = (state_dim * 32) + 32 + (depth * 32 * 32) + (depth * 32) + (32 * action_dim) + action_dim
```

## Model Depth Impact / 模型深度影响

| Depth / 深度 | Hidden Layers / 隐藏层数 | Complexity / 复杂度 |
|-------------|------------------------|-------------------|
| 0 | 1 | Easiest - single layer / 最简单 - 单层 |
| 1 | 2 | Easy / 简单 |
| 2 | 3 | Medium / 中等 |
| 3 | 4 | Medium-Hard / 中等偏难 |
| 4 | 5 | Hard / 困难 |
| 5 | 6 | Hardest - deepest network / 最难 - 最深网络 |

## Search Space / 搜索空间

- **Lower Bound / 下界:** -0.2
- **Upper Bound / 上界:** 0.2
- **Initialization / 初始化:** Uniform random in [-0.2, 0.2]

## Evaluation Protocol / 评估协议

Each solution is evaluated by:

每个解通过以下方式评估：

1. Encoding weights into the neural network policy / 将权重编码到神经网络策略
2. Running the policy in the Brax environment for **10 episodes** / 在 Brax 环境中运行策略 **10 个 episodes**
3. Each episode runs for max **200 timesteps** / 每个 episode 最多运行 **200 个时间步**
4. Final fitness is the **mean reward** across episodes / 最终适应度是所有 episodes 的**平均奖励**
5. Returned as a **minimization problem**: `fitness = 100000 - mean_reward` / 作为**最小化问题**返回

**Note:** NaN and Inf values are replaced with `-1000` (equivalent to failing immediately).

**注意：** NaN 和 Inf 值会被替换为 `-1000`（相当于立即失败）。

## Device Selection / 设备选择

The library automatically detects and uses GPU (CUDA) if available:

库会自动检测并使用 GPU（如果可用）：

```python
# Auto-detect (default) / 自动检测（默认）
benchmark = Benchmark()

# Force CPU / 强制使用 CPU
benchmark = Benchmark(device='cpu')

# Force CUDA / 强制使用 CUDA
benchmark = Benchmark(device='cuda')
```

## API Reference / API 参考

### `Benchmark(device=None)`

Main benchmark class / 主基准测试类。

**Parameters / 参数:**
- `device` (str): Device to use ('cuda' or 'cpu', auto-detect if None) / 设备选择

**Methods / 方法:**
- `get_function(func_id)` - Returns an instance of the specified problem / 返回指定问题的实例
- `get_info(func_id)` - Returns problem metadata / 返回问题元数据
- `get_num_functions()` - Returns total number of problems (66) / 返回问题总数（66）
- `list_functions()` - Lists all available problems with details / 列出所有可用问题

### Problem Methods / 问题方法

Each problem instance supports / 每个问题实例支持：
- `info()` - Returns problem information dictionary / 返回问题信息字典
- `compute(x)` - Evaluates fitness for population `x` / 评估种群 `x` 的适应度
- `__call__(x)` - Alias for `compute(x)` / `compute(x)` 的别名

## Example Usage / 示例用法

```python
from benchmark.ne.brax import Benchmark
import numpy as np

# Initialize with auto-detected device / 使用自动检测的设备初始化
benchmark = Benchmark()

# List all available problems / 列出所有可用问题
functions = benchmark.list_functions()
for f in functions[:10]:
    print(f"ID {f['id']}: {f['env_name']} (depth={f['model_depth']}, dim={f['dimension']})")

# Get a specific problem / 获取特定问题
problem = benchmark.get_function(1)  # Ant, depth 0
info = benchmark.get_info(1)
print(f"Environment: {info['env_name']}")
print(f"Model Depth: {info['model_depth']}")
print(f"Dimension: {info['dimension']}")

# Create random population / 创建随机种群
pop_size = 100
dimension = info['dimension']
population = np.random.uniform(-0.2, 0.2, (pop_size, dimension))

# Evaluate / 评估
fitness = problem(population)
print(f"Fitness shape: {fitness.shape}")  # (100,)
print(f"Best fitness: {fitness.min()}")
```

## Citation / 引用

If you use this benchmark in your research, please cite:

如果您在研究中使用此基准测试，请引用：

```bibtex
@misc{ne_benchmark_2026,
  title={NeuroEvolution Benchmark Suite for Brax Environments},
  author={WCC-CIM Project},
  year={2026}
}
```

## License / 许可证

See project root LICENSE file / 参见项目根目录 LICENSE 文件。
