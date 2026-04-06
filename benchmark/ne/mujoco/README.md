# MuJoCo NeuroEvolution Benchmark Library / MuJoCo 神经进化基准测试库

A benchmark suite for NeuroEvolution (NE) algorithms using MuJoCo reinforcement learning environments through mujoco_playground.

基于 mujoco_playground 的神经进化算法基准测试套件。

## Background / 背景

NeuroEvolution combines evolutionary algorithms with neural networks, optimizing neural network weights through evolutionary strategies. This library provides a benchmark suite to evaluate NE algorithms on MuJoCo control tasks.

神经进化将进化算法与神经网络结合，通过进化策略来优化神经网络权重。本库提供了一个基准测试套件，用于在 MuJoCo 控制任务上评估神经进化算法。

The benchmark uses **mujoco_playground** - a JAX-based MuJoCo interface for training and evaluating reinforcement learning agents.

本基准测试使用 **mujoco_playground** - 基于 JAX 的 MuJoCo 接口。

## Installation Requirements / 安装要求

```bash
# Basic installation / 基础安装
pip install numpy jax mujoco-playground

# For CUDA support / CUDA 支持 (NVIDIA GPUs)
pip install jax==0.6.2 jaxlib==0.6.2 jax_cuda12_plugin==0.6.2 jax_cuda12_pjrt==0.6.2 --extra-index-url https://pypi.nvidia.com
```

## Quick Start / 快速开始

```python
from benchmark.ne.mujoco import Benchmark

# Initialize the benchmark / 初始化基准测试
benchmark = Benchmark()

# Get a specific problem / 获取特定问题
problem = benchmark.get_function(1)  # AcrobotSwingup

# Get problem information / 获取问题信息
info = benchmark.get_info(1)
print(f"Dimension: {info['dimension']}")
print(f"Environment: {info['env_name']}")

# Evaluate a population / 评估一组解
import numpy as np
population = np.random.uniform(-1.0, 1.0, size=(10, info['dimension']))
fitness = problem(population)
```

## Problem Suite / 问题集

The benchmark suite consists of **54 MuJoCo environments** organized into 4 categories:

基准测试套件包含 **54 个 MuJoCo 环境**，分为 4 个类别：

### Classic Control (9 problems) / 经典控制（9 个问题）

| ID | Name / 名称 | Dimension / 维度 | Description / 描述 |
|----|-------------|------------------|---------------------|
| 1 | AcrobotSwingup | 65 | Two-link robot arm swing-up / 双连杆机械臂摆起 |
| 2 | AcrobotSwingupSparse | 65 | Sparse reward variant / 稀疏奖励变体 |
| 3 | BallInCup | 90 | Ball-in-cup game / 杯中球游戏 |
| 4 | CartpoleBalance | 57 | Cartpole balancing / 倒立摆平衡 |
| 5 | CartpoleBalanceSparse | 57 | Sparse reward variant / 稀疏奖励变体 |
| 6 | CartpoleSwingup | 57 | Cartpole swing-up / 倒立摆摆起 |
| 7 | CartpoleSwingupSparse | 57 | Sparse reward variant / 稀疏奖励变体 |
| 8 | PendulumSwingup | 41 | Pendulum swing-up / 单摆摆起 |
| 9 | PointMass | 58 | Point mass navigation / 质点导航 |

### Locomotion (11 problems) / 运动控制（11 个问题）

| ID | Name / 名称 | Dimension / 维度 | Description / 描述 |
|----|-------------|------------------|---------------------|
| 10 | SwimmerSwimmer6 | 253 | 6-link swimming robot / 6连杆游泳机器人 |
| 11 | CheetahRun | 198 | Half-cheetah running / 半猎豹跑动 |
| 12 | FishSwim | 245 | Fish swimming / 鱼类游动 |
| 13 | HopperHop | 164 | Hopper hopping / 单脚跳跳跃 |
| 14 | HopperStand | 164 | Hopper standing / 单脚跳站立 |
| 15 | WalkerRun | 254 | Bipedal walker running / 双足机器人跑动 |
| 16 | WalkerStand | 254 | Bipedal walker standing / 双足机器人站立 |
| 17 | WalkerWalk | 254 | Bipedal walker walking / 双足机器人行走 |
| 18 | HumanoidStand | 733 | Humanoid standing / 人形机器人站立 |
| 19 | HumanoidWalk | 733 | Humanoid walking / 人形机器人行走 |
| 20 | HumanoidRun | 733 | Humanoid running / 人形机器人跑动 |

### Manipulation (15 problems) / 操控（15 个问题）

| ID | Name / 名称 | Dimension / 维度 | Description / 描述 |
|----|-------------|------------------|---------------------|
| 21 | FingerSpin | 98 | Finger spin task / 手指旋转任务 |
| 22 | FingerTurnEasy | 122 | Easy finger turn / 简单手指转动 |
| 23 | FingerTurnHard | 122 | Hard finger turn / 困难手指转动 |
| 24 | ReacherEasy | 74 | Easy reaching task / 简单到达任务 |
| 25 | ReacherHard | 74 | Hard reaching task / 困难到达任务 |
| 26 | PandaPickCube | 608 | Panda pick and place / Panda 抓取放置 |
| 27 | PandaPickCubeOrientation | 608 | Panda pick with orientation / Panda 定向抓取 |
| 28 | PandaPickCubeCartesian | 595 | Panda Cartesian pick / Panda 笛卡尔抓取 |
| 29 | PandaOpenCabinet | 520 | Panda cabinet opening / Panda 打开柜门 |
| 30 | PandaRobotiqPushCube | 455 | Panda push cube / Panda 推方块 |
| 31 | AlohaHandOver | 798 | ALOHA handover / ALOHA 传递任务 |
| 32 | AlohaSinglePegInsertion | 790 | ALOHA peg insertion / ALOHA 销插入 |
| 33 | LeapCubeReorient | 608 | Leap hand cube reorientation / Leap 手爪方块重定向 |
| 34 | LeapCubeRotateZAxis | 408 | Leap hand Z-axis rotation / Leap 手爪 Z轴旋转 |
| 35 | AeroCubeRotateZAxis | 183 | Aero hand Z-axis rotation / Aero 手爪 Z轴旋转 |

### Humanoid/Robot (19 problems) | 人形/机器人（19 个问题）

| ID | Name / 名称 | Dimension / 维度 | Description / 描述 |
|----|-------------|------------------|---------------------|
| 36 | ApolloJoystickFlatTerrain | 1192 | Apollo on flat terrain / Apollo 平坦地形 |
| 37 | BarkourJoystick | 3836 | Barkour quadruped / Barkour 四足机器人 |
| 38 | BerkeleyHumanoidJoystickFlatTerrain | 532 | Berkeley humanoid flat / Berkeley 人形平坦地形 |
| 39 | BerkeleyHumanoidJoystickRoughTerrain | 532 | Berkeley humanoid rough / Berkeley 人形粗糙地形 |
| 40 | G1JoystickFlatTerrain | 1093 | G1 robot flat terrain / G1 机器人平坦地形 |
| 41 | G1JoystickRoughTerrain | 1093 | G1 robot rough terrain / G1 机器人粗糙地形 |
| 42 | Go1JoystickFlatTerrain | 500 | Go1 flat terrain / Go1 平坦地形 |
| 43 | Go1JoystickRoughTerrain | 500 | Go1 rough terrain / Go1 粗糙地形 |
| 44 | Go1Getup | 452 | Go1 get up / Go1 起立 |
| 45 | Go1Handstand | 476 | Go1 handstand / Go1 倒立 |
| 46 | Go1Footstand | 476 | Go1 footstand / Go1 单脚站立 |
| 47 | H1InplaceGaitTracking | 1667 | H1 gait tracking / H1 步态跟踪 |
| 48 | H1JoystickGaitTracking | 1083 | H1 joystick gait / H1 摇杆步态 |
| 49 | Op3Joystick | 1364 | OP3 joystick / OP3 摇杆控制 |
| 50 | SpotFlatTerrainJoystick | 764 | Spot flat terrain / Spot 平坦地形 |
| 51 | SpotJoystickGaitTracking | 668 | Spot gait tracking / Spot 步态跟踪 |
| 52 | SpotGetup | 356 | Spot get up / Spot 起立 |
| 53 | T1JoystickFlatTerrain | 895 | T1 flat terrain / T1 平坦地形 |
| 54 | T1JoystickRoughTerrain | 895 | T1 rough terrain / T1 粗糙地形 |

## Neural Network Architecture / 神经网络架构

All MuJoCo problems use a **2-layer MLP** (Multi-Layer Perceptron) policy network:

所有 MuJoCo 问题使用 **2 层 MLP**（多层感知机）策略网络：

- **Input / 输入:** Environment observation dimension / 环境观测维度
- **Hidden Layer / 隐藏层:** 256 neurons + Tanh activation / 256 个神经元 + Tanh 激活
- **Output / 输出:** Action dimension + Tanh / 动作维度 + Tanh

**Parameter Count Formula / 参数数量公式:**
```
params = (obs_dim * 256) + 256 + (256 * action_dim) + action_dim
```

## Search Space / 搜索空间

- **All MuJoCo Problems:** [-1.0, 1.0] / [-1.0, 1.0]

## Evaluation Protocol / 评估协议

Each solution is evaluated by:

每个解通过以下方式评估：

1. Encoding weights into the neural network policy / 将权重编码到神经网络策略
2. Running the policy for **3 episodes** / 运行策略 **3 个 episodes**
3. Each episode runs for max **100 timesteps** / 每个 episode 最多运行 **100 个时间步**
4. Final fitness is the **mean reward** across episodes / 最终适应度是所有 episodes 的**平均奖励**
5. Returned as a **minimization problem**: `fitness = -mean_reward` / 作为**最小化问题**返回

## API Reference / API 参考

### `Benchmark(random_seed=42, device=None)`

Main benchmark class / 主基准测试类。

**Parameters / 参数:**
- `random_seed` (int): Random seed for reproducibility / 随机种子
- `device` (str): Device to use ('cuda' or 'cpu', auto-detect if None) / 设备选择

**Methods / 方法:**
- `get_function(func_id)` - Returns an instance of the specified problem / 返回指定问题的实例
- `get_info(func_id)` - Returns problem metadata / 返回问题元数据
- `get_num_functions()` - Returns total number of problems (54) / 返回问题总数（54）
- `list_functions()` - Lists all available problems / 列出所有可用问题

### Problem Methods / 问题方法

Each problem instance supports / 每个问题实例支持：
- `info()` - Returns problem information dictionary / 返回问题信息字典
- `compute(x)` - Evaluates fitness for population `x` / 评估种群 `x` 的适应度
- `__call__(x)` - Alias for `compute(x)` / `compute(x)` 的别名

## Example Usage / 示例用法

```python
from benchmark.ne.mujoco import Benchmark
import numpy as np

# Initialize with auto-detected device / 使用自动检测的设备初始化
benchmark = Benchmark(random_seed=42)

# List all available problems / 列出所有可用问题
functions = benchmark.list_functions()
for f in functions[:5]:
    print(f"ID {f['id']}: {f['env_name']} (dim={f['dimension']})")

# Get a specific problem / 获取特定问题
problem = benchmark.get_function(1)  # AcrobotSwingup
info = benchmark.get_info(1)
print(f"Environment: {info['env_name']}")
print(f"Dimension: {info['dimension']}")
print(f"Category: {info.get('category', 'N/A')}")

# Create random population / 创建随机种群
pop_size = 10
dimension = info['dimension']
population = np.random.uniform(-1.0, 1.0, (pop_size, dimension))

# Evaluate / 评估
fitness = problem(population)
print(f"Fitness shape: {fitness.shape}")  # (10,)
print(f"Fitness values: {fitness}")
```

## Device Selection / 设备选择

The benchmark supports both CPU and CUDA (GPU) devices:

基准测试支持 CPU 和 CUDA (GPU) 设备：

```python
# Auto-detect device / 自动检测设备
benchmark = Benchmark()

# Force CPU / 强制使用 CPU
benchmark = Benchmark(device='cpu')

# Force CUDA / 强制使用 CUDA
benchmark = Benchmark(device='cuda')
```

**Note:** CUDA support requires JAX CUDA plugins. See installation requirements above.

**注意：** CUDA 支持需要 JAX CUDA 插件。参见上方安装要求。

## Design Consistency / 设计一致性

This benchmark is designed to be consistent with `mujoco_playground`:

本基准测试设计与 `mujoco_playground` 保持一致：

- Uses `registry.load()` for environment loading / 使用 `registry.load()` 加载环境
- Fixed evaluation parameters (max_episode_length=100, num_episodes=3) / 固定的评估参数
- 2-layer MLP policy network with 256 hidden neurons / 2 层 MLP 策略网络，256 个隐藏神经元
- JAX-based implementation for acceleration / 基于 JAX 的实现以加速
- Support for both CPU and CUDA devices / 支持 CPU 和 CUDA 设备

## Citation / 引用

If you use this benchmark in your research, please cite:

如果您在研究中使用此基准测试，请引用：

```bibtex
@misc{mujoco_ne_benchmark_2026,
  title={MuJoCo NeuroEvolution Benchmark Suite},
  author={WCC-CIM Project},
  year={2026}
}
```

## License / 许可证

See project root LICENSE file / 参见项目根目录 LICENSE 文件。
