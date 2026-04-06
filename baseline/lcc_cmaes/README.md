# LCC-CMAES

**Learning-based Cooperative Coevolution CMA-ES**

A simplified, clean interface for testing the trained LCC-CMAES optimizer on large-scale optimization problems.

**Paper / 论文**: "Advancing CMA-ES with Learning-Based Cooperative Coevolution for Scalable Optimization", GECCO 2026

## Features / 特点

- **Simple API**: Similar to baseline optimizers, just `problem` + `options`
- **No Training Complexity**: Only inference/testing, removes all training overhead
- **Flexible Configuration**: Easy to customize parameters
- **CEC2013 LSGO Support**: Ready for benchmark problems

## Installation / 安装

```bash
# The package is already part of the project
# No additional installation needed
```

## Quick Start / 快速开始

### Method 1: Using the class directly / 方法1：直接使用类

```python
from baseline.lcc_cmaes import LCC_CMAES

# Define your problem / 定义你的问题
problem = {
    'fitness_function': lambda x: sum(x**2),  # Function to minimize / 要最小化的函数
    'ndim_problem': 1000,                      # Dimension / 维度
    'upper_boundary': 100.0,                   # Upper bound / 上界
    'lower_boundary': -100.0                   # Lower bound / 下界
}

# Configure optimizer / 配置优化器
options = {
    'model_path': 'path/to/your/model.pt',     # Trained LCC-CMAES model / 训练好的 LCC-CMAES 模型
    'max_function_evaluations': 3000000,       # Budget / 预算
    'm': 10,                                   # Number of subgroups / 子组数量
    'device': 'cuda'                           # Device / 设备
}

# Run optimization / 运行优化
optimizer = LCC_CMAES(problem, options)
results = optimizer.optimize()

# Get results / 获取结果
print(f"Best fitness: {results['best_so_far_y']}")
```

### Method 2: Using the convenience function / 方法2：使用便捷函数

```python
from baseline.lcc_cmaes import optimize_with_lcc_cmaes

results = optimize_with_lcc_cmaes(
    fitness_function=lambda x: sum(x**2),
    ndim_problem=1000,
    model_path='path/to/model.pt',
    max_function_evaluations=1000000
)
```

### Method 3: Command line / 方法3：命令行

```bash
# Test on single problem / 在单个问题上测试
python -m baseline.lcc_cmaes.run \
    --problem_id 1 \
    --model_path path/to/model.pt \
    --max_fes 1000000

# Test on all CEC2013 LSGO problems / 在所有 CEC2013 LSGO 问题上测试
python -m baseline.lcc_cmaes.run \
    --all_problems \
    --model_path path/to/model.pt
```

## Parameters / 参数

### Problem Dictionary / 问题字典

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `fitness_function` | callable | Function to minimize (must accept numpy array) | Required |
| `ndim_problem` | int | Problem dimension | 1000 |
| `upper_boundary` | float | Upper bound of search space | 100.0 |
| `lower_boundary` | float | Lower bound of search space | -100.0 |

### Options Dictionary / 选项字典

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `model_path` | str | Path to trained LCC-CMAES model (.pt file) | Required |
| `max_function_evaluations` | int | Maximum function evaluations budget | 3000000 |
| `m` | int | Number of subgroups for CC | 10 |
| `sub_popsize` | int | Population size per subgroup | 10 |
| `subFEs` | int | FEs per subspace optimization | 1000 |
| `each_question_batch_num` | int | Parallel instances | 4 |
| `seed_rng` | int | Random seed | 2024 |
| `device` | str | Device ('cuda' or 'cpu') | 'cuda' |
| `verbose` | int | Logging verbosity (0=silent) | 1 |

## Results / 结果

The `optimize()` method returns a dictionary:

```python
{
    'best_so_far_x': np.ndarray,      # Best solution found / 找到的最优解
    'best_so_far_y': float,            # Best fitness value / 最优适应度值
    'n_function_evaluations': int,     # Total FEs used / 使用的总 FEs
    'runtime': float,                  # Runtime in seconds / 运行时间（秒）
    'fitness_history': List[float]     # Fitness progression / 适应度进程
}
```

## Example on CEC2013 LSGO / CEC2013 LSGO 示例

```python
from baseline.lcc_cmaes import LCC_CMAES
from benchmark.cec2013lsgo.cec2013 import Benchmark

# Get CEC2013 problem / 获取 CEC2013 问题
bench = Benchmark()
info = bench.get_info(1)  # Problem F1

problem = {
    'fitness_function': bench.get_function(1),
    'ndim_problem': info['dimension'],
    'upper_boundary': info['upper'],
    'lower_boundary': info['lower']
}

options = {
    'model_path': 'path/to/model.pt',
    'max_function_evaluations': 3000000
}

optimizer = LCC_CMAES(problem, options)
results = optimizer.optimize()
```

## Comparison with Original run.py / 与原始 run.py 的对比

| Aspect | Original (run.py) | New (lcc_cmaes) |
|--------|------------------|---------------------|
| Lines of code | ~400+ | ~50 (user code) |
| Training support | Yes | No (inference only) |
| Configuration | Complex options dict | Simple problem/options |
| TensorBoard logging | Built-in | Optional |
| Parallel environments | Manual | Automatic |
| Data saving | Excel-based | Simple dict return |

## File Structure / 文件结构

```
baseline/lcc_cmaes/
├── __init__.py          # Package initialization / 包初始化
├── lcc_cmaes.py         # Main optimizer class / 主优化器类
├── run.py               # Command-line runner / 命令行运行器
├── README.md            # This file / 本文件
├── GECCO 2026 LCC.pdf   # Paper / 论文
│
├── env/                 # Environment and agent code / 环境和智能体代码
│   ├── agent/           # PPO agent implementation / PPO 智能体实现
│   ├── optimizer/       # CMA-ES optimizer / CMA-ES 优化器
│   └── parallel/        # Parallel execution / 并行执行
│
├── utils/               # Utility functions (self-contained) / 工具函数（自包含）
│   ├── options.py       # Configuration options / 配置选项
│   ├── utils.py         # Helper functions / 辅助函数
│   ├── logger.py        # Logging utilities / 日志工具
│   └── make_dataset.py  # Dataset utilities / 数据集工具
│
└── model/               # Place trained models here / 放置训练好的模型
    └── epoch-9.pt       # Example: copy your trained model here / 示例：将训练好的模型复制到这里
```

## Model Setup / 模型设置

Place your trained model in the `model/` directory:
将训练好的模型放在 `model/` 目录中：

```bash
# Copy your trained model / 复制训练好的模型
cp repository/save_dir/ppo_model/model_CEC2013LSGO_3.0e+06_10_10_0.0006_1000.0_20241121T093031/epoch-9.pt baseline/lcc_cmaes/model/

# Or use a symbolic link / 或者使用符号链接
ln -s /full/path/to/epoch-9.pt baseline/lcc_cmaes/model/epoch-9.pt
```

Then update the `model_path` in your code:
然后更新代码中的 `model_path`：

```python
options = {
    'model_path': 'baseline/lcc_cmaes/model/epoch-9.pt',
    ...
}
```

## Notes / 注意事项

1. The model must be trained first using the original `run.py`
2. This is for inference/testing only - no training functionality
3. Uses the same underlying environment (cmaes) and agent (PPO)
4. Results may vary slightly due to randomness and hardware differences

## Citation / 引用

If you use this code, please cite our GECCO 2026 paper:

```bibtex
@inproceedings{lcc_cmaes_gecco2026,
  title={Advancing CMA-ES with Learning-Based Cooperative Coevolution for Scalable Optimization},
  booktitle={GECCO},
  year={2026}
}
```
