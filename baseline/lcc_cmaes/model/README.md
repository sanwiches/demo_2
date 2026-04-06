# LCC-CMAES Trained Model

This directory contains the trained LCC-CMAES model for inference/testing.

## Model Information / 模型信息

- **File**: `epoch-9.pt`
- **Training Configuration**:
  - Max FEs: 3.0e+06
  - Number of subgroups (m): 10
  - Subgroup population: 10
  - SubFEs: 1000.0
  - Learning rate: 0.0006
  - Training date: 2024-11-21

## Usage / 使用方法

### Option 1: Let the code use the default path / 选项1：让代码使用默认路径

```python
from baseline.lcc_cmaes import LCC_CMAES

# No need to specify model_path / 无需指定 model_path
optimizer = LCC_CMAES(problem, options)
```

### Option 2: Specify the model path explicitly / 选项2：显式指定模型路径

```python
from baseline.lcc_cmaes import LCC_CMAES

options = {
    'model_path': 'baseline/lcc_cmaes/model/epoch-9.pt',
    ...
}
```

## Adding New Models / 添加新模型

To add a new trained model:

1. Copy the `.pt` file to this directory
   ```bash
   cp /path/to/new-model.pt baseline/lcc_cmaes/model/
   ```

2. Update the code to use the new model
   ```python
   options = {
       'model_path': 'baseline/lcc_cmaes/model/new-model.pt',
       ...
   }
   ```
