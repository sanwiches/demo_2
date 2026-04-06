"""
Inference Agent for LCC-CMAES
LCC-CMAES 的推理智能体

Simplified agent for inference mode, containing only the Actor network
and necessary methods for loading pretrained models.
简化的推理模式智能体，仅包含 Actor 网络和加载预训练模型的必要方法。

Usage / 使用方法:
    >>> from env.agent.inference import InferenceAgent
    >>> from utils.options import get_options
    >>> opts = get_options()
    >>> agent = InferenceAgent(opts)
    >>> agent.load('model/epoch-9.pt')
    >>> action, _, _ = agent.actor(state)
"""

# =============================================================================
# Imports / 导入
# =============================================================================
import os
import torch
from env.agent.network.actor_network import Actor
from utils.utils import get_inner_model, torch_load_cpu


# =============================================================================
# Inference Agent / 推理智能体
# =============================================================================

class InferenceAgent:
    """
    Simplified agent for inference mode.
    简化的推理模式智能体。

    This agent contains only the Actor network and methods needed for
    loading pretrained models and performing inference.
    此智能体仅包含 Actor 网络和加载预训练模型及执行推理所需的方法。

    Attributes:
        opts: Configuration options / 配置选项
        actor: Policy network (Actor) / 策略网络 (Actor)
        device: Torch device (cuda/cpu) / PyTorch 设备 (cuda/cpu)

    Example:
        >>> agent = InferenceAgent(opts)
        >>> agent.load('model/epoch-9.pt')
        >>> agent.eval()
        >>> action, logprob, state_value = agent.actor(state)
    """

    def __init__(self, opts):
        """
        Initialize the inference agent.
        初始化推理智能体。

        Args:
            opts: Configuration options containing device and network settings
                  / 包含设备和网络设置的配置选项
        """
        self.opts = opts
        self.device = opts.device if hasattr(opts, 'device') else 'cuda'

        # Create Actor network / 创建 Actor 网络
        self.actor = Actor()

        # Move to device if CUDA is available / 如果 CUDA 可用，移动到设备
        if self.device == 'cuda' and torch.cuda.is_available():
            self.actor.to(self.device)

    def load(self, load_path):
        """
        Load pretrained Actor model from checkpoint.
        从检查点加载预训练的 Actor 模型。

        Args:
            load_path (str): Path to the checkpoint file (.pt file)
                           / 检查点文件路径 (.pt 文件)

        Raises:
            AssertionError: If load_path is None / 如果 load_path 为 None
            FileNotFoundError: If load_path does not exist / 如果 load_path 不存在
        """
        assert load_path is not None, "load_path cannot be None"
        assert os.path.exists(load_path), f"Model file not found: {load_path}"

        # Load checkpoint data / 加载检查点数据
        load_data = torch_load_cpu(load_path)

        # Load Actor weights / 加载 Actor 权重
        model_actor = get_inner_model(self.actor)
        actor_state_dict = load_data.get('actor', {})

        if actor_state_dict:
            model_actor.load_state_dict({
                **model_actor.state_dict(),
                **actor_state_dict
            })
        else:
            raise ValueError(f"No 'actor' weights found in {load_path}")

        # Determine target device / 确定目标设备
        target_device = torch.device(self.device)
        if self.device == 'cuda' and not torch.cuda.is_available():
            target_device = torch.device('cpu')

        # Move model to target device / 将模型移动到目标设备
        self.actor.to(target_device)

        # IMPORTANT: Update Actor's internal device reference
        # 重要：更新 Actor 的内部设备引用
        # The Actor class stores self._device which is used in forward()
        # Actor 类存储了 self._device，在 forward() 中使用
        if hasattr(self.actor, '_device'):
            self.actor._device = target_device

        print(' [*] Loading data from {}'.format(load_path))

    def eval(self):
        """
        Set the Actor network to evaluation mode.
        将 Actor 网络设置为评估模式。

        This disables dropout and batch normalization updates.
        这会禁用 dropout 和批归一化更新。
        """
        self.actor.eval()

    def to(self, device):
        """
        Move the Actor network to a specific device.
        将 Actor 网络移动到指定设备。

        Args:
            device: Target device ('cuda' or 'cpu') / 目标设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.actor.to(device)
