"""
网络通用模块 / Network Common Module.

提供 Actor 和 Critic 网络共用的基础组件。
Provides common components shared by Actor and Critic networks.

主要组件 / Main Components:
    - SkipConnection: 跳跃连接模块 / Skip connection module
    - MLP_for_actor/critic: 多层感知机 / Multi-layer perceptron
    - Normalization: 归一化层 / Normalization layer
    - MultiHeadAttention: 多头注意力机制 / Multi-head attention mechanism
    - ValueDecoder: Critic 价值解码器 / Critic value decoder
    - EmbeddingNet: 嵌入网络 / Embedding network
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)


# ==========================================
# 基础模块 / Basic Modules
# ==========================================

class SkipConnection(nn.Module):
    """
    跳跃连接模块 / Skip Connection Module.

    实现残差连接，输入与模块输出相加。
    Implements residual connection by adding input to module output.

    Example:
        >>> skip = SkipConnection(nn.Linear(10, 10))
        >>> x = torch.randn(5, 10)
        >>> output = skip(x)  # output = x + Linear(x)
    """

    def __init__(self, module: nn.Module) -> None:
        """
        初始化跳跃连接 / Initialize skip connection.

        Args:
            module: 要包裹的模块 / Module to wrap
        """
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            input: 输入张量 / Input tensor

        Returns:
            输入与模块输出之和 / Sum of input and module output
        """
        return input + self.module(input)


class Normalization(nn.Module):
    """
    归一化模块 / Normalization Module.

    支持批次归一化、实例归一化和层归一化。
    Supports batch, instance, and layer normalization.

    Attributes:
        normalization (str): 归一化类型 / Normalization type
        normalizer: 归一化层 / Normalization layer
    """

    def __init__(self, embed_dim: int, normalization: str = 'batch') -> None:
        """
        初始化归一化层 / Initialize normalization layer.

        Args:
            embed_dim: 嵌入维度 / Embedding dimension
            normalization: 归一化类型 / Normalization type ('batch', 'instance', 'layer')
        """
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if self.normalization != 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            input: 输入张量 / Input tensor

        Returns:
            归一化后的张量 / Normalized tensor
        """
        if self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(input.var((1, 2)).view(-1, 1, 1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return input


# ==========================================
# MLP 模块 / MLP Modules
# ==========================================

class MLP_for_critic(nn.Module):
    """
    Critic 专用的多层感知机 / MLP specialized for Critic.

    三层全连接网络，用于价值函数估计。
    Three-layer fully connected network for value function estimation.

    Attributes:
        fc1 (nn.Linear): 第一全连接层 / First linear layer
        fc2 (nn.Linear): 第二全连接层 / Second linear layer
        fc3 (nn.Linear): 第三全连接层 / Third linear layer
        dropout (nn.Dropout): Dropout 层 / Dropout layer
        ReLU (nn.ReLU): ReLU 激活函数 / ReLU activation
    """

    def __init__(
        self,
        input_dim: int = 128,
        feed_forward_dim: int = 64,
        embedding_dim: int = 64,
        output_dim: int = 1,
        p: float = 0.001
    ) -> None:
        """
        初始化 MLP / Initialize MLP.

        Args:
            input_dim: 输入维度 / Input dimension
            feed_forward_dim: 前馈层维度 / Feed-forward layer dimension
            embedding_dim: 嵌入维度 / Embedding dimension
            output_dim: 输出维度 / Output dimension
            p: Dropout 概率 / Dropout probability
        """
        super(MLP_for_critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(p=p)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, in_: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            in_: 输入张量 / Input tensor

        Returns:
            输出张量 / Output tensor
        """
        result = self.fc1(in_)
        result = self.ReLU(result)
        result = self.fc2(result)
        result = self.ReLU(result)
        result = self.fc3(result).squeeze(-1)
        result = self.ReLU(result)
        return result


class MLP_for_actor(nn.Module):
    """
    Actor 专用的多层感知机 / MLP specialized for Actor.

    三层全连接网络，用于策略输出。
    Three-layer fully connected network for policy output.

    Attributes:
        fc1 (nn.Linear): 第一全连接层 / First linear layer
        fc2 (nn.Linear): 第二全连接层 / Second linear layer
        fc3 (nn.Linear): 第三全连接层 / Third linear layer
        dropout (nn.Dropout): Dropout 层 / Dropout layer
        ReLU (nn.ReLU): ReLU 激活函数 / ReLU activation
    """

    def __init__(
        self,
        input_dim: int = 58,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 1,
        p: float = 0.001
    ) -> None:
        """
        初始化 MLP / Initialize MLP.

        Args:
            input_dim: 输入维度 / Input dimension
            embedding_dim: 嵌入维度 / Embedding dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension
            output_dim: 输出维度 / Output dimension
            p: Dropout 概率 / Dropout probability
        """
        super(MLP_for_actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=p)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, in_: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            in_: 输入张量 / Input tensor

        Returns:
            输出张量 / Output tensor
        """
        result = self.fc1(in_)
        result = self.fc2(result)
        result = self.fc3(result).squeeze(-1)
        return result


# ==========================================
# 注意力机制模块 / Attention Modules
# ==========================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 / Multi-Head Attention Mechanism.

    实现标准的多头自注意力机制。
    Implements standard multi-head self-attention mechanism.

    Attributes:
        n_heads (int): 注意力头数 / Number of attention heads
        input_dim (int): 输入维度 / Input dimension
        embed_dim (int): 嵌入维度 / Embedding dimension
        val_dim (int): 值维度 / Value dimension
        key_dim (int): 键维度 / Key dimension
        W_query (Parameter): 查询权重矩阵 / Query weight matrix
        W_key (Parameter): 键权重矩阵 / Key weight matrix
        W_val (Parameter): 值权重矩阵 / Value weight matrix
        W_out (Parameter): 输出权重矩阵 / Output weight matrix
    """

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: Optional[int] = None,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None
    ) -> None:
        """
        初始化多头注意力 / Initialize multi-head attention.

        Args:
            n_heads: 注意力头数 / Number of attention heads
            input_dim: 输入维度 / Input dimension
            embed_dim: 嵌入维度 / Embedding dimension (default: None)
            val_dim: 值维度 / Value dimension (default: None)
            key_dim: 键维度 / Key dimension (default: None)
        """
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / (key_dim ** 0.5)

        self.W_query = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.randn(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.randn(n_heads, val_dim, embed_dim))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            q: 查询张量 / Query tensor, shape (batch_size, graph_size, input_dim)

        Returns:
            输出张量 / Output tensor, shape (batch_size, n_query, embed_dim)
        """
        h = q  # 自注意力 / Self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # 计算查询、键、值 / Compute queries, keys, values
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # 计算兼容性分数 / Compute compatibility scores
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attn = F.softmax(compatibility, dim=-1)

        # 计算注意力输出 / Compute attention output
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class MultiHeadCompat(nn.Module):
    """
    多头兼容性层 / Multi-Head Compatibility Layer.

    计算查询和键之间的兼容性分数（未归一化）。
    Computes compatibility scores between queries and keys (unnormalized).

    Attributes:
        n_heads (int): 注意力头数 / Number of attention heads
        input_dim (int): 输入维度 / Input dimension
        key_dim (int): 键维度 / Key dimension
        norm_factor (float): 归一化因子 / Normalization factor
        W_query (Parameter): 查询权重矩阵 / Query weight matrix
        W_key (Parameter): 键权重矩阵 / Key weight matrix
    """

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: Optional[int] = None,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None
    ) -> None:
        """
        初始化多头兼容性层 / Initialize multi-head compatibility layer.

        Args:
            n_heads: 注意力头数 / Number of attention heads
            input_dim: 输入维度 / Input dimension
            embed_dim: 嵌入维度 / Embedding dimension (unused, for compatibility)
            val_dim: 值维度 / Value dimension (unused, for compatibility)
            key_dim: 键维度 / Key dimension (default: None)
        """
        super(MultiHeadCompat, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / (key_dim ** 0.5)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            q: 查询张量 / Query tensor
            h: 键值张量（默认为 q）/ Key-value tensor (defaults to q)
            mask: 掩码张量（未实现）/ Mask tensor (not implemented)

        Returns:
            兼容性分数 / Compatibility scores
        """
        if h is None:
            h = q

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        compatibility = torch.matmul(Q, K.transpose(2, 3))

        return self.norm_factor * compatibility


# ==========================================
# 编码器和解码器模块 / Encoder and Decoder Modules
# ==========================================

class MultiHeadAttentionLayerforCritic(nn.Sequential):
    """
    Critic 专用的多头注意力层 / Multi-head attention layer for Critic.

    包含注意力、前馈网络、残差连接和归一化的完整层。
    Complete layer with attention, feed-forward, residual and normalization.

    Attributes:
        attention sub-layer with skip connection
        normalization layer
        feed-forward sub-layer with skip connection
        normalization layer
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str = 'layer'
    ) -> None:
        """
        初始化注意力层 / Initialize attention layer.

        Args:
            n_heads: 注意力头数 / Number of attention heads
            embed_dim: 嵌入维度 / Embedding dimension
            feed_forward_hidden: 前馈隐藏层维度 / Feed-forward hidden dimension
            normalization: 归一化类型 / Normalization type
        """
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class MultiHeadAttentionsubLayer(nn.Module):
    """
    多头注意力子层 / Multi-Head Attention Sub-Layer.

    包含注意力和层归一化的子层。
    Sub-layer with attention and layer normalization.

    Attributes:
        MHA (MultiHeadAttention): 多头注意力 / Multi-head attention
        Norm (Normalization): 归一化层 / Normalization layer
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str = 'layer'
    ) -> None:
        """
        初始化注意力子层 / Initialize attention sub-layer.

        Args:
            n_heads: 注意力头数 / Number of attention heads
            embed_dim: 嵌入维度 / Embedding dimension
            feed_forward_hidden: 前馈隐藏层维度 / Feed-forward hidden dimension
            normalization: 归一化类型 / Normalization type
        """
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            input: 输入张量 / Input tensor

        Returns:
            输出张量 / Output tensor
        """
        out = self.MHA(input)
        return self.Norm(out + input)


class FFandNormsubLayer(nn.Module):
    """
    前馈和归一化子层 / Feed-Forward and Normalization Sub-Layer.

    包含前馈网络和层归一化的子层。
    Sub-layer with feed-forward network and layer normalization.

    Attributes:
        FF (nn.Module): 前馈网络 / Feed-forward network
        Norm (Normalization): 归一化层 / Normalization layer
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str = 'layer'
    ) -> None:
        """
        初始化前馈子层 / Initialize feed-forward sub-layer.

        Args:
            n_heads: 注意力头数（未使用）/ Number of attention heads (unused)
            embed_dim: 嵌入维度 / Embedding dimension
            feed_forward_hidden: 前馈隐藏层维度 / Feed-forward hidden dimension
            normalization: 归一化类型 / Normalization type
        """
        super(FFandNormsubLayer, self).__init__()

        self.FF = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            input: 输入张量 / Input tensor

        Returns:
            输出张量 / Output tensor
        """
        out = self.FF(input)
        return self.Norm(out + input)


class MultiHeadEncoder(nn.Module):
    """
    多头编码器 / Multi-Head Encoder.

    组合注意力和前馈子层的编码器。
    Encoder combining attention and feed-forward sub-layers.

    Attributes:
        MHA_sublayer (MultiHeadAttentionsubLayer): 注意力子层 / Attention sub-layer
        FFandNorm_sublayer (FFandNormsubLayer): 前馈子层 / Feed-forward sub-layer
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str = 'layer'
    ) -> None:
        """
        初始化编码器 / Initialize encoder.

        Args:
            n_heads: 注意力头数 / Number of attention heads
            embed_dim: 嵌入维度 / Embedding dimension
            feed_forward_hidden: 前馈隐藏层维度 / Feed-forward hidden dimension
            normalization: 归一化类型 / Normalization type
        """
        super(MultiHeadEncoder, self).__init__()
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            input: 输入张量 / Input tensor

        Returns:
            输出张量 / Output tensor
        """
        out = self.MHA_sublayer(input)
        return self.FFandNorm_sublayer(out)


class ValueDecoder(nn.Module):
    """
    价值解码器 / Value Decoder.

    将嵌入特征映射到状态价值。
    Maps embedded features to state values.

    Attributes:
        MLP (MLP_for_critic): 多层感知机 / Multi-layer perceptron
    """

    def __init__(self, embed_dim: int, input_dim: int) -> None:
        """
        初始化价值解码器 / Initialize value decoder.

        Args:
            embed_dim: 嵌入维度 / Embedding dimension
            input_dim: 输入维度 / Input dimension
        """
        super(ValueDecoder, self).__init__()
        self.hidden_dim = embed_dim
        self.embedding_dim = embed_dim
        self.MLP = MLP_for_critic(input_dim=input_dim)

    def forward(self, h_em: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            h_em: 嵌入特征张量 / Embedded feature tensor, shape (bs, ps, feature_dim)

        Returns:
            价值张量 / Value tensor
        """
        mean_pooling = h_em
        value = self.MLP(mean_pooling)
        return value


class EmbeddingNet(nn.Module):
    """
    嵌入网络 / Embedding Network.

    将输入映射到嵌入空间。
    Maps inputs to embedding space.

    Attributes:
        node_dim (int): 节点维度 / Node dimension
        embedding_dim (int): 嵌入维度 / Embedding dimension
        embedder (nn.Linear): 线性嵌入层 / Linear embedding layer
    """

    def __init__(self, node_dim: int, embedding_dim: int) -> None:
        """
        初始化嵌入网络 / Initialize embedding network.

        Args:
            node_dim: 输入节点维度 / Input node dimension
            embedding_dim: 输出嵌入维度 / Output embedding dimension
        """
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward propagation.

        Args:
            x: 输入张量 / Input tensor

        Returns:
            嵌入张量 / Embedded tensor
        """
        h_em = self.embedder(x)
        return h_em
