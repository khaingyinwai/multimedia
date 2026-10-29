"""Transformer blocks written for learning, not speed."""
import math
import torch
from torch import nn
from .attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """Layer normalization over the final embedding dimension."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal position information; input/output [B,L,E]."""
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        if dim > 1:
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError("sequence longer than max_len")
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block, input/output [B,L,E]."""
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = LayerNorm(dim)
        self.ff = FeedForward(dim, dim * mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
