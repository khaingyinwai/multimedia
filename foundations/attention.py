"""Attention primitives. Tensor shapes are documented in each function."""
import math
import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    """q:[B,H,Q,D], k:[B,H,K,D], v:[B,H,K,Dv], mask broadcastable to [B,H,Q,K]."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # [B,H,Q,K]
    if mask is not None:
        # Boolean True means a position is allowed; False is blocked.
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = torch.nn.functional.dropout(weights, p=dropout_p, training=training)
    return weights @ v, weights  # [B,H,Q,Dv], [B,H,Q,K]


def padding_mask(valid_tokens):
    """valid_tokens:[B,K] bool; returns [B,1,1,K] broadcast mask."""
    return valid_tokens[:, None, None, :].bool()


def causal_mask(length, device=None):
    """Returns [1,1,L,L], where query position i may see key positions <= i."""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()[None, None, :, :]


class MultiHeadAttention(nn.Module):
    """Self-attention. Input/output shape [B,L,E]."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x, mask=None, return_weights=False):
        b, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.reshape(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        y, weights = scaled_dot_product_attention(split(q), split(k), split(v), mask,
                                                  self.dropout, self.training)
        y = y.transpose(1, 2).contiguous().reshape(b, length, self.embed_dim)
        y = self.out(y)
        return (y, weights) if return_weights else y
