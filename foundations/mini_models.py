"""Small example models. These are architecture demos, not pretrained recognizers."""
import torch
from torch import nn
from .attention import causal_mask
from .blocks import EncoderBlock, PositionalEncoding
from .tokenizers import ImagePatchTokenizer, AudioPatchTokenizer


class MiniViT(nn.Module):
    """Tiny image classifier: image [B,3,H,W] -> class logits [B,num_classes]."""
    def __init__(self, num_classes=10, image_size=32, patch_size=8, dim=64, heads=4, depth=2):
        super().__init__()
        self.tokens = ImagePatchTokenizer(3, patch_size, dim)
        self.pos = PositionalEncoding(dim, max_len=(image_size // patch_size) ** 2 + 1)
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_ratio=2) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, image):
        x = self.pos(self.tokens(image))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


class MiniAudioTransformer(nn.Module):
    """Tiny audio classifier: waveform [B,S] -> class logits [B,num_classes]."""
    def __init__(self, num_classes=10, dim=64, heads=4, depth=2):
        super().__init__()
        self.tokens = AudioPatchTokenizer(n_mels=32, time_patch=16, mel_patch=8, embed_dim=dim)
        self.pos = PositionalEncoding(dim, max_len=4096)
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_ratio=2) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, waveform):
        x = self.pos(self.tokens(waveform))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x.mean(dim=1)))
