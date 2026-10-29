"""Simple tokenizers that show how different media become token sequences."""
import torch
from torch import nn
import torch.nn.functional as F


class CharacterTokenizer:
    """Character tokenizer. A tiny teaching tokenizer, not production BPE."""
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-\n"):
        self.alphabet = alphabet
        self.to_id = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.unk_id = 0

    def encode(self, text):
        return [self.to_id.get(ch.lower(), self.unk_id) for ch in text]

    def decode(self, ids):
        inverse = {v: k for k, v in self.to_id.items()}
        return "".join(inverse.get(int(i), "�") for i in ids)


class ImagePatchTokenizer(nn.Module):
    """ViT-style patch projection. Input image [B,C,H,W]; output [B,N+1,E]."""
    def __init__(self, in_channels=3, patch_size=16, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, images):
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        x = self.proj(images).flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(images.shape[0], -1, -1), x], dim=1)


def _mel_filterbank(n_fft, n_mels, sample_rate, device, dtype):
    """Triangular mel filters, shape [n_mels, n_fft//2+1], no torchaudio required."""
    def hz_to_mel(hz): return 2595.0 * torch.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mels = torch.linspace(float(hz_to_mel(torch.tensor(0.))), float(hz_to_mel(torch.tensor(sample_rate / 2))), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = torch.floor((n_fft + 1) * hz / sample_rate).long().clamp(0, n_fft // 2)
    bank = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m-1].item(), bins[m].item(), bins[m+1].item()
        if center > left:
            bank[m-1, left:center] = torch.arange(center-left, dtype=torch.float32) / (center-left)
        if right > center:
            bank[m-1, center:right] = torch.arange(right-center, 0, -1, dtype=torch.float32) / (right-center)
    return bank.to(device=device, dtype=dtype)


class AudioPatchTokenizer(nn.Module):
    """Waveform [B,S] -> log-mel [B,M,T] -> patches [B,N,E]."""
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=64,
                 time_patch=16, mel_patch=8, embed_dim=192):
        super().__init__()
        self.sample_rate, self.n_fft, self.hop_length, self.n_mels = sample_rate, n_fft, hop_length, n_mels
        self.time_patch, self.mel_patch = time_patch, mel_patch
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(mel_patch, time_patch),
                              stride=(mel_patch, time_patch))

    def log_mel(self, waveform):
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        window = torch.hann_window(self.n_fft, device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.n_fft, window=window, return_complex=True,
                          center=True).abs().pow(2)
        bank = _mel_filterbank(self.n_fft, self.n_mels, self.sample_rate, waveform.device, spec.dtype)
        return torch.log(torch.clamp(bank @ spec, min=1e-6))  # [B,M,T]

    def forward(self, waveform):
        mel = self.log_mel(waveform)
        # Pad to complete patches, preserving the last short time/frequency area.
        pad_m = (-mel.shape[1]) % self.mel_patch
        pad_t = (-mel.shape[2]) % self.time_patch
        mel = F.pad(mel, (0, pad_t, 0, pad_m))
        return self.proj(mel.unsqueeze(1)).flatten(2).transpose(1, 2)


class VideoTubeletTokenizer(nn.Module):
    """Video frames [B,T,C,H,W] -> space-time tubelet tokens [B,N,E]."""
    def __init__(self, in_channels=3, tubelet_t=2, patch_size=16, embed_dim=192):
        super().__init__()
        self.tubelet_t, self.patch_size = tubelet_t, patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=(tubelet_t, patch_size, patch_size),
                              stride=(tubelet_t, patch_size, patch_size))

    def forward(self, frames):
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")
        b, t, c, h, w = frames.shape
        if t < self.tubelet_t or t % self.tubelet_t or h % self.patch_size or w % self.patch_size:
            raise ValueError("time must be divisible by tubelet_t; height/width by patch_size")
        x = self.proj(frames.transpose(1, 2).contiguous())
        return x.flatten(2).transpose(1, 2)
