import torch
from foundations.attention import scaled_dot_product_attention, padding_mask, causal_mask, MultiHeadAttention
from foundations.blocks import LayerNorm, PositionalEncoding, EncoderBlock
from foundations.tokenizers import CharacterTokenizer, ImagePatchTokenizer, AudioPatchTokenizer, VideoTubeletTokenizer
from foundations.mini_models import MiniViT, MiniAudioTransformer


def test_attention_shapes_and_weights():
    q = torch.randn(2, 3, 4, 8); k = torch.randn(2, 3, 5, 8); v = torch.randn(2, 3, 5, 6)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 4, 6)
    assert weights.shape == (2, 3, 4, 5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 3, 4), atol=1e-5)


def test_masks():
    m = causal_mask(4)
    assert m.shape == (1, 1, 4, 4) and m[0, 0, 3, 0] and not m[0, 0, 0, 3]
    p = padding_mask(torch.tensor([[True, True, False]]))
    assert p.shape == (1, 1, 1, 3) and not p[..., 2].item()


def test_attention_permutation_equivariance_without_positions():
    torch.manual_seed(1); attn = MultiHeadAttention(16, 4).eval(); x = torch.randn(2, 5, 16)
    perm = torch.tensor([2, 0, 4, 1, 3])
    y1 = attn(x[:, perm]); y2 = attn(x)[:, perm]
    assert torch.allclose(y1, y2, atol=1e-5)


def test_blocks_shapes():
    x = torch.randn(2, 7, 16)
    assert LayerNorm(16)(x).shape == x.shape
    assert PositionalEncoding(16)(x).shape == x.shape
    assert EncoderBlock(16, 4)(x).shape == x.shape


def test_character_tokenizer():
    tok = CharacterTokenizer(); ids = tok.encode('hello!')
    assert len(ids) == 6 and tok.decode(ids) == 'hello!'


def test_image_patch_tokens_include_cls():
    tok = ImagePatchTokenizer(patch_size=8, embed_dim=32)
    assert tok(torch.randn(2, 3, 32, 32)).shape == (2, 17, 32)


def test_audio_and_video_token_shapes():
    audio = AudioPatchTokenizer(n_fft=256, hop_length=128, n_mels=32, time_patch=8, mel_patch=8, embed_dim=16)
    assert audio(torch.randn(2, 4000)).ndim == 3
    video = VideoTubeletTokenizer(tubelet_t=2, patch_size=8, embed_dim=16)
    assert video(torch.randn(2, 4, 3, 16, 16)).shape == (2, 8, 16)


def test_mini_model_logits():
    assert MiniViT()(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert MiniAudioTransformer()(torch.randn(1, 4000)).shape == (1, 10)
