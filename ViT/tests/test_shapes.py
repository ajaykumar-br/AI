import torch

from vit.attention import MultiHeadAttention
from vit.blocks import TransformerEncoderBlock
from vit.config import ViTConfig
from vit.model import ViTEncoder, VisionTransformer
from vit.patch_embedding import PatchEmbedding


def test_attention_shape():
    x = torch.randn(2, 65, 128)
    assert MultiHeadAttention(128, 8)(x).shape == (2, 65, 128)


def test_causal_mask_blocks_future():
    """Changing a FUTURE token must not change earlier outputs when a causal mask is used."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(32, 4).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.tril(torch.ones(6, 6)).bool()
    y1 = attn(x, mask)
    x2 = x.clone(); x2[:, 5] += 10.0          # perturb the last token only
    y2 = attn(x2, mask)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)   # earlier positions unchanged
    assert not torch.allclose(y1[:, 5], y2[:, 5])            # the perturbed position did change
    # ...and without the mask, earlier positions DO change (leak)
    assert not torch.allclose(attn(x)[:, :5], attn(x2)[:, :5])


def test_block_is_shape_preserving():
    x = torch.randn(2, 65, 128)
    assert TransformerEncoderBlock(128, 8)(x).shape == x.shape


def test_patch_embedding_shape():
    assert PatchEmbedding(32, 4, 3, 128)(torch.randn(2, 3, 32, 32)).shape == (2, 64, 128)


def test_patch_embedding_matches_manual_unfold_linear():
    """Conv2d(k=s=patch) must equal: cut patches -> flatten -> shared Linear."""
    torch.manual_seed(0)
    pe = PatchEmbedding(8, 4, 3, 16)
    x = torch.randn(1, 3, 8, 8)
    patches = x.unfold(2, 4, 4).unfold(3, 4, 4)                 # (1, 3, 2, 2, 4, 4)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 4, 3 * 4 * 4)   # (1, N, C*p*p)
    w = pe.proj.weight.reshape(16, -1)                          # (D, C*p*p)
    manual = patches @ w.T + pe.proj.bias
    assert torch.allclose(pe(x), manual, atol=1e-5)


def test_encoder_and_classifier_shapes():
    cfg = ViTConfig()
    x = torch.randn(2, 3, 32, 32)
    assert ViTEncoder(cfg)(x).shape == (2, 65, cfg.embed_dim)
    assert VisionTransformer(cfg)(x).shape == (2, 10)


def test_backward_runs():
    m = VisionTransformer()
    loss = m(torch.randn(2, 3, 32, 32)).sum()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
