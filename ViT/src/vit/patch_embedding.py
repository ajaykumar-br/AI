import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Image -> sequence of patch tokens.

    (B, C, H, W) -> (B, num_patches, embed_dim)

    A Conv2d with kernel_size == stride == patch_size is exactly "cut into
    non-overlapping patches, flatten each, apply one shared Linear" in a single op.
    Each output position sees one patch and nothing else (no overlap).

    Returns only the patch tokens. The [CLS] token and positional embeddings are
    added in the encoder, so this module can be reused as-is for dense tasks
    (e.g. detection/segmentation) that don't want a CLS token.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                      # (B, D, grid, grid)
        return x.flatten(2).transpose(1, 2)   # (B, D, N) -> (B, N, D)
