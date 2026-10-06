import torch
import torch.nn as nn

from vit.blocks import TransformerEncoderBlock
from vit.config import ViTConfig
from vit.patch_embedding import PatchEmbedding


class ViTEncoder(nn.Module):
    """Backbone: image -> token features. Task-agnostic, reusable for detection later.

    (B, C, H, W) -> (B, 1 + num_patches, embed_dim)
    Token 0 is [CLS]; tokens 1.. are patch tokens (row-major over the patch grid).
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + cfg.num_patches, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout)
            for _ in range(cfg.depth)
        ])
        # Pre-LN keeps the residual stream un-normalized, so a final LayerNorm is required.
        self.norm = nn.LayerNorm(cfg.embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                                   # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)           # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                            # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VisionTransformer(nn.Module):
    """Classifier = ViTEncoder + linear head on the [CLS] token. (B,C,H,W) -> (B, num_classes)."""

    def __init__(self, cfg: ViTConfig | None = None):
        super().__init__()
        cfg = cfg or ViTConfig()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)          # (B, N+1, D)
        return self.head(tokens[:, 0])    # CLS token only -> (B, num_classes)
