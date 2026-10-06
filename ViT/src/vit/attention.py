import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (single big projection, split into heads by reshape).

    x: (batch, seq_len, embed_dim) -> (batch, seq_len, embed_dim)

    `mask` is optional so the same module works for:
      - ViT / encoders: mask=None (full bidirectional attention)
      - decoders:       pass a boolean mask where True = "allowed to attend"
    """

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape

        # (B, T, D) -> (B, heads, T, head_dim)
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, heads, T, T)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask True = keep. masked_fill replaces where the condition is True -> invert it (~mask)
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = self.attn_drop(torch.softmax(scores, dim=-1))

        out = weights @ v                                   # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # merge heads
        return self.proj_drop(self.W_o(out))
