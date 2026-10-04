import torch.nn as nn

from mini_transformer.multi_head_attention import MultiHeadAttention
from mini_transformer.feed_forward import FeedForward
from mini_transformer.layer_norm import LayerNorm

# ============================================================
# TRANSFORMER BLOCK (pre-LayerNorm, as in GPT-2)
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, ff_hidden_dim):
        super().__init__()

        self.norm_1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        self.norm_2 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)

    def forward(self, x):

        """
        x:
            (batch_size, sequence_length, embedding_dim)

        Output has the same shape, so blocks can be stacked.
        """

        # 1. Tokens gather information from earlier tokens.
        #    Residual: keep x, add what attention found.
        x = x + self.attention(self.norm_1(x))

        # 2. Each token processes what it gathered.
        #    Residual again.
        x = x + self.feed_forward(self.norm_2(x))

        return x
