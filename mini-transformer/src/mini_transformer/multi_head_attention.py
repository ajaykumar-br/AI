import torch
import torch.nn as nn

from mini_transformer.attention import SelfAttention

# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible"
                f" by num_heads ({num_heads})"
            )

        # Split the embedding between the heads:
        # 64 dims / 4 heads = 16 dims per head.
        head_dim = embedding_dim // num_heads

        # Each head is an independent SelfAttention with its own
        # W_Q, W_K, W_V, so each can learn a different pattern.
        self.heads = nn.ModuleList([
            SelfAttention(embedding_dim, head_dim)
            for _ in range(num_heads)
        ])

        # W_O: mixes what the heads found back into one vector.
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        """
        x:
            (batch_size, sequence_length, embedding_dim)

        each head:
            (batch_size, sequence_length, head_dim)

        concatenated:
            (batch_size, sequence_length, num_heads * head_dim)
            = (batch_size, sequence_length, embedding_dim)
        """

        head_outputs = [head(x) for head in self.heads]

        concatenated = torch.cat(head_outputs, dim=-1)

        return self.output_projection(concatenated)
