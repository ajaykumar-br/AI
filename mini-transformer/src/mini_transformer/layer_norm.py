import torch
import torch.nn as nn

# ============================================================
# LAYER NORMALIZATION
# ============================================================

class LayerNorm(nn.Module):

    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()

        # Learnable scale (gamma) and shift (beta), one per feature.
        # Start as the identity: scale 1, shift 0.
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):

        """
        x:
            (batch_size, sequence_length, embedding_dim)

        Each token's vector is normalized on its own,
        across its embedding_dim features.
        """

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        return self.gamma * x_normalized + self.beta
