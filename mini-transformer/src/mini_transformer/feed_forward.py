import torch.nn as nn

# ============================================================
# FEED-FORWARD (MLP) SUBLAYER
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        # Expand -> non-linearity -> project back.
        # hidden_dim is usually 4 * embedding_dim.
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):

        """
        x:
            (batch_size, sequence_length, embedding_dim)

        Applied to every position independently:
        tokens do not talk to each other here.
        That already happened in attention.
        """

        return self.net(x)
