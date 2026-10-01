import torch
import torch.nn as nn

# ============================================================
# 5. TOKEN EMBEDDING
# ============================================================

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):

        # TODO
        # input_ids shape:
        #     (batch_size, sequence_length)
        #
        # output shape:
        #     (batch_size, sequence_length, embedding_dim)

        return self.embedding_layer(input_ids)

# ============================================================
# 6. POSITIONAL EMBEDDING
# ============================================================

class PositionalEmbedding(nn.Module):

    def __init__(self, sequence_length, embedding_dim):
        super().__init__()

        # TODO:
        # Create a learnable embedding for each position
        # Shape should effectively be:
        # (sequence_length, embedding_dim)
        self.position = nn.Embedding(sequence_length, embedding_dim)

    def forward(self, input_ids):
        # TODO:
        # input_ids shape: (batch_size, sequence_length)
        #
        # Create position IDs:
        # [0, 1, 2, ..., sequence_length - 1]
        #
        # Look up their embeddings
        #
        # Return positional embeddings
        sequence_length = input_ids.shape[1]
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device
        )
        position_embeddings = self.position(position_ids)
        return position_embeddings