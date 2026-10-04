import torch.nn as nn

from mini_transformer.embeddings import (
    TokenEmbedding,
    PositionalEmbedding
)

from mini_transformer.attention import SelfAttention

class TinyLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        head_dim,
        sequence_length
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.attention = SelfAttention(
            embedding_dim,
            head_dim
        )

        self.positional_embedding = PositionalEmbedding(
            sequence_length,
            embedding_dim
        )

        # TODO
        # Create final projection:
        #
        # embedding/head_dim -> vocab_size
        #
        # Think about what the output of attention represents.
        self.output_layer = nn.Linear(head_dim, vocab_size)

    def forward(self, input_ids):

        # TODO
        # 1. Convert token IDs -> embeddings

        # TODO
        # 2. Pass embeddings through attention

        # TODO
        # 3. Convert attention output -> vocabulary logits

        # Expected final shape:
        #
        # (batch_size, sequence_length, vocab_size)

        embedding = self.embedding(input_ids)

        pos_embedding = self.positional_embedding(input_ids)

        combined = embedding + pos_embedding

        attention = self.attention(combined)
        
        logits = self.output_layer(attention)

        return logits