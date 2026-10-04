import torch.nn as nn

from mini_transformer.embeddings import (
    TokenEmbedding,
    PositionalEmbedding
)

from mini_transformer.block import TransformerBlock
from mini_transformer.layer_norm import LayerNorm

class TinyLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        sequence_length,
        num_heads,
        num_layers,
        ff_hidden_dim=None
    ):
        super().__init__()

        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embedding_dim

        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.positional_embedding = PositionalEmbedding(
            sequence_length,
            embedding_dim
        )

        # num_layers transformer blocks, applied one after another.
        # Each has its own weights.
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])

        # One last normalization before predicting words
        self.final_norm = LayerNorm(embedding_dim)

        # embedding_dim -> vocab_size
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):

        # 1. token IDs -> token + position embeddings
        #    (batch_size, sequence_length, embedding_dim)
        embedding = self.embedding(input_ids)

        pos_embedding = self.positional_embedding(input_ids)

        x = embedding + pos_embedding

        # 2. transformer blocks (same shape in and out)
        for block in self.blocks:
            x = block(x)

        # 3. normalize, then project to vocabulary logits
        #    (batch_size, sequence_length, vocab_size)
        x = self.final_norm(x)

        logits = self.output_layer(x)

        return logits
