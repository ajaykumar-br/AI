import torch.nn as nn

from mini_transformer.embeddings import (
    TokenEmbedding,
    PositionalEmbedding
)

from mini_transformer.attention import SelfAttention

# ============================================================
# THE FIRST MODEL: one attention head, nothing else
# ============================================================
#
# Kept so checkpoints/tiny_lm_attention_only.pt can still be loaded.
# No residuals, LayerNorm or feed-forward. TinyLanguageModel
# in model.py is the current architecture.
#
# The attribute names (embedding, attention, ...) must stay the
# same, or the saved weights will not match.

class AttentionOnlyLanguageModel(nn.Module):

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

        self.output_layer = nn.Linear(head_dim, vocab_size)

    def forward(self, input_ids):

        embedding = self.embedding(input_ids)

        pos_embedding = self.positional_embedding(input_ids)

        combined = embedding + pos_embedding

        attention = self.attention(combined)

        logits = self.output_layer(attention)

        return logits
