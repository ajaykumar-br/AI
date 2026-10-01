import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        # TODO
        # Create the three projections:
        #
        # W_Q
        # W_K
        # W_V
        #
        # Each should transform:
        # embedding_dim -> head_dim

        self.head_dim = head_dim

        self.W_Q = nn.Linear(embedding_dim, head_dim)
        self.W_K = nn.Linear(embedding_dim, head_dim)
        self.W_V = nn.Linear(embedding_dim, head_dim)

    def forward(self, x):

        """
        x:
            (batch_size, sequence_length, embedding_dim)

        Q:
            (batch_size, sequence_length, head_dim)

        K:
            (batch_size, sequence_length, head_dim)

        V:
            (batch_size, sequence_length, head_dim)

        """

        # TODO
        # 1. Create Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # TODO
        # 2. Calculate attention scores
        #
        #        Q @ K^T
        #
        # Remember:
        # K needs to be transposed across the
        # sequence dimensions.

        attention_scores = Q @ K.transpose(-2, -1)

        # TODO
        # 3. Scale the scores
        #
        # divide by sqrt(head_dim)
        scaled_attention = attention_scores / math.sqrt(self.head_dim)

        # TODO
        # 4. Apply causal mask
        #
        # A token should NOT be allowed to
        # attend to future tokens.

        T = x.shape[1]
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        masked_attention = scaled_attention.masked_fill(~mask, float('-inf'))

        # TODO
        # 5. Softmax over the correct dimension
        attention_weights = F.softmax(masked_attention, dim = -1)
        # print(attention_weights[0])
        # TODO
        # 6. Multiply attention weights by V
        updated_rep = attention_weights @ V

        # TODO
        # Return attention output

        return updated_rep