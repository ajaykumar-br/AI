import torch.nn as nn

def calculate_loss(logits, targets):

    """
    logits:
        (batch_size, sequence_length, vocab_size)

    targets:
        (batch_size, sequence_length)
    """

    # TODO
    # Cross entropy expects the vocabulary dimension
    # in the appropriate position.
    #
    # You may need to reshape/flatten logits and targets.

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)

    # print(f'{logits.shape}\n {targets.shape}')

    return loss