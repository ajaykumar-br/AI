import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(input_ids, target_ids, batch_size, shuffle=True):
    """
    Wrap input_ids / target_ids into a DataLoader that yields mini-batches.

    input_ids, target_ids:
        lists (from create_training_data) or tensors of shape
        (num_sequences, sequence_length)

    Each batch:
        (batch_size, sequence_length), (batch_size, sequence_length)
    """

    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    target_ids = torch.as_tensor(target_ids, dtype=torch.long)

    dataset = TensorDataset(input_ids, target_ids)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
