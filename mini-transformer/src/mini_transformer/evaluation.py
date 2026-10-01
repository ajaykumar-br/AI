import math

import torch
import torch.nn.functional as F


def count_correct(logits, targets, k=1):
    """
    Number of positions where the target is in the model's top-k predictions.

    logits:
        (batch_size, sequence_length, vocab_size)

    targets:
        (batch_size, sequence_length)
    """

    top_k = logits.topk(k, dim=-1).indices
    return (top_k == targets.unsqueeze(-1)).any(dim=-1).sum().item()


@torch.no_grad()
def evaluate(model, loader, device, top_k=5):
    """
    Average loss, perplexity, top-1 accuracy and top-k accuracy over a loader.
    """

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top_k = 0
    total_tokens = 0

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)

        total_loss += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="sum"
        ).item()
        total_correct += count_correct(logits, target_ids, k=1)
        total_top_k += count_correct(logits, target_ids, k=top_k)
        total_tokens += target_ids.numel()

    model.train(was_training)

    loss = total_loss / total_tokens

    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "accuracy": total_correct / total_tokens,
        f"top_{top_k}_accuracy": total_top_k / total_tokens,
    }
