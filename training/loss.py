"""Loss helpers for next-token prediction.

The dataloader already emits (input, target) pairs where target is the input
window shifted by one; the loss is plain CrossEntropyLoss over the vocabulary
axis:

    CE(logits (B,T,V), targets (B,T)) = mean of -log p(target | prefix)

Perplexity = exp(mean CE): the model's effective branching factor -- as if it
were choosing uniformly among exp(CE) tokens at every step.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def next_token_ce(logits: torch.Tensor, targets: torch.Tensor,
                  ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy of logits against shifted targets.

    Args:
        logits: (B, T, V)
        targets: (B, T) long, -100 positions are excluded from the average.
    """
    b, t, v = logits.shape
    return F.cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t),
                           ignore_index=ignore_index)


def perplexity(ce_loss: float | torch.Tensor) -> float:
    """exp(loss), guarded for large losses (avoid overflow in reports)."""
    val = float(ce_loss)
    return math.exp(min(val, 100.0))
