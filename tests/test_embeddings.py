"""Tests for the from-scratch TokenEmbedding layer."""
import math

import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding

V, D = 64, 16


def make_ids(b: int = 2, t: int = 5) -> torch.Tensor:
    return torch.randint(0, V, (b, t))


def test_shape() -> None:
    emb = TokenEmbedding(V, D)
    ids = make_ids()
    assert emb(ids).shape == (2, 5, D)
    assert emb(make_ids(1, 1)).shape == (1, 1, D)          # single-token batch


def test_lookup_is_row_gather() -> None:
    torch.manual_seed(1)
    emb = TokenEmbedding(V, D)
    ids = make_ids()
    out = emb(ids)
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            assert torch.equal(out[b, t], emb.weight[ids[b, t]])


def test_manual_onehot_view_identical() -> None:
    torch.manual_seed(2)
    emb = TokenEmbedding(V, D)
    ids = make_ids()
    assert torch.allclose(emb.forward_manual(ids), emb(ids), atol=1e-6)


def test_equal_to_nn_embedding() -> None:
    torch.manual_seed(3)
    ids = make_ids()                                       # draw ids ONCE, reuse both
    emb = TokenEmbedding(V, D)
    ref = nn.Embedding(V, D)
    ref.weight.data.copy_(emb.weight.data)
    assert torch.equal(ref(ids), emb(ids))


def test_gradient_only_on_used_rows() -> None:
    emb = TokenEmbedding(V, D)
    ids = torch.tensor([[3, 7], [3, 9]])
    emb(ids).sum().backward()
    for i in range(V):
        grad = emb.weight.grad[i]
        if i in (3, 7, 9):
            assert torch.any(grad != 0), f"row {i} should have grad"
        else:
            assert torch.all(grad == 0), f"row {i} should have ZERO grad"


def test_grad_counts_repetitions() -> None:
    """Row k used twice gets twice the gradient of a singly-used row."""
    emb = TokenEmbedding(V, D)
    ids = torch.tensor([[5, 5], [5, 6]])                   # id 5 appears 3x
    emb(ids).sum().backward()
    # d(sum)/d(W[row]) = number of occurrences * ones(D)
    assert torch.allclose(emb.weight.grad[5], torch.full((D,), 3.0))
    assert torch.allclose(emb.weight.grad[6], torch.full((D,), 1.0))


def test_init_scale() -> None:
    torch.manual_seed(4)
    emb = TokenEmbedding(1000, 256)                         # larger table for stats
    std = emb.weight.data.std().item()
    assert abs(std - 1 / math.sqrt(256)) < 0.02            # init ~ N(0, 1/sqrt(D))
