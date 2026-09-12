"""Lab 04 — nn.Module: how every PyTorch model (including our GPT) is built.

What nn.Module gives you:
  self.parameters()   -> all trainable tensors, found recursively
  named_parameters()  -> same, with names (what Adam/AdamW updates)
  forward(x)          -> the computation; __call__ wraps it with hooks
  train()/eval()      -> toggles behaviour of layers like Dropout and BatchNorm
  to(device)/to(dtype)-> moves the whole model (cpu/cuda/fp16)

We build a small 2-layer MLP "by hand" to see what a module really is.
Dropout is included to show why train() vs eval() matters.

Run:  python labs/04_nnmodule.py
"""
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """2-layer MLP: Linear -> Dropout -> ReLU -> Linear."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()                       # REQUIRED: sets up internal machinery
        self.fc1 = nn.Linear(in_dim, hidden)     # weight (hidden, in), bias (hidden,)
        self.drop = nn.Dropout(dropout)          # random zeros during TRAINING only
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim) -> (batch, out_dim)
        h = self.fc1(x)                          # (batch, hidden)
        h = self.drop(h)                         # zeroed in train(), identity in eval()
        h = self.relu(h)                         # nonlinearity: without it, 2 layers == 1
        return self.fc2(h)                       # (batch, out_dim)


def verify() -> None:
    torch.manual_seed(0)
    model = TinyMLP(in_dim=8, hidden=16, out_dim=4)

    # --- parameters: what training actually updates -------------------------
    params = list(model.parameters())
    assert len(params) == 4                      # fc1.W, fc1.b, fc2.W, fc2.b
    named = dict(model.named_parameters())
    assert set(named) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert named["fc1.weight"].shape == (16, 8)  # nn.Linear stores (out, in)!
    assert named["fc2.bias"].shape == (4,)
    total = sum(p.numel() for p in params)
    assert total == 8 * 16 + 16 + 16 * 4 + 4     # 212 params

    # all parameters require grad by default
    assert all(p.requires_grad for p in params)

    # --- forward pass: call the MODULE, not .forward() directly ----------------
    # __call__ triggers hooks/autograd bookkeeping; .forward() would skip them.
    x = torch.randn(5, 8)
    out = model(x)                               # NOT model.forward(x)
    assert out.shape == (5, 4)

    # --- train() vs eval(): Dropout behaviour -------------------------------------
    model.train()
    outs_train = [model(x) for _ in range(3)]
    assert not all(torch.equal(outs_train[0], o) for o in outs_train[1:])  # stochastic

    model.eval()
    outs_eval = [model(x) for _ in range(3)]
    assert all(torch.equal(outs_eval[0], o) for o in outs_eval[1:])        # deterministic
    # In eval mode Dropout is the identity function:
    drop = nn.Dropout(0.5)
    h = torch.ones(1000)
    drop.eval()
    assert torch.equal(drop(h), h)
    drop.train()
    masked = drop(h)                             # ~half zeros, survivors scaled by 1/(1-p)
    assert (masked == 0).float().mean().item() > 0.3

    # --- gradients flow through the module -------------------------------------
    loss = model(x).sum()
    loss.backward()
    assert named["fc1.weight"].grad is not None
    assert not torch.equal(named["fc1.weight"].grad, torch.zeros_like(named["fc1.weight"].grad))

    # --- moving/casting the whole module ------------------------------------------
    m2 = TinyMLP(4, 4, 2).to(torch.float64)
    assert next(m2.parameters()).dtype == torch.float64
    # no_grad is the standard inference pattern:
    model.eval()
    with torch.no_grad():
        pred = model(x)
    assert pred.requires_grad is False


def main() -> None:
    torch.manual_seed(0)
    model = TinyMLP(8, 16, 4, dropout=0.5)
    print(model)
    for name, p in model.named_parameters():
        print(f"  {name:<12} shape={tuple(p.shape)}  numel={p.numel()}")
    print("total parameters:", sum(p.numel() for p in model.parameters()))
    x = torch.randn(5, 8)
    print("\ninput (5, 8) -> output", tuple(model(x).shape))
    model.eval()
    with torch.no_grad():
        print("eval-mode output is deterministic:", tuple(model(x).shape))
    print("\nALL LAB-04 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
