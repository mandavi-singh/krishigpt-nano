"""Lab 03 — Autograd: automatic differentiation.

Why we need it: training a GPT means computing d(loss)/d(param) for MILLIONS
of parameters. Autograd records every operation you do (the computational
graph) and applies the chain rule backwards through it to get all gradients
with a single backward() call.

Key ideas:
  requires_grad=True      -> "track operations on this tensor"
  leaf tensors            -> tensors you created (inputs/parameters); gradients
                             are ACCUMULATED into .grad
  backward()              -> walks the graph backwards applying the chain rule
  torch.no_grad()         -> turn off tracking (used at inference, saves memory)
  zero gradients          -> autograd ACCUMULATES; you must zero .grad between
                             training steps (that's what optimizer.zero_grad() does)

Math: for y = f(x), backward() computes dy/dx.
  Compound example: y = (2x + 1)^2 at x = 3
    dy/dx = 2*(2x + 1)*2          (chain rule)
          = 4*(2x + 1) = 4*7 = 28

Run:  python labs/03_autograd.py
"""
import torch


def verify() -> None:
    # --- manual derivative vs autograd -------------------------------------
    # y = (2x + 1)^2
    x = torch.tensor(3.0, requires_grad=True)
    y = (2 * x + 1) ** 2
    y.backward()
    expected = 4 * (2 * 3.0 + 1)          # chain rule by hand = 28
    assert abs(x.grad.item() - expected) < 1e-6, x.grad
    assert x.grad.item() == 28.0

    # y is now consumed; calling backward again would fail. New graph:
    x2 = torch.tensor(3.0, requires_grad=True)
    y2 = x2 ** 2
    y2.backward()
    assert abs(x2.grad.item() - 6.0) < 1e-6      # d(x^2)/dx = 2x = 6 at x=3

    # --- gradient is the gradient of a SCALAR loss -----------------------------
    # Mean-squared error over 3 points; gradient matches manual derivative.
    # L(w) = mean((w*x - y)^2),  dL/dw = mean(2*(w*x - y)*x)
    xs = torch.tensor([1.0, 2.0, 3.0])
    ys = torch.tensor([2.0, 4.0, 6.0])         # y = 2x
    w = torch.tensor(1.0, requires_grad=True)
    loss = ((w * xs - ys) ** 2).mean()
    loss.backward()
    manual = (2 * (w.detach() * xs - ys) * xs).mean()
    assert torch.isclose(w.grad, manual, atol=1e-6)
    assert w.grad.item() < 0                    # loss decreases as w increases

    # --- accumulation: gradients ADD UP unless zeroed ---------------------------
    x3 = torch.tensor(2.0, requires_grad=True)
    (x3 ** 2).backward()                        # grad = 4
    (x3 ** 2).backward()                        # grad = 4 + 4 = 8  (accumulated!)
    assert abs(x3.grad.item() - 8.0) < 1e-6
    x3.grad.zero_()                             # manual zeroing
    (x3 ** 2).backward()
    assert abs(x3.grad.item() - 4.0) < 1e-6

    # --- leaf / non-leaf -------------------------------------------------------
    # Only leaf tensors (the ones with requires_grad that YOU made) keep .grad
    a = torch.tensor(3.0, requires_grad=True)   # leaf
    b = a * 2                                    # non-leaf: result of an op
    b.retain_grad()                              # opt in to keeping its grad
    (b ** 2).backward()                          # d(b^2)/db = 2b = 12
    assert abs(a.grad.item() - 24.0) < 1e-6      # db/da = 2 -> 12 * 2 = 24 (chain)
    assert abs(b.grad.item() - 12.0) < 1e-6

    # --- no_grad: inference mode -------------------------------------------------
    with torch.no_grad():
        z = torch.tensor(5.0, requires_grad=True) * 2
        assert not z.requires_grad               # tracking disabled
    # detaching cuts a tensor out of the graph explicitly:
    p = torch.tensor(4.0, requires_grad=True)
    q = p * 3
    assert q.detach().requires_grad is False

    # --- vector-valued functions need an upstream gradient ---------------------
    # backward() needs a scalar. For a vector output, pass the weighting vector.
    v = torch.arange(4.0, requires_grad=True)
    out = v * 2                                  # out = [0,2,4,6]
    out.backward(torch.ones(4))                  # d(out_i)/d(v_i) = 2 each
    assert torch.equal(v.grad, torch.full((4,), 2.0))


def main() -> None:
    x = torch.tensor(3.0, requires_grad=True)
    y = (2 * x + 1) ** 2
    y.backward()
    print("y = (2x+1)^2 = ", y.item())
    print("manual dy/dx  = 4*(2x+1) = 28.0")
    print("autograd dy/dx =", x.grad.item())

    xs = torch.tensor([1.0, 2.0, 3.0])
    ys = torch.tensor([2.0, 4.0, 6.0])
    w = torch.tensor(0.5, requires_grad=True)
    loss = ((w * xs - ys) ** 2).mean()
    loss.backward()
    print(f"\nMSE loss w=0.5: {loss.item():.4f}, dL/dw = {w.grad.item():.4f}")
    print("(negative gradient => increase w reduces loss)")
    print("\nALL LAB-03 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
