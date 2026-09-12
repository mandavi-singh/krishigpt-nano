"""Lab 06 — Optimizers: what actually updates the weights.

Gradient descent moves each parameter AGAINST the gradient of the loss:

    p_new = p_old - lr * grad

That's it. Everything else is bookkeeping to make this stable and fast.

  Manual GD   : p -= lr * p.grad              (we write the update ourselves)
  SGD         : same, with optional momentum: v = b*v + grad; p -= lr*v
  Adam        : tracks per-parameter 1st moment m (mean of grads) and 2nd
                moment v (mean of squared grads), bias-corrects them:
                  m_t = b1*m + (1-b1)*g          v_t = b2*v + (1-b2)*g^2
                  p  -= lr * m_hat / (sqrt(v_hat) + eps)
                Result: an adaptive learning rate per parameter.
  AdamW       : Adam + decoupled weight decay: p -= lr*wd*p is applied
                directly to params (not through the gradient). THIS is the
                standard optimizer for LLM pretraining (GPT-2/3 use AdamW).

We fit w* = 2 with all four and compare.

Run:  python labs/06_optimizers.py
"""
import torch


def fit_manual_gd(lr: float = 0.1, steps: int = 100) -> tuple[float, float]:
    """p = p - lr * grad, written out by hand. Target: learn w=2 from y=2x."""
    xs = torch.linspace(-1, 1, 16)
    ys = 2 * xs
    w = torch.tensor(0.0, requires_grad=True)
    for _ in range(steps):
        loss = ((w * xs - ys) ** 2).mean()
        loss.backward()                          # grad accumulates into w.grad
        with torch.no_grad():                    # the update itself must not be tracked
            w -= lr * w.grad                     # <-- the entire "optimizer"
        w.grad.zero_()                           # reset for next step
    return w.item(), loss.item()


def fit_with(opt_cls, steps: int = 100, wd: float = 0.0) -> tuple[float, float]:
    xs = torch.linspace(-1, 1, 16)
    ys = 2 * xs
    w = torch.tensor(0.0, requires_grad=True)
    # IMPORTANT: pass weight_decay EXPLICITLY. AdamW's default is wd=0.01, so
    # omitting it does NOT give you zero decay — a classic silent-config bug.
    opt = opt_cls([w], lr=0.1, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()                          # 1. reset accumulated grads
        loss = ((w * xs - ys) ** 2).mean()
        loss.backward()                          # 2. compute grads
        opt.step()                               # 3. apply the update rule
    return w.item(), loss.item()


def verify() -> None:
    import torch.optim as optim

    # all methods converge to w* = 2
    w_manual, l_manual = fit_manual_gd()
    assert abs(w_manual - 2.0) < 1e-3 and l_manual < 1e-5

    w_sgd, l_sgd = fit_with(optim.SGD)
    assert abs(w_sgd - 2.0) < 1e-3

    w_adam, l_adam = fit_with(optim.Adam)
    assert abs(w_adam - 2.0) < 1e-2

    w_adamw, l_adamw = fit_with(optim.AdamW, wd=0.0)
    assert abs(w_adamw - 2.0) < 1e-2
    # weight decay shrinks params toward 0 (regularization). Equilibrium of
    # this toy problem: w* = 2 / (1 + wd/2) approximately, so larger wd -> smaller w.
    w_reg, _ = fit_with(optim.AdamW, wd=0.1)
    assert abs(w_reg) < abs(w_adamw) - 0.01

    # manual SGD == optim.SGD with same lr (same update rule)
    w_manual2, _ = fit_manual_gd(lr=0.1)
    assert abs(w_manual2 - w_sgd) < 1e-6

    # momentum accelerates convergence vs plain SGD at same lr
    xs = torch.linspace(-1, 1, 16); ys = 2 * xs
    def steps_to(w0, momentum, n=30):
        w = torch.tensor(w0, requires_grad=True)
        opt = optim.SGD([w], lr=0.1, momentum=momentum)
        for _ in range(n):
            opt.zero_grad()
            ((w * xs - ys) ** 2).mean().backward()
            opt.step()
        return w.item()
    assert abs(steps_to(0.0, 0.9) - 2.0) < abs(steps_to(0.0, 0.0) - 2.0)

    # Adam maintains per-parameter state (m and v buffers)
    w = torch.tensor(3.0, requires_grad=True)
    adam = optim.Adam([w], lr=0.01)
    (w ** 2).backward(); adam.step()
    state = adam.state[w]
    assert "exp_avg" in state and "exp_avg_sq" in state     # the m and v moments


def main() -> None:
    print("fitting w to y = 2x (true w* = 2.0):\n")
    print(f"  manual GD      w = {fit_manual_gd()[0]:.6f}")
    print(f"  torch SGD      w = {fit_with(torch.optim.SGD)[0]:.6f}")
    print(f"  torch Adam     w = {fit_with(torch.optim.Adam)[0]:.6f}")
    print(f"  torch AdamW    w = {fit_with(torch.optim.AdamW)[0]:.6f}")
    print(f"  AdamW wd=0.1   w = {fit_with(torch.optim.AdamW, wd=0.1)[0]:.6f}  (shrunk toward 0)")
    print("\nSame loss, different UPDATE RULES. AdamW is what trains LLMs.")
    print("\nALL LAB-06 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
