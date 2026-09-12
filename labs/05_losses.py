"""Lab 05 — Loss functions.

MSE  (mean squared error): for regression (predict a number).
  MSE = mean((pred - target)^2)

Cross Entropy: for classification (predict one of N classes).
  CE(pred_logits, target_class) = -log(softmax(logits)[target_class])

WHY CROSS ENTROPY FOR NEXT-TOKEN PREDICTION?
  Generating text is, at each step, a classification problem over the vocabulary:
  "which token comes next?" out of V candidates.
  If the model assigns probability p_t to the actual next token t, the loss is
  -log(p_t). This is 0 when the model is certain (p_t = 1) and blows up as
  p_t -> 0 — an excellent, smooth training signal.
  Note: CE in PyTorch takes RAW logits, not probabilities: it applies
  log_softmax internally in a numerically stable way.

Perplexity = exp(mean CE). The single most important LLM evaluation metric.

Run:  python labs/05_losses.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify() -> None:
    # ================= MSE =================
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 2.5])
    mse = nn.MSELoss()(pred, target)
    manual = ((pred - target) ** 2).mean()       # mean((.5)^2,(.5)^2,(.5)^2)
    assert torch.isclose(mse, manual)
    assert abs(mse.item() - 0.25) < 1e-6

    # ================= Cross Entropy, by hand =================
    # 2 examples, 4-token vocabulary, true next tokens: [2, 0]
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.0],
                           [2.0, 1.0, 0.0, 1.0]])
    targets = torch.tensor([2, 0])

    # Hand calculation:
    # CE = -log(exp(logit_t) / sum(exp(logits)))
    #    = -(logit_t - logsumexp(logits))
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # log softmax
    ce_manual = -log_probs[torch.arange(2), targets].mean()

    ce_torch = nn.CrossEntropyLoss()(logits, targets)
    assert torch.isclose(ce_manual, ce_torch, atol=1e-6)

    # sanity: CE equals -log(p_true). For the first example p_true = e^3/(e^1+e^2+e^3+e^0)
    p1 = torch.softmax(logits[0], dim=0)[2]
    assert torch.isclose(-torch.log(p1), -log_probs[0, 2], atol=1e-6)

    # ================= CE punishes wrong confident answers =================
    good = torch.tensor([[0.0, 0.0, 10.0, 0.0]])   # very confident on class 2
    bad = torch.tensor([[10.0, 0.0, 0.0, 0.0]])    # very confident on wrong class
    t = torch.tensor([2])
    loss_good = nn.CrossEntropyLoss()(good, t)
    loss_bad = nn.CrossEntropyLoss()(bad, t)
    assert loss_good < 1e-3                        # near zero: right & certain
    assert loss_bad > 9.0                          # huge: wrong & certain

    # ================= gradient of CE is elegant =================
    # For a single example:  d(CE)/d(logits) = softmax(logits) - onehot(target).
    # With the default reduction='mean' over N examples, the whole gradient is
    # additionally divided by N (mean rule for sums: the 1/N factor passes through).
    lg = logits.clone().requires_grad_(True)
    nn.CrossEntropyLoss()(lg, targets).backward()
    expected = torch.softmax(logits, dim=1)
    expected[torch.arange(2), targets] -= 1.0
    assert torch.allclose(lg.grad, expected / 2, atol=1e-6)   # N = 2 examples

    # ================= Perplexity =================
    ppl = torch.exp(ce_torch)                      # e^CE
    assert ppl.item() > 1.0                        # CE>0 always (unless p=1)
    assert torch.allclose(ppl, torch.exp(ce_manual))

    # ================= ignore_index: pad tokens don't count =================
    logits3 = torch.randn(1, 5, 10)                # (batch, seq, vocab)
    tgt3 = torch.tensor([[1, 2, -100, -100, 3]])   # -100 = padded positions
    ce_ign = F.cross_entropy(logits3.reshape(-1, 10), tgt3.reshape(-1), ignore_index=-100)
    assert torch.isfinite(ce_ign)


def main() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.0],
                           [2.0, 1.0, 0.0, 1.0]])
    targets = torch.tensor([2, 0])
    probs = torch.softmax(logits, dim=1)
    print("vocabulary probabilities:")
    print(" ", probs.numpy().round(3))
    print("true next tokens:", targets.tolist())
    print("p(true) per example:", probs[torch.arange(2), targets].numpy().round(3))
    ce = nn.CrossEntropyLoss()(logits, targets)
    print(f"\ncross entropy      = {ce.item():.4f}")
    print(f"perplexity = e^CE  = {torch.exp(ce).item():.4f}")
    print("\n(Interpretation: on average the model is as uncertain")
    print(" as if choosing uniformly among ~3.2 tokens.)")
    print("\nALL LAB-05 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
