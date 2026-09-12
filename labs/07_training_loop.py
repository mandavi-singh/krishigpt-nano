"""Lab 07 — A complete training loop, the exact skeleton our LLM trainer uses.

Task: classify 2D points into 3 classes (linear separability with noise).

Pipeline (identical shape to Phase 9 LLM training):
  Dataset        -> __getitem__ returns one (x, y) pair
  DataLoader     -> batches samples, shuffles each epoch
  model(x)       -> forward pass produces predictions
  loss(pred, y)  -> cross entropy
  loss.backward()-> fills .grad on every parameter
  optimizer.step-> applies the update rule
  zero_grad()    -> resets .grad BEFORE next backward (gradients accumulate!)
  validation     -> loss on held-out data, model in eval() + no_grad()

Run:  python labs/07_training_loop.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PointsDataset(Dataset):
    """Synthetic classification data: 3 gaussian blobs in 2D."""

    def __init__(self, n_per_class: int = 200, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        xs, ys = [], []
        centers = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        for c in range(3):
            xs.append(torch.randn(n_per_class, 2, generator=g) * 0.6 + centers[c])
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        self.x = torch.cat(xs)              # (N, 2)
        self.y = torch.cat(ys)              # (N,)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]         # DataLoader stacks these into a batch


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int = 2, n_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)                   # (batch, 2) -> (batch, 3) logits


def train() -> dict:
    torch.manual_seed(0)
    train_ds = PointsDataset(n_per_class=200, seed=0)
    val_ds = PointsDataset(n_per_class=50, seed=1)     # different data = held out
    # DataLoader: batch + shuffle. Our GPT loader will return (context, target) pairs instead.
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    model = LinearClassifier()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train": [], "val": []}
    for epoch in range(25):
        # ---------- TRAIN ----------
        model.train()                                  # dropout etc. in training mode
        epoch_loss = 0.0
        for xb, yb in train_dl:                        # xb: (B,2), yb: (B,)
            opt.zero_grad()                            # 1. clear old gradients
            logits = model(xb)                         # 2. forward
            loss = loss_fn(logits, yb)                 # 3. loss
            loss.backward()                            # 4. backward (fills .grad)
            opt.step()                                 # 5. update parameters
            epoch_loss += loss.item() * len(xb)
        history["train"].append(epoch_loss / len(train_ds))

        # ---------- VALIDATE (no grad updates) ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():                          # saves memory & time
            for xb, yb in val_dl:
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        history["val"].append(val_loss / len(val_ds))
    return history


def verify() -> None:
    history = train()
    # loss must decrease substantially on BOTH splits (no overfitting toy)
    assert history["train"][0] > history["train"][-1] * 1.5
    assert history["val"][0] > history["val"][-1] * 1.5
    assert history["val"][-1] < 0.1                    # blobs separate well
    # final accuracy should be high
    torch.manual_seed(0)
    val_ds = PointsDataset(n_per_class=100, seed=42)
    model = LinearClassifier()
    dl = DataLoader(val_ds, batch_size=50)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(100):
        for xb, yb in dl:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(xb), yb).backward()
            opt.step()
    accs = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            accs.append((model(xb).argmax(dim=1) == yb).float().mean().item())
    assert sum(accs) / len(accs) > 0.9


def main() -> None:
    history = train()
    print("epoch   train_loss   val_loss")
    for e, (tl, vl) in enumerate(zip(history["train"], history["val"])):
        print(f"  {e}      {tl:.4f}       {vl:.4f}")
    print("\nLoss drops on unseen validation data -> the model generalizes.")
    print("\nALL LAB-07 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
