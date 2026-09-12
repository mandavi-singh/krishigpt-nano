"""Lab 01 — Tensor fundamentals.

Everything in an LLM is tensors: token IDs are integer tensors, embeddings are
float tensors of shape (batch, seq_len, embed_dim), attention scores are
(batch, heads, seq_len, seq_len). Getting fluent with shapes here makes the
rest of the project readable.

Run:  python labs/01_tensors.py
"""
import torch


def verify() -> None:
    """Assertions that prove the concepts in this lab."""
    # --- creation ---------------------------------------------------------
    a = torch.tensor([1.0, 2.0, 3.0])          # from Python data
    z = torch.zeros(2, 3)                      # all zeros
    o = torch.ones(2, 3)                       # all ones
    r = torch.randn(2, 3)                      # standard normal (random)
    s = torch.arange(0, 6)                     # 0..5
    assert a.shape == torch.Size([3])
    assert z.shape == o.shape == r.shape == torch.Size([2, 3])
    assert s.dtype == torch.int64              # arange of ints -> int64

    # --- dtype ------------------------------------------------------------
    assert r.dtype == torch.float32            # torch default dtype
    ri = s.float()                             # cast int64 -> float32
    assert ri.dtype == torch.float32

    # --- indexing / slicing -------------------------------------------------
    m = torch.arange(12).reshape(3, 4)         # rows [0..3], [4..7], [8..11]
    assert m[0, 0].item() == 0
    assert m[2, 3].item() == 11
    assert torch.equal(m[1], torch.arange(4, 8))      # row 1
    assert torch.equal(m[:, 0], torch.tensor([0, 4, 8]))  # column 0
    assert torch.equal(m[0:2, 1:3], torch.tensor([[1, 2], [5, 6]]))

    # --- reshape / view -----------------------------------------------------
    v = m.view(2, 6)                           # view: zero-copy, needs contiguous memory
    assert v.shape == torch.Size([2, 6])
    assert v.data_ptr() == m.data_ptr()        # same underlying memory!
    rs = m.reshape(6, 2)                       # reshape: view if possible, else copy
    assert rs.shape == torch.Size([6, 2])
    # -1 means "infer this dimension": 12 elements -> (4, 3)
    f = m.reshape(-1, 3)
    assert f.shape == torch.Size([4, 3])

    # --- transpose ----------------------------------------------------------
    t = m.t()                                  # 2D transpose: rows <-> columns
    assert t.shape == torch.Size([4, 3])
    assert t[0, 1].item() == m[1, 0].item()
    b = torch.randn(2, 3, 5)
    assert b.transpose(0, 2).shape == torch.Size([5, 3, 2])
    # transpose changes LOGICAL layout without moving data; the tensor becomes
    # non-contiguous. .contiguous() materializes a copy in the new row-major order.
    assert not b.transpose(0, 2).is_contiguous()
    assert b.transpose(0, 2).contiguous().is_contiguous()

    # --- broadcasting ---------------------------------------------------------
    # Rules: dimensions are aligned from the right; sizes must be equal or one
    # of them 1. The size-1 dim is "stretched" (no copy is made).
    mat = torch.arange(6).reshape(2, 3).float()  # (2, 3)
    row = torch.tensor([10.0, 20.0, 30.0])       # (3,) -> treated as (1, 3) -> (2, 3)
    added = mat + row
    assert added.shape == torch.Size([2, 3])
    assert torch.equal(added, mat + row.unsqueeze(0))
    col = torch.tensor([[100.0], [200.0]])       # (2, 1) -> broadcasts across columns
    assert (mat + col).shape == torch.Size([2, 3])
    # (2,3) + (2,1) + (3,) all combine legally: each size-1 / missing dim stretches
    assert (mat + col + row).shape == torch.Size([2, 3])
    try:
        mat + torch.randn(4)                     # (2,3) + (4,) -> illegal
        raise AssertionError("expected broadcast failure")
    except RuntimeError:
        pass                                     # good: incompatible shapes are rejected


def main() -> None:
    torch.manual_seed(0)
    m = torch.arange(12).reshape(3, 4)
    print("m =\n", m)
    print("m.shape, m.dtype:", tuple(m.shape), m.dtype)
    print("row 1:       ", m[1])
    print("col 0:       ", m[:, 0])
    print("m.view(2,6):\n", m.view(2, 6), " <- same memory, zero copy")
    print("m.t() shape: ", tuple(m.t().shape), " contiguous?", m.t().is_contiguous())
    mat = torch.arange(6).reshape(2, 3).float()
    row = torch.tensor([10.0, 20.0, 30.0])
    print("\nbroadcasting (2,3) + (3,):\n", mat + row)
    print("\nALL LAB-01 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
