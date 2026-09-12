"""Lab 02 — Matrix operations.

The four operations that dominate an LLM:
  1. element-wise ops     -> feedforward activations, LayerNorm math
  2. dot product          -> attention scores (similarity between two vectors)
  3. matmul               -> Q/K/V projections, logits projection
  4. batch matmul         -> attention applied to (batch, heads, seq, dim)

Shape rules to memorize:
  matmul:  (m, k) @ (k, n)        -> (m, n)          # inner dims must match
  batched: (B, m, k) @ (B, k, n)  -> (B, m, n)       # batch dim B stays put

Run:  python labs/02_matrix_ops.py
"""
import torch


def verify() -> None:
    # --- element-wise -------------------------------------------------------
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    assert torch.equal(x + y, torch.tensor([[11.0, 22.0], [33.0, 44.0]]))
    assert torch.equal(x * y, torch.tensor([[10.0, 40.0], [90.0, 160.0]]))  # Hadamard product
    assert torch.allclose(x ** 2, torch.tensor([[1.0, 4.0], [9.0, 16.0]]))
    # NOTE: x * y is element-wise; x @ y is matrix product. Different things!

    # --- dot product (1D) -----------------------------------------------------
    # a . b = sum(a_i * b_i) = |a||b|cos(angle) -> measures alignment/similarity.
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    assert torch.dot(a, b).item() == 4 + 10 + 18  # 32

    # --- matmul (2D) ----------------------------------------------------------
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = A @ B                       # (3,4) @ (4,5) -> (3,5); k=4 must match
    assert C.shape == (3, 5)
    # matmul = one dot product per (row of A, column of B):
    manual = sum(A[0, k] * B[k, 2] for k in range(4))
    assert torch.isclose(C[0, 2], manual, atol=1e-6)

    # incompatible inner dims raise
    try:
        A @ torch.randn(3, 5)       # (3,4) @ (3,5): 4 != 3
        raise AssertionError("expected matmul failure")
    except RuntimeError:
        pass

    # --- batch matmul -----------------------------------------------------------
    # Attention will hold (batch=2, heads=3, seq=7, head_dim=8).
    q = torch.randn(2, 3, 7, 8)     # queries
    k = torch.randn(2, 3, 7, 8)     # keys
    # scores[b,h,i,j] = q[b,h,i,:] . k[b,h,j,:]  ->  (2,3,7,7)
    scores = q @ k.transpose(-2, -1)
    assert scores.shape == (2, 3, 7, 7)
    # torch.matmul broadcasts: (2,3,7,8) @ (2,3,8,7) -> (2,3,7,7)
    # bmm requires exactly 3 dims and no broadcasting:
    s2 = torch.bmm(q.reshape(6, 7, 8), k.transpose(-2, -1).reshape(6, 8, 7))
    assert s2.shape == (6, 7, 7)
    assert torch.allclose(scores.reshape(6, 7, 7), s2, atol=1e-6)

    # --- einsum (compact notation for the same thing) -----------------------------
    # 'i k, j k -> i j'  ==  q_i . k_j for every pair (i, j): exactly the score
    # matrix of ONE head. Compare to scores[0, 0] (batch 0, head 0).
    s3 = torch.einsum("ik,jk->ij", q[0, 0], k[0, 0])   # (7,8) vs (7,8) -> (7,7)
    assert s3.shape == (7, 7)
    assert torch.allclose(s3, scores[0, 0], atol=1e-6)


def main() -> None:
    torch.manual_seed(0)
    a, b = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])
    print("dot product      :", torch.dot(a, b).item(), "(= 1*4 + 2*5 + 3*6)")
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    print("(3,4) @ (4,5)    ->", tuple((A @ B).shape))
    q, k = torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    print("q (2,3,7,8) @ k^T (2,3,8,7) ->", tuple((q @ k.transpose(-2, -1)).shape))
    print("bmm (6,7,8)@(6,8,7) ->", tuple(torch.bmm(q.reshape(6, 7, 8), k.transpose(-2, -1).reshape(6, 8, 7)).shape))
    print("\nALL LAB-02 ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
