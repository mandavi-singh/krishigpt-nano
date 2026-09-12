"""Tests for the assembled KrishiGPT-nano GPT."""
import torch
import torch.nn as nn

from model.gpt import GPT, GPTConfig

V, D, H, N = 64, 32, 4, 2     # tiny for fast tests


def make(tie: bool = True) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=V, d_model=D, n_heads=H, n_layers=N,
                         max_len=64, tie_weights=tie))


def test_logits_shape() -> None:
    m = make()
    out = m(torch.randint(0, V, (2, 9)))
    assert out.shape == (2, 9, V)


def test_parameter_accounting_tied() -> None:
    m = make(tie=True)
    blk = 4 * D * D + 4 * D + 8 * D * D + 5 * D + 4 * D      # bias=True block
    expected = N * blk + V * D + 2 * D                       # embed + final LN only
    assert sum(p.numel() for p in m.parameters()) == expected


def test_parameter_accounting_untied() -> None:
    m = make(tie=False)
    blk = 4 * D * D + 4 * D + 8 * D * D + 5 * D + 4 * D
    expected = N * blk + V * D + 2 * D + V * D               # + separate head
    assert sum(p.numel() for p in m.parameters()) == expected


def test_weight_tying_is_same_storage() -> None:
    m = make(tie=True)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    # a gradient step through the head must update the embedding too
    ids = torch.randint(0, V, (1, 5))
    loss = m(ids).sum()
    loss.backward()
    assert m.tok_emb.weight.grad is not None
    assert torch.any(m.tok_emb.weight.grad != 0)


def test_untied_is_independent_storage() -> None:
    m = make(tie=False)
    assert m.lm_head.weight.data_ptr() != m.tok_emb.weight.data_ptr()


def test_loss_on_shifted_targets() -> None:
    m = make()
    ids = torch.randint(0, V, (2, 8))
    logits = m(ids)
    loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1)
    )
    assert torch.isfinite(loss)
    # untrained ~ ln(V); allow generous slack
    assert abs(loss.item() - float(torch.log(torch.tensor(float(V))))) < 0.5


def test_causality_through_full_model() -> None:
    """Editing a token must not change logits at earlier positions."""
    m = make()
    m.eval()
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    a = m(ids)
    ids2 = torch.tensor([[3, 5, 7, 61, 12]])          # change positions 3..4
    b = m(ids2)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-5)
    assert not torch.allclose(a[:, 3:], b[:, 3:], atol=1e-3)


def test_translation_invariance_rope_only() -> None:
    """No learned position table: shifting the positions argument must not
    change the logits at all (content + RoPE is offset-only). Shift stays
    within max_len — the RoPE table IS the context-window boundary, so
    positions >= max_len index out of bounds by design."""
    m = make()
    m.eval()
    ids = torch.tensor([[3, 5, 7, 9]])
    a = m(ids, positions=torch.arange(4))
    b = m(ids, positions=torch.arange(4) + 20)     # in-range shift
    assert torch.allclose(a, b, atol=1e-5)


def test_init_scales() -> None:
    m = make()
    w = m.blocks[0].attn.wq.weight
    assert abs(w.std().item() - 0.02) < 0.005          # GPT-2 init
    wo = m.blocks[0].attn.wo.weight
    assert wo.std().item() < w.std().item() * 0.8      # 1/sqrt(2N) down-scale
    assert m.ln_final.gamma.data.eq(1).all()
    assert m.ln_final.beta.data.eq(0).all()


def test_gradient_flows_to_all_parameters() -> None:
    m = make()
    ids = torch.randint(0, V, (1, 6))
    m(ids).sum().backward()
    dead = [n for n, p in m.named_parameters()
            if p.grad is None or not torch.any(p.grad != 0)]
    assert dead == [], dead
