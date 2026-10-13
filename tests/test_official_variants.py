"""Official-variant checks: RLT-0 control and RLT-2 chunk semantics (App. I, J.2)."""
import torch
from RLT import RLT


def test_rlt0_removes_feedback_params_and_runs():
    torch.manual_seed(0)
    m1 = RLT(num_tokens=2, dim=32, enc_depth=1, dec_depth=1)
    m0 = RLT(num_tokens=2, dim=32, enc_depth=1, dec_depth=1, use_feedback=False)
    assert m0.initial_state is None
    assert m0.combine_encoded_token_and_state is None
    assert sum(p.numel() for p in m1.parameters()) > sum(p.numel() for p in m0.parameters())
    t = torch.randint(0, 2, (2, 12))
    out, mem = m0(t)
    assert out.shape == (2, 12, 2)
    assert mem.decoder_memories.state is None
    out.sum().backward()


def test_rlt2_chunk1_recovers_rlt1():
    torch.manual_seed(1)
    m1 = RLT(num_tokens=2, dim=32, enc_depth=1, dec_depth=1, dec_sliding_window_size=4)
    m2 = RLT(num_tokens=2, dim=32, enc_depth=1, dec_depth=1, dec_sliding_window_size=4,
              feedback_chunk_size=1)
    m2.load_state_dict(m1.state_dict())
    m1.eval()
    m2.eval()
    t = torch.randint(0, 2, (2, 16))
    o1, _ = m1(t)
    o2, _ = m2(t)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_rlt2_chunk_parallel_matches_sequential():
    torch.manual_seed(2)
    m = RLT(num_tokens=2, dim=32, enc_depth=1, dec_depth=1, dec_sliding_window_size=4,
            feedback_chunk_size=4)
    m.eval()
    t = torch.randint(0, 2, (2, 16))
    par, _ = m(t)
    mem, outs = None, []
    for i in range(0, 16, 4):
        o, mem = m(t[:, i:i + 4], memories=mem)
        outs.append(o)
    assert torch.allclose(par, torch.cat(outs, dim=1), atol=1e-5)


def test_activation_arms_build_and_run():
    import torch
    import torch.nn.functional as F
    from train_official import build
    plain_s, _ = build(64, 4, 5, 3, 2, variant="rlt1", activation="swiglu")
    plain_g, _ = build(64, 4, 5, 3, 2, variant="rlt1", activation="gelu")
    n_s = sum(p.numel() for p in plain_s.parameters())
    n_g = sum(p.numel() for p in plain_g.parameters())
    assert n_s != n_g  # GLU gate projection vs plain FF differ in size
    torch.manual_seed(0)
    t = torch.randint(0, 2, (2, 12))
    for m in (plain_s, plain_g):
        m.eval()
        with torch.no_grad():
            out = m(t)
            logits = out[0] if isinstance(out, tuple) else out
            assert torch.isfinite(logits).all()
            assert F.cross_entropy(logits[:, -1], torch.zeros(2, dtype=torch.long)).item() > 0
    import pytest
    with pytest.raises(AssertionError):
        build(64, 4, 5, 3, 2, activation="relu")
