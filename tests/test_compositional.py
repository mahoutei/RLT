"""Compositional chain tasks (scope expansion, part B)."""
import random
import torch


def test_chain_mod5_deterministic_labels_match_eval():
    from official_tasks import sample_chain_mod5, eval_chain_mod5, MOD_BOS, MOD_EQ
    toks1, lab1, pos1 = sample_chain_mod5(random.Random(7), 3)
    toks2, lab2, pos2 = sample_chain_mod5(random.Random(7), 3)
    assert (toks1, lab1, pos1) == (toks2, lab2, pos2)
    assert toks1[0] == MOD_BOS and toks1[-1] == MOD_EQ
    assert len(lab1) == 3 and len(pos1) == 3
    assert pos1 == sorted(pos1) and all(0 <= v <= 4 for v in lab1)
    assert lab1[-1] == eval_chain_mod5(toks1)


def test_chain_mod5_single_op_has_one_step():
    from official_tasks import sample_chain_mod5
    toks, lab, pos = sample_chain_mod5(random.Random(1), 1)
    assert len(lab) == 1 and len(pos) == 1


def test_chain_mod5_batch_pads_positions_with_minus_one():
    from official_tasks import chain_mod5_batch, MOD_PAD
    seq, lab, pos = chain_mod5_batch(4, 2, seed=9)
    assert seq.shape[0] == 4 and lab.shape == pos.shape
    assert ((pos == -1) == (lab == -100)).all()


def test_existing_parity_stream_unchanged():
    from official_tasks import make_parity_batch
    s, l = make_parity_batch(8, 12, seed=5)
    assert s.shape == (8, 12) and l.shape == (8,)
    assert int(l.sum()) == 3  # pinned value; fails if the stream changes


def test_step_token_loss_matches_manual_ce_and_ignores_pad():
    import torch
    import torch.nn.functional as F
    from train_official import step_token_loss
    torch.manual_seed(0)
    logits = torch.randn(2, 10, 14)
    pos = torch.tensor([[3, 5, -1], [2, -1, -1]])
    lab = torch.tensor([[1, 4, -100], [0, -100, -100]])
    got = step_token_loss(logits, pos, lab)
    manual = (F.cross_entropy(logits[0, 3].unsqueeze(0), torch.tensor([1]))
              + F.cross_entropy(logits[0, 5].unsqueeze(0), torch.tensor([4]))
              + F.cross_entropy(logits[1, 2].unsqueeze(0), torch.tensor([0]))) / 3
    assert torch.isclose(got, manual)
    empty = step_token_loss(logits, torch.full((2, 1), -1), torch.full((2, 1), -100))
    assert empty.item() == 0.0
    empty.backward()  # must still backprop without error


def test_first_error_position():
    import torch
    from train_official import first_error_position
    pred = torch.tensor([[1, 2, 3], [1, 9, 3]])
    true = torch.tensor([[1, 2, 3], [1, 2, 3]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    assert first_error_position(pred, true, valid).tolist() == [-1, 1]


def test_bracketed_steps_positions_increase_final_matches_eval():
    import random
    from official_tasks import sample_bracketed_steps, eval_bracketed
    toks, pos, lab = sample_bracketed_steps(random.Random(3), 20)
    toks2, pos2, lab2 = sample_bracketed_steps(random.Random(3), 20)
    assert (toks, pos, lab) == (toks2, pos2, lab2)
    assert pos == sorted(pos) and all(0 <= v <= 4 for v in lab)
    assert lab[-1] == eval_bracketed(toks)
    first_pos = pos[0]
    assert toks[first_pos] == lab[0] and toks[first_pos] <= 4  # first step is a leaf digit


def test_bracketed_steps_batch_pads():
    from official_tasks import bracketed_steps_batch
    seq, lab, pos = bracketed_steps_batch(3, 16, seed=11)
    assert seq.shape[0] == 3 and lab.shape == pos.shape
    assert ((pos == -1) == (lab == -100)).all()


def test_bitflip_noise_zero_is_identity_full_is_consistent():
    import random
    import torch
    from official_tasks import make_parity_batch, add_bitflip_noise
    s, l = make_parity_batch(8, 12, seed=5)
    s0, l0 = add_bitflip_noise(s, l, random.Random(2), 0.0)
    assert torch.equal(s0, s) and torch.equal(l0, l)
    s1, l1 = add_bitflip_noise(s, l, random.Random(2), 1.0)
    assert torch.equal(s1, 1 - s)
    assert torch.equal(l1, s1.sum(-1) % 2)
    sa, _ = add_bitflip_noise(s, l, random.Random(2), 0.1)
    sb, _ = add_bitflip_noise(s, l, random.Random(2), 0.1)
    assert torch.equal(sa, sb)


def test_mod5_distractors_keep_value():
    import random
    from official_tasks import encode_flat_mod5, mod5_with_distractors
    toks, v = encode_flat_mod5([1, 2, 3], ["+", "*"])
    noisy, v2 = mod5_with_distractors([1, 2, 3], ["+", "*"], random.Random(4), 2)
    assert v2 == v
    assert len(noisy) == len(toks) + 4


def test_bracketed_steps_trainer_branch_routes_and_pads():
    import random
    import torch
    from train_official import sample_train_batch, batch_loss, test_lengths
    rng = random.Random(0)
    kind, seq, lab, pos = sample_train_batch("bracketed-steps", 8, rng)
    assert kind == "bsteps"
    assert seq.shape[0] == 8 and lab.shape == pos.shape
    assert ((pos == -1) == (lab == -100)).all()
    assert test_lengths("bracketed-steps") == [16, 24, 32, 48, 64, 96]


def test_encode_bracketed_mod5_handles_parens():
    import random
    from official_tasks import sample_bracketed_mod5, encode_bracketed_mod5, MOD_BOS, MOD_EQ
    tt, v = sample_bracketed_mod5(random.Random(0), 20)
    toks = encode_bracketed_mod5(tt)
    assert toks[0].item() == MOD_BOS and toks[-1].item() == MOD_EQ
    assert len(toks) == len(tt) + 2
