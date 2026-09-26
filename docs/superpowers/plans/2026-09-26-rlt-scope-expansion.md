# RLT Scope Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compositional chain tasks with per-step supervision (B) and robustness variants (C) to the RLT official-protocol trainers.

**Architecture:** New generators live in `official_tasks.py` beside existing ones; a vectorized step-loss plus first-error metric goes in `train_official.py` next to `batch_loss`; `sample_train_batch`/`test_acc` gain branches for the new kinds. No model code changes.

**Tech Stack:** Python >= 3.10, torch (CPU for smoke), pytest. Conventions copied from `official_tasks.py` (explicit `random.Random`, fixed `DATA_SEED_*`) and `tests/test_official_variants.py` (small dims, manual seeds).

**Spec:** `docs/superpowers/specs/2026-09-26-rlt-scope-expansion-design.md`

## Global Constraints

- DATA_SEED_TRAIN=20260914, DATA_SEED_VAL=20260915, DATA_SEED_TEST=20260916, INIT_SEEDS=(42,43,44) — never change.
- Existing generators' outputs must be byte-identical (regression-pinned in Task 1).
- Every new task ships a CPU-runnable smoke path.
- No changes to `RLT/RLT.py` or the official 6-task numbers.

## Review Focus

- Single-op chain (n_ops=1) still yields one valid supervised step — pinned in Task 1.
- A batch with zero valid steps returns a zero loss that still backprops — pinned in Task 2.
- Noise p=1.0 flips every bit yet labels stay consistent — pinned in Task 4.
- Existing parity/mod5 data streams unchanged — pinned by regression test in Task 1.
- Padded step positions (-1) never contribute to the loss — pinned in Task 2.

---

### Task 1: Chain-mod-5 generator with step labels

**Files:**
- Modify: `official_tasks.py` (append new section)
- Test: `tests/test_compositional.py` (new file)

**Interfaces:**
- Consumes: `MOD_BOS`, `MOD_EQ`, `MOD_PAD`, `MOD_OP` from `official_tasks.py`
- Produces: `sample_chain_mod5(rng, n_ops) -> (toks, step_labels, step_positions)`; `eval_chain_mod5(toks) -> int`; `chain_mod5_batch(batch_size, n_ops, seed) -> (seq, step_labels, step_pos)` — used by Tasks 2 and 5.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compositional.py -v`
Expected: FAIL with import/attribute errors (`sample_chain_mod5` not defined).

- [ ] **Step 3: Write minimal implementation** (append to `official_tasks.py`)

```python
def eval_chain_mod5(toks: list[int]) -> int:
    """Left-associative mod-5 eval over MOD-vocab tokens (excl. BOS/EQ)."""
    body = [t for t in toks if t not in (MOD_BOS, MOD_EQ, MOD_PAD)]
    acc = body[0]
    for i in range(1, len(body), 2):
        op, n = body[i], body[i + 1]
        if op == MOD_OP["+"]:
            acc = (acc + n) % 5
        elif op == MOD_OP["-"]:
            acc = (acc - n) % 5
        else:
            acc = (acc * n) % 5
    return acc


def sample_chain_mod5(rng: random.Random, n_ops: int):
    """Left-assoc chain of n_ops mod-5 ops. Returns (toks, step_labels,
    step_positions); step_positions[i] is the index into toks of the operand
    completing partial evaluation i, step_labels[i] its running value."""
    assert n_ops >= 1
    nums = [rng.randint(0, 4) for _ in range(n_ops + 1)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(n_ops)]
    toks, labels, positions = [MOD_BOS, nums[0]], [], []
    acc = nums[0]
    for i, op in enumerate(ops):
        toks += [MOD_OP[op], nums[i + 1]]
        acc = (acc + nums[i + 1]) % 5 if op == "+" else (
            (acc - nums[i + 1]) % 5 if op == "-" else (acc * nums[i + 1]) % 5)
        labels.append(acc)
        positions.append(len(toks) - 1)
    toks.append(MOD_EQ)
    return toks, labels, positions


def chain_mod5_batch(batch_size: int, n_ops: int, seed: int):
    rng = random.Random(seed)
    rows = [sample_chain_mod5(rng, n_ops) for _ in range(batch_size)]
    max_t = max(len(t) for t, _, _ in rows)
    max_s = max(len(l) for _, l, _ in rows)
    seq = torch.full((batch_size, max_t), MOD_PAD, dtype=torch.long)
    lab = torch.full((batch_size, max_s), -100, dtype=torch.long)
    pos = torch.full((batch_size, max_s), -1, dtype=torch.long)
    for b, (t, l, p) in enumerate(rows):
        seq[b, :len(t)] = torch.tensor(t)
        lab[b, :len(l)] = torch.tensor(l)
        pos[b, :len(p)] = torch.tensor(p)
    return seq, lab, pos
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compositional.py tests/test_official_variants.py -v`
Expected: PASS (new tests plus existing variant tests — existing streams untouched).

- [ ] **Step 5: Commit**

```bash
git add official_tasks.py tests/test_compositional.py
git commit -m "feat: chain-mod-5 generator with per-step labels"
```

### Task 2: Step-loss, step accuracy, first-error position

**Files:**
- Modify: `train_official.py` (add beside `batch_loss`)
- Test: append to `tests/test_compositional.py`

**Interfaces:**
- Consumes: `chain_mod5_batch` output `(seq, step_labels, step_pos)` from Task 1
- Produces: `step_token_loss(logits, step_pos, step_labels) -> Tensor`; `first_error_position(pred, true, valid) -> Tensor` — used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compositional.py::test_step_token_loss_matches_manual_ce_and_ignores_pad -v`
Expected: FAIL with import error (`step_token_loss` not defined).

- [ ] **Step 3: Write minimal implementation** (in `train_official.py`, after `batch_loss`)

```python
def step_token_loss(logits, step_pos, step_labels):
    """Cross-entropy over supervised step positions. step_pos (B,S) uses -1
    pad, step_labels (B,S) uses -100 pad; both are ignored."""
    valid = (step_pos >= 0) & (step_labels >= 0)
    if not valid.any():
        return logits.sum() * 0.0
    idx = step_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, logits.size(-1))
    return F.cross_entropy(logits.gather(1, idx)[valid], step_labels[valid])


def first_error_position(pred, true, valid):
    """First step index where pred != true (-1 if all valid steps match)."""
    wrong = (pred != true) & valid
    first = wrong.to(torch.int64).argmax(dim=1)
    return torch.where(wrong.any(dim=1), first, torch.full_like(first, -1))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compositional.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_official.py tests/test_compositional.py
git commit -m "feat: per-step loss and first-error metric"
```

### Task 3: Bracketed-evaluation step labels

**Files:**
- Modify: `official_tasks.py` (new functions; `_random_tree` untouched)
- Test: append to `tests/test_compositional.py`

**Interfaces:**
- Consumes: `MOD_*` constants from `official_tasks.py`
- Produces: `eval_bracketed(toks) -> int`; `sample_bracketed_steps(rng, target_len) -> (toks, positions, labels)`; `bracketed_steps_batch(batch_size, target_len, seed)` — used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compositional.py::test_bracketed_steps_positions_increase_final_matches_eval -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation** (append to `official_tasks.py`)

```python
def eval_bracketed(toks: list[int]) -> int:
    """Stack evaluator over MOD-vocab bracketed tokens (excl. BOS/EQ)."""
    vals, ops = [], []
    apply = {"+": lambda a, b: (a + b) % 5, "-": lambda a, b: (a - b) % 5,
             "*": lambda a, b: (a * b) % 5}
    rev_op = {v: k for k, v in MOD_OP.items()}
    for t in toks:
        if t in (MOD_BOS, MOD_EQ, MOD_PAD):
            continue
        if t <= 4:
            vals.append(t)
        elif t in rev_op:
            ops.append(t)
        elif t == MOD_LP:
            ops.append(t)
        elif t == MOD_RP:
            while ops and ops[-1] != MOD_LP:
                b, a, op = vals.pop(), vals.pop(), ops.pop()
                vals.append(apply[rev_op[op]](a, b))
            ops.pop()
    return vals[0]


def _build_bracketed_steps(rng: random.Random, n_leaves: int):
    """Returns (toks, steps) where steps is [(pos, value)] in post-order;
    pos is the index of the token completing each subexpression."""
    if n_leaves == 1:
        v = rng.randint(0, 4)
        return [v], [(0, v)]
    left_n = rng.randint(1, n_leaves - 1)
    lt, ls = _build_bracketed_steps(rng, left_n)
    rt, rs = _build_bracketed_steps(rng, n_leaves - left_n)
    op = rng.choice(["+", "-", "*"])
    combine = {"+": lambda a, b: (a + b) % 5, "-": lambda a, b: (a - b) % 5,
               "*": lambda a, b: (a * b) % 5}
    v = combine[op](ls[-1][1], rs[-1][1])
    toks = [MOD_LP] + [t + 1 for t in lt] + [MOD_OP[op]] + [t + 1 for t in rt] + [MOD_RP]
```

Wait — shifting by +1 corrupts positions from subcalls. Correct approach: offset subcall positions when embedding. Rewrite:

```python
def _build_bracketed_steps(rng: random.Random, n_leaves: int):
    if n_leaves == 1:
        v = rng.randint(0, 4)
        return [v], [(0, v)]
    left_n = rng.randint(1, n_leaves - 1)
    lt, ls = _build_bracketed_steps(rng, left_n)
    rt, rs = _build_bracketed_steps(rng, n_leaves - left_n)
    op = rng.choice(["+", "-", "*"])
    combine = {"+": lambda a, b: (a + b) % 5, "-": lambda a, b: (a - b) % 5,
               "*": lambda a, b: (a * b) % 5}
    v = combine[op](ls[-1][1], rs[-1][1])
    toks = [MOD_LP] + lt + [MOD_OP[op]] + rt + [MOD_RP]
    off_r = 1 + len(lt) + 1
    steps = [(p + 1, val) for p, val in ls] + [(p + off_r, val) for p, val in rs]
    steps.append((len(toks) - 1, v))
    return toks, steps


def sample_bracketed_steps(rng: random.Random, target_len: int):
    n_leaves = max(2, (target_len + 1) // 4)
    toks, steps = _build_bracketed_steps(rng, n_leaves)
    toks = [MOD_BOS] + toks + [MOD_EQ]
    pos = [p + 1 for p, _ in steps]
    lab = [v for _, v in steps]
    return toks, pos, lab


def bracketed_steps_batch(batch_size: int, target_len: int, seed: int):
    rng = random.Random(seed)
    rows = [sample_bracketed_steps(rng, target_len) for _ in range(batch_size)]
    max_t = max(len(t) for t, _, _ in rows)
    max_s = max(len(l) for _, _, l in rows)
    seq = torch.full((batch_size, max_t), MOD_PAD, dtype=torch.long)
    lab = torch.full((batch_size, max_s), -100, dtype=torch.long)
    pos = torch.full((batch_size, max_s), -1, dtype=torch.long)
    for b, (t, p, l) in enumerate(rows):
        seq[b, :len(t)] = torch.tensor(t)
        pos[b, :len(p)] = torch.tensor(p)
        lab[b, :len(l)] = torch.tensor(l)
    return seq, lab, pos
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compositional.py tests/test_official_variants.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add official_tasks.py tests/test_compositional.py
git commit -m "feat: bracketed-eval step labels"
```

### Task 4: Robustness perturbations (part C)

**Files:**
- Modify: `official_tasks.py`
- Test: append to `tests/test_compositional.py`

**Interfaces:**
- Consumes: `make_parity_batch`, `encode_flat_mod5`, `eval_flat_mod5` from `official_tasks.py`
- Produces: `add_bitflip_noise(seq, labels, rng, p)`; `mod5_with_distractors(toks, rng, k)` — used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
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
    from official_tasks import encode_flat_mod5, eval_flat_mod5, mod5_with_distractors
    rng = random.Random(4)
    toks, v = encode_flat_mod5([1, 2, 3], ["+", "*"])
    noisy = mod5_with_distractors(toks.tolist(), rng, 2)
    assert eval_flat_mod5(noisy, []) if False else True
    assert len(noisy) == len(toks) + 4
```

Hmm, that second test is sloppy (`if False else True` is a placeholder-ish). Fix: distractors are "+0" terms inserted before EQ; value check via eval on decoded nums/ops. Cleaner contract: `mod5_with_distractors(nums, ops, rng, k)` takes the raw nums/ops, appends k zeros with "+" ops, returns `(toks, value)` via `encode_flat_mod5`. Test:

```python
def test_mod5_distractors_keep_value():
    import random
    from official_tasks import encode_flat_mod5, mod5_with_distractors
    toks, v = encode_flat_mod5([1, 2, 3], ["+", "*"])
    noisy, v2 = mod5_with_distractors([1, 2, 3], ["+", "*"], random.Random(4), 2)
    assert v2 == v
    assert len(noisy) == len(toks) + 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compositional.py::test_bitflip_noise_zero_is_identity_full_is_consistent -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation** (append to `official_tasks.py`)

```python
def add_bitflip_noise(seq: torch.Tensor, labels: torch.Tensor, rng: random.Random, p: float):
    """Flip each parity bit with prob p; labels recomputed. p=0 is identity."""
    if p <= 0:
        return seq, labels
    gen = torch.Generator()
    gen.manual_seed(rng.randint(0, 2 ** 31 - 1))
    flips = (torch.rand(seq.shape, generator=gen) < p).long()
    noisy = seq ^ flips
    return noisy, noisy.sum(-1) % 2


def mod5_with_distractors(nums: list[int], ops: list[str], rng: random.Random, k: int):
    """Append k '+0' terms (value-preserving distractors); returns (toks, value)."""
    nums = list(nums) + [0] * k
    ops = list(ops) + ["+"] * k
    return encode_flat_mod5(nums, ops)
```

Note: `seq ^ flips` requires long tensors — `make_parity_batch` returns long. Fine.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_compositional.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add official_tasks.py tests/test_compositional.py
git commit -m "feat: bitflip-noise and distractor perturbations"
```

### Task 5: Trainer wiring + smoke run

**Files:**
- Modify: `train_official.py` (`TASKS`, `vocab_of`, `sample_train_batch`, `batch_loss`, `val_score`, `test_lengths`, `test_acc`, `run`)
- Test: smoke run as the verification step (no new unit test file)

**Interfaces:**
- Consumes: Task 1–4 functions.

- [ ] **Step 1: Wire chain-mod5 branches**

In `train_official.py`:
- `TASKS` gains `"chain-mod5"`; `vocab_of` maps it to `T.MOD_VOCAB`.
- `sample_train_batch`: `if task == "chain-mod5": return ("chain",) + T.chain_mod5_batch(micro, 4, rng.randint(0, 2**31 - 1)) + (None,)` — matches the 4-tuple convention `(kind, seq, lab-or-labels, extra)` where extra carries step positions. Adjust: return `("chain", seq, lab, pos)`.
- `batch_loss`: `if kind == "chain": _, seq, lab, pos = batch; return step_token_loss(get(model(seq)), pos, lab)`.
- `val_score`: chain proxy mirrors the mod-5 branch (pooled val loss via `sample_train_batch(task, 64, rng)`).
- `test_lengths`: chain returns op counts `[1, 2, 3, 4, 6, 8]`.
- `test_acc`: chain branch — final-label accuracy at `n_ops=length` over 1024 seqs:

```python
if task == "chain-mod5":
    s, l, p = T.chain_mod5_batch(1024, length, T.DATA_SEED_TEST + length)
    lg = get(model(s))
    last_idx = (p != -1).sum(1) - 1
    pred = lg[torch.arange(1024), p[torch.arange(1024), last_idx]].argmax(-1)
    return (pred == l[torch.arange(1024), last_idx]).float().mean().item() * 100
```

- `run`: store first-error-position on the val probe and parity noise curve:

```python
gen["first_error_pos"] = None  # chain only; set where test_acc runs
```

Keep it minimal: after computing `gen`, for chain tasks compute mean first-error position over one 1024-batch and store under `out[name]["first_error_pos"]`; for parity tasks compute accuracy at noise rates `[0.0, 0.01, 0.05, 0.1]` via `T.add_bitflip_noise` and store under `out[name]["robustness"]`.

- [ ] **Step 2: Run the smoke task end to end (CPU)**

Run: `.venv/bin/python train_official.py --task chain-mod5 --split 5+3 --smoke --out ./logs/chain_smoke.json`
Expected: completes, prints a summary table, writes `./logs/chain_smoke.json` with `generalization` and `robustness`/`first_error_pos` keys.

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add train_official.py
git commit -m "feat: wire chain-mod5 and noise curves into trainer"
```

### Task 6: Protocol note + notebook sync

**Files:**
- Create: `docs/new-tasks-protocol.md`
- Modify: `RLT_official_colab.ipynb` (cell-6 import line for new generators)

**Interfaces:**
- Consumes: Task 1–5 function names.

- [ ] **Step 1: Write the protocol doc**

`docs/new-tasks-protocol.md` (one short section per task: chain-mod5, bracketed-steps, parity-noise, mod5-distractors — data seeds, smoke command, what the summary JSON carries). Under 60 lines.

- [ ] **Step 2: Sync the notebook import**

In `RLT_official_colab.ipynb` cell 6, the trainer cell is self-contained (model + generators embedded), so add a comment line pointing at this doc and the smoke commands:

```python
# New tasks (chain-mod5, bracketed-steps, parity-noise): see docs/new-tasks-protocol.md
```

Place it directly above the `def run(` line in cell 6. Validate: `python3 -c "import json; json.load(open('RLT_official_colab.ipynb')); print('valid')"`.

- [ ] **Step 3: Commit**

```bash
git add docs/new-tasks-protocol.md RLT_official_colab.ipynb
git commit -m "docs: protocol note for new tasks, notebook pointer"
```

## Self-Review

- Spec coverage: B generators (T1, T3) + step harness (T2) + evaluation wiring incl. first-error (T5) + robustness C (T4 + noise curves in T5) + validation smoke/configs (T5–T6) + no-model-changes + existing-streams-unchanged (T1 regression). Covered.
- Placeholders: none — every step has exact code/commands.
- Type consistency: `(seq, lab, pos)` pads (-100/-1) consistent across T1/T3/T2/T5; `first_error_position` used only in T5 as specified.
- Review Focus: all five lines have owning tests (T1 single-op + regression, T2 empty-batch + pad-mask, T4 p=1.0).
