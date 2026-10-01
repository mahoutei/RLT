"""Shared data generators + metrics matching the official RLT protocol.

Reference: yifanzhang-pro/recurrent-looped-tranformer, paper App. J.
  formal data seeds: train 20260914 / val 20260915 / test 20260916
  addition data seed: 42
  init seeds: 42, 43, 44 (fixed train stream + held-out across inits)
  depth-8: 2000 steps, global batch 512, micro 32
  mod-5 variants: 5000 steps

Token conventions here are explicit and minimal (the official repo ships no
code, only the spec). Loss masking follows the paper: parity / mod-5 score
the final label only; S5 scores every prefix (final-state primary);
addition scores teacher-forced answer tokens.
"""
from __future__ import annotations

import itertools
import math
import random

import torch

# seeds from App. J.4
DATA_SEED_TRAIN = 20260914
DATA_SEED_VAL = 20260915
DATA_SEED_TEST = 20260916
DATA_SEED_ADDITION = 42
INIT_SEEDS = (42, 43, 44)

# ---------------------------------------------------------------------------
# parity: binary strings 3-40 train, val {32,36,40}, test to 256
# ---------------------------------------------------------------------------

def sample_parity(rng: random.Random, length: int):
    bits = [rng.randint(0, 1) for _ in range(length)]
    return bits, sum(bits) % 2


def make_parity_batch(batch_size: int, length: int, seed: int):
    """Returns (seq [B,L] of 0/1, final parity [B]). Deterministic per seed."""
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, 2, (batch_size, length), generator=g)
    labels = (seq.sum(dim=-1) % 2).long()
    return seq, labels


def sample_walk_batch(batch_size: int, n_steps: int, seed: int):
    """1D symmetric walk: tokens 0/1 mean steps -1/+1. Returns (seq [B,N],
    sign(final displacement) [B]: 1 if end > start else 0). Deterministic."""
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, 2, (batch_size, n_steps), generator=g)
    displacement = seq.float().mul(2).sub(1).sum(dim=-1)
    return seq, (displacement > 0).long()


def sample_walk2d_batch(batch_size: int, n_steps: int, seed: int):
    """2D symmetric walk: tokens 0/1/2/3 mean +x/-x/+y/-y. Returns (seq [B,N],
    quadrant(final displacement) [B]: 2*sx + sy with sx/sy = 1 iff coord > 0
    (ties fold to 0, same convention as the 1D walk). Deterministic."""
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, 4, (batch_size, n_steps), generator=g)
    dx = (seq == 0).float() - (seq == 1).float()
    dy = (seq == 2).float() - (seq == 3).float()
    sx = (dx.sum(dim=-1) > 0).long()
    sy = (dy.sum(dim=-1) > 0).long()
    return seq, (2 * sx + sy).long()


def sample_parity_length(rng: random.Random, lo: int = 3, hi: int = 40):
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# addition: 1-8 digit operands, reversed-digit serialization, teacher-forced
# vocab: 0-9 digits, 10='+', 11='=', 12=BOS, 13=EOS, 14=PAD
# format: BOS rev(a) + rev(b) = rev(a+b) EOS   (all digit fields reversed)
# ---------------------------------------------------------------------------

ADD_VOCAB = 15
ADD_BOS, ADD_EOS, ADD_PAD = 12, 13, 14
ADD_PLUS, ADD_EQ = 10, 11


def sample_addition(rng: random.Random, lo_digits: int = 1, hi_digits: int = 8):
    da = rng.randint(lo_digits, hi_digits)
    db = rng.randint(lo_digits, hi_digits)
    a = rng.randint(0 if da == 1 else 10 ** (da - 1), 10**da - 1)
    b = rng.randint(0 if db == 1 else 10 ** (db - 1), 10**db - 1)
    return a, b


def encode_addition(a: int, b: int, max_len: int | None = None):
    ra = list(map(int, str(a)[::-1]))
    rb = list(map(int, str(b)[::-1]))
    rc = list(map(int, str(a + b)[::-1]))
    toks = [ADD_BOS] + ra + [ADD_PLUS] + rb + [ADD_EQ] + rc + [ADD_EOS]
    # mask: 1 on answer tokens (rev sum + EOS), 0 elsewhere/pad
    ans_start = 1 + len(ra) + 1 + len(rb) + 1
    mask = [0] * ans_start + [1] * (len(rc) + 1)
    if max_len is not None:
        pad = max_len - len(toks)
        toks += [ADD_PAD] * pad
        mask += [0] * pad
    return torch.tensor(toks, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def addition_max_len(max_digits: int = 8):
    # BOS + 8 + PLUS + 8 + EQ + 9 (carry) + EOS
    return 1 + max_digits + 1 + max_digits + 1 + (max_digits + 1) + 1


# ---------------------------------------------------------------------------
# mod-5: flat (no brackets, odd lens 3-39) and bracketed trees (3-40)
# vocab: 0-4 digits, 5='+', 6='-', 7='*', 8='(', 9=')', 10='=', 11=BOS,
#        12=EOS, 13=PAD. Final value (0-4) supervised.
# ---------------------------------------------------------------------------

MOD_VOCAB = 14
MOD_BOS, MOD_EOS, MOD_PAD = 11, 12, 13
MOD_OP = {"+": 5, "-": 6, "*": 7}
MOD_LP, MOD_RP, MOD_EQ = 8, 9, 10


def eval_flat_mod5(nums: list[int], ops: list[str]) -> int:
    # * before +,-  (standard precedence, left-assoc)
    vals, cur_ops = [nums[0]], []
    for op, n in zip(ops, nums[1:]):
        if op == "*":
            vals[-1] = (vals[-1] * n) % 5
        else:
            vals.append(n)
            cur_ops.append(op)
    total = vals[0]
    for op, v in zip(cur_ops, vals[1:]):
        total = (total + v) % 5 if op == "+" else (total - v) % 5
    return total % 5


def sample_flat_mod5(rng: random.Random, odd_len: int):
    n_nums = (odd_len + 1) // 2
    nums = [rng.randint(0, 4) for _ in range(n_nums)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(n_nums - 1)]
    return nums, ops, eval_flat_mod5(nums, ops)


def encode_flat_mod5(nums, ops):
    toks = [MOD_BOS]
    for i, n in enumerate(nums):
        toks.append(n)
        if i < len(ops):
            toks.append(MOD_OP[ops[i]])
    toks += [MOD_EQ]
    return torch.tensor(toks, dtype=torch.long), eval_flat_mod5(nums, ops)


def sample_bracketed_mod5(rng: random.Random, target_len: int):
    """Random full binary tree with +-* internal nodes, leaves 0-4.

    target_len counts payload tokens (digits+ops+parens, excl. BOS/EQ).
    Grows a tree until the payload reaches/exceeds target, then trims by
    collapsing back — simple approach: build random tree with bounded leaves.
    """
    n_leaves = max(2, (target_len + 1) // 4)
    return _random_tree(rng, n_leaves)


def _random_tree(rng: random.Random, n_leaves: int):
    if n_leaves == 1:
        v = rng.randint(0, 4)
        return ([str(v)], v)
    left_n = rng.randint(1, n_leaves - 1)
    lt, lv = _random_tree(rng, left_n)
    rt, rv = _random_tree(rng, n_leaves - left_n)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        v = (lv + rv) % 5
    elif op == "-":
        v = (lv - rv) % 5
    else:
        v = (lv * rv) % 5
    return (["("] + lt + [op] + rt + [")"], v)


def encode_bracketed_mod5(tree_toks: list[str]):
    mp = {"+": 5, "-": 6, "*": 7, "(": 8, ")": 9}
    toks = [MOD_BOS] + [mp[t] if t in mp else int(t) for t in tree_toks] + [MOD_EQ]
    return torch.tensor(toks, dtype=torch.long)


# ---------------------------------------------------------------------------
# S5: symmetric group on 5 elements, 120 perms.
# standard: all 120; swaps: identity + 10 single transpositions.
# compose 32 perms in training; supervise running product after each input.
# Represent perm as tuple of 5 images of 0..4. Product: apply left then right:
#   (p*q)[i] = q[p[i]]  (running product p_{k} = p_{k-1} composed with input)
# ---------------------------------------------------------------------------

def _all_s5():
    return list(itertools.permutations(range(5)))


ALL_S5 = _all_s5()
IDENTITY = (0, 1, 2, 3, 4)


def _transpositions():
    out = [IDENTITY]
    for i in range(5):
        for j in range(i + 1, 5):
            p = list(range(5))
            p[i], p[j] = p[j], p[i]
            out.append(tuple(p))
    return out


SWAPS_S5 = _transpositions()  # 1 + 10 = 11
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_S5)}


def compose(p, q):
    return tuple(q[p[i]] for i in range(5))


def sample_s5(rng: random.Random, length: int, variant: str):
    pool = ALL_S5 if variant == "standard" else SWAPS_S5
    inputs = [rng.choice(pool) for _ in range(length)]
    states, cur = [], IDENTITY
    for x in inputs:
        cur = compose(cur, x)
        states.append(PERM_TO_IDX[cur])
    return [PERM_TO_IDX[x] for x in inputs], states


def s5_batch(batch_size: int, length: int, variant: str, seed: int):
    g = torch.Generator().manual_seed(seed)
    pool_n = 120 if variant == "standard" else 11
    idx = torch.randint(0, pool_n, (batch_size, length), generator=g)
    pool = ALL_S5 if variant == "standard" else SWAPS_S5
    inputs = torch.tensor([[PERM_TO_IDX[pool[i]] for i in row] for row in idx.tolist()])
    # running products
    states = torch.empty_like(inputs)
    for b in range(batch_size):
        cur = IDENTITY
        for t in range(length):
            cur = compose(cur, pool[idx[b, t].item()])
            states[b, t] = PERM_TO_IDX[cur]
    return inputs, states


# ---------------------------------------------------------------------------
# metrics: mean +- sample SD (ddof=1) over init seeds, matching App. J.3
# ---------------------------------------------------------------------------

def mean_sample_sd(vals: list[float]):
    vals = [v for v in vals if v == v]  # drop NaN
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


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
