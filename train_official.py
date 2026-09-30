"""Generic official-protocol trainer for all six RLT algorithmic tasks.

Tasks: parity, addition, mod5-flat, mod5-bracketed, s5-standard, s5-swaps
Splits: e.g. 4+4, 5+3, 6+2, 7+1, 8+0 (+ plain Transformer baseline)

Official hyperparams (depth-8): d=512, FFN=1365, heads=4, SWA=8, G=1,
alpha=0.1, TBPTT=128, batch 512/micro 32, AdamW(0.9,0.95) wd 0.1 clip 1,
warmup 200 1e-4 -> cosine 5e-6 @2000 steps (5000 for mod-5 variants).
Seeds 42,43,44 on fixed data; best ID-loss checkpoint; shared test sets.

Example:
  python train_official.py --task parity --split 5+3 --smoke
  python train_official.py --task s5-swaps --split 4+4 --smoke
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from x_transformers import TransformerWrapper, Decoder


def system_stats() -> dict:
    """Best-effort host stats. Mirrors train_enwik8.system_stats; missing tools mean missing keys."""
    stats: dict = {}
    try:
        import psutil
        stats["cpu_pct"] = psutil.cpu_percent()
        stats["ram_gb"] = psutil.virtual_memory().used / 1e9
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output = True, text = True, timeout = 10)
        utils, mems = [], []
        for line in out.stdout.strip().splitlines():
            u, m = line.split(",")
            utils.append(float(u))
            mems.append(float(m))
        if utils:
            stats["gpu_util_pct"] = sum(utils) / len(utils)
            stats["gpu_mem_mb"] = sum(mems)
    except Exception:
        pass
    return stats


def _agg_sys(samples: list) -> dict:
    def agg(key):
        v = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        return {"avg": sum(v) / len(v), "peak": max(v)} if v else None
    return {"gpu_util_pct": agg("gpu_util_pct"), "gpu_mem_mb": agg("gpu_mem_mb"),
            "cpu_pct": agg("cpu_pct"), "ram_gb": agg("ram_gb"),
            "num_samples": len(samples)}

from RLT import RLT
import official_tasks as T

BASE_LR, MIN_LR, WARMUP = 1e-4, 5e-6, 200
TASKS = ("parity", "addition", "mod5-flat", "mod5-bracketed", "s5-standard", "s5-swaps", "chain-mod5", "bracketed-steps")


def lr_schedule(step, total):
    if step < WARMUP:
        return BASE_LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, total - WARMUP)
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * p))


def build(dim, heads, enc, dec, vocab, swa=8, variant="rlt1", chunk_size=None,
          activation="swiglu"):
    """activation: 'swiglu' (status quo) or 'gelu'. Toggles the plain baseline's
    x_transformers FF (GLU+SwiGLU vs plain GELU). RLT's internal blocks are
    GELU-fixed, so this ablates the baseline FF, not the whole stack."""
    assert activation in ("swiglu", "gelu"), activation
    dim_head = dim // heads
    ffn_mult = 1365 / 512
    plain = TransformerWrapper(
        num_tokens=vocab, max_seq_len=2048,
        attn_layers=Decoder(dim=dim, depth=enc + dec, heads=heads,
                            attn_dim_head=dim_head, rotary_pos_emb=True,
                            rotary_emb_dim=min(dim_head, 64),
                            use_rmsnorm=True, ff_glu=(activation == "swiglu"), ff_mult=ffn_mult))
    if variant == "rlt0":
        # official control: no feedback, full BPTT, -787,968 params/split at d=512
        rlt = RLT(num_tokens=vocab, dim=dim, enc_depth=enc, dec_depth=dec,
                  heads=heads, dim_head=dim_head, dec_sliding_window_size=swa,
                  use_feedback=False,
                  num_kv_layers=1, ff_expansion_factor=ffn_mult)
    elif variant == "rlt2":
        assert chunk_size is not None and chunk_size >= 1, 'RLT-2 needs --chunk_size >= 1 (B=1 recovers RLT-1)'
        rlt = RLT(num_tokens=vocab, dim=dim, enc_depth=enc, dec_depth=dec,
                  heads=heads, dim_head=dim_head, dec_sliding_window_size=swa,
                  recurrent_transition_alpha=0.1, tbptt_step_size=128,
                  feedback_chunk_size=chunk_size,
                  num_kv_layers=1, ff_expansion_factor=ffn_mult)
    else:
        assert variant == "rlt1", variant
        rlt = RLT(num_tokens=vocab, dim=dim, enc_depth=enc, dec_depth=dec,
                  heads=heads, dim_head=dim_head, dec_sliding_window_size=swa,
                  recurrent_transition_alpha=0.1, tbptt_step_size=128,
                  num_kv_layers=1, ff_expansion_factor=ffn_mult)
    return plain, rlt


def vocab_of(task):
    return {"parity": 2, "addition": T.ADD_VOCAB, "mod5-flat": T.MOD_VOCAB,
            "mod5-bracketed": T.MOD_VOCAB, "s5-standard": 120, "s5-swaps": 120,
            "chain-mod5": T.MOD_VOCAB, "bracketed-steps": T.MOD_VOCAB}[task]


def sample_train_batch(task, micro, rng):
    if task == "parity":
        L = rng.randint(3, 40)
        return ("parity",) + T.make_parity_batch(micro, L, rng.randint(0, 2**31 - 1)) + (L,)
    if task == "addition":
        Lm = T.addition_max_len()
        toks, masks, labs = [], [], []
        for _ in range(micro):
            da = rng.randint(1, 8)
            db = rng.randint(1, 8)
            a = rng.randint(0 if da == 1 else 10**(da-1), 10**da - 1)
            b = rng.randint(0 if db == 1 else 10**(db-1), 10**db - 1)
            t, m = T.encode_addition(a, b, max_len=Lm)
            toks.append(t); masks.append(m)
            full = t.clone(); labs.append(full)
        return ("addition", torch.stack(toks), torch.stack(labs), torch.stack(masks))
    if task == "mod5-flat":
        seqs, labs = [], []
        L = rng.choice([x for x in range(3, 40, 2)])
        for _ in range(micro):
            nums = [rng.randint(0, 4) for _ in range((L + 1) // 2)]
            ops = [rng.choice(["+", "-", "*"]) for _ in range((L - 1) // 2)]
            t = T.encode_flat_mod5(nums, ops)[0]
            seqs.append(t); labs.append(T.eval_flat_mod5(nums, ops))
        return ("mod5", torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=T.MOD_PAD),
                torch.tensor(labs), L)
    if task == "mod5-bracketed":
        seqs, labs = [], []
        for _ in range(micro):
            tt, v = T.sample_bracketed_mod5(rng, rng.randint(3, 40))
            seqs.append(T.encode_bracketed_mod5(tt)); labs.append(v)
        return ("mod5", torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=T.MOD_PAD),
                torch.tensor(labs), None)
    if task == "chain-mod5":
        seq, lab, pos = T.chain_mod5_batch(micro, 4, rng.randint(0, 2**31 - 1))
        return ("chain", seq, lab, pos)
    if task == "bracketed-steps":
        seq, lab, pos = T.bracketed_steps_batch(micro, 16, rng.randint(0, 2**31 - 1))
        return ("bsteps", seq, lab, pos)
    variant = "standard" if task == "s5-standard" else "swaps"
    L = 32
    return ("s5",) + T.s5_batch(micro, L, variant, rng.randint(0, 2**31 - 1)) + (L,)


def batch_loss(model, task, batch):
    kind = batch[0]
    get = lambda o: o[0] if isinstance(o, tuple) else o
    if kind == "parity":
        _, seq, lab, _ = batch
        return F.cross_entropy(get(model(seq))[:, -1], lab)
    if kind == "addition":
        _, seq, lab, mask = batch
        logits = get(model(seq[:, :-1]))  # next-token LM
        tgt = seq[:, 1:].clone()
        tgt[~mask[:, 1:]] = -100
        return F.cross_entropy(logits.transpose(1, 2), tgt, ignore_index=-100)
    if kind == "mod5":
        _, seq, lab, _ = batch
        return F.cross_entropy(get(model(seq))[:, -1], lab)
    if kind == "chain":
        _, seq, lab, pos = batch
        return step_token_loss(get(model(seq)), pos, lab)
    if kind == "bsteps":
        _, seq, lab, pos = batch
        return step_token_loss(get(model(seq)), pos, lab)
    _, seq, states, _ = batch  # s5: supervise every prefix state
    logits = get(model(seq))
    return F.cross_entropy(logits.transpose(1, 2), states)


def step_token_loss(logits, step_pos, step_labels):
    """Cross-entropy over supervised step positions. step_pos (B,S) uses -1
    pad, step_labels (B,S) uses -100 pad; both are ignored."""
    valid = (step_pos >= 0) & (step_labels >= 0)
    if not valid.any():
        zero = logits.sum() * 0.0
        # Guard (brief deviation): without it, empty.backward() raises when
        # logits lack grad; no-op when logits require grad (normal training).
        return zero if zero.requires_grad else zero.requires_grad_()
    idx = step_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, logits.size(-1))
    return F.cross_entropy(logits.gather(1, idx)[valid], step_labels[valid])


def first_error_position(pred, true, valid):
    """First step index where pred != true (-1 if all valid steps match)."""
    wrong = (pred != true) & valid
    first = wrong.to(torch.int64).argmax(dim=1)
    return torch.where(wrong.any(dim=1), first, torch.full_like(first, -1))


@torch.no_grad()
def val_score(model, task, seed_base=0):
    model.eval()
    get = lambda o: o[0] if isinstance(o, tuple) else o
    if task == "parity":
        tot_l = tot_a = 0.0
        for L in (32, 36, 40):
            s, l = T.make_parity_batch(256, L, T.DATA_SEED_VAL + L)
            lg = get(model(s))
            tot_l += F.cross_entropy(lg[:, -1], l).item()
            tot_a += (lg[:, -1].argmax(-1) == l).float().mean().item()
        return tot_l / 3, tot_a / 3 * 100
    if task in ("s5-standard", "s5-swaps"):
        v = "standard" if task == "s5-standard" else "swaps"
        s, st = T.s5_batch(256, 32, v, T.DATA_SEED_VAL)
        lg = get(model(s))
        loss = F.cross_entropy(lg.transpose(1, 2), st).item()
        acc = (lg.argmax(-1)[:, -1] == st[:, -1]).float().mean().item() * 100
        return loss, acc
    # mod-5 / chain / addition: single pooled val loss proxy
    rng = random.Random(T.DATA_SEED_VAL)
    b = sample_train_batch(task, 64, rng)
    return batch_loss(model, task, b).item(), 0.0


def test_lengths(task):
    if task == "parity":
        return [32, 40, 48, 64, 96, 128, 192, 256]
    if task == "addition":
        return [9, 10, 11, 12, 14, 16, 32]
    if task == "mod5-flat":
        return [31, 39, 47, 63, 95, 127, 191, 255]
    if task == "mod5-bracketed":
        return [32, 40, 48, 64, 96, 128, 192, 256]
    if task == "chain-mod5":
        return [1, 2, 3, 4, 6, 8]
    if task == "bracketed-steps":
        return [16, 24, 32, 48, 64, 96]
    return [32, 48, 64, 96, 128, 192, 256, 512]


@torch.no_grad()
def test_acc(model, task, length):
    get = lambda o: o[0] if isinstance(o, tuple) else o
    model.eval()
    if task == "parity":
        s, l = T.make_parity_batch(1024, length, T.DATA_SEED_TEST + length)
        return (get(model(s))[:, -1].argmax(-1) == l).float().mean().item() * 100
    if task in ("s5-standard", "s5-swaps"):
        v = "standard" if task == "s5-standard" else "swaps"
        s, st = T.s5_batch(1024, length, v, T.DATA_SEED_TEST + length)
        return (get(model(s)).argmax(-1)[:, -1] == st[:, -1]).float().mean().item() * 100
    if task == "chain-mod5":
        s, l, p = T.chain_mod5_batch(1024, length, T.DATA_SEED_TEST + length)
        lg = get(model(s))
        last_idx = (p != -1).sum(1) - 1
        pred = lg[torch.arange(1024), p[torch.arange(1024), last_idx]].argmax(-1)
        return (pred == l[torch.arange(1024), last_idx]).float().mean().item() * 100
    if task == "bracketed-steps":
        s, l, p = T.bracketed_steps_batch(1024, length, T.DATA_SEED_TEST + length)
        lg = get(model(s))
        last_idx = (p != -1).sum(1) - 1
        pred = lg[torch.arange(1024), p[torch.arange(1024), last_idx]].argmax(-1)
        return (pred == l[torch.arange(1024), last_idx]).float().mean().item() * 100
    return float("nan")  # addition/mod5 grids need seq-gen harness; val only in smoke


def run(task, split, total_steps, global_batch, micro_batch, dim, heads, init_seed, ckpt_dir,
        variant="rlt1", chunk_size=None, activation="swiglu"):
    import time
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    enc, dec = map(int, split.split("+"))
    vocab = vocab_of(task)
    tag = variant if variant != "rlt2" else f"rlt2-b{chunk_size}"
    tag = f"{tag}-{activation}" if activation != "swiglu" else tag
    get = lambda o: o[0] if isinstance(o, tuple) else o
    out = {}
    for name, model in zip((f"plain-{enc+dec}", f"{tag}-{split}"),
                           build(dim, heads, enc, dec, vocab, variant=variant, chunk_size=chunk_size,
                                 activation=activation)):
        opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=0.1)
        rng = random.Random(T.DATA_SEED_TRAIN + init_seed)
        best, best_state = float("inf"), None
        accum = max(1, global_batch // micro_batch)
        step_times = []
        sys_samples = []
        for step in tqdm(range(total_steps), desc=f"{task}/{name}/s{init_seed}", leave=False):
            t0 = time.perf_counter()
            model.train()
            opt.zero_grad()
            for _ in range(accum):
                (batch_loss(model, task, sample_train_batch(task, micro_batch, rng)) / accum).backward()
            for g in opt.param_groups:
                g["lr"] = lr_schedule(step, total_steps)
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_times.append(time.perf_counter() - t0)
            if (step + 1) % 100 == 0 or step == total_steps - 1:
                vl, _ = val_score(model, task)
                sys_samples.append(system_stats())
                if vl < best:
                    best = vl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        gen = {L: test_acc(model, task, L) for L in test_lengths(task)}
        # official averages steps 1001-2000; fall back to overall mean on short runs
        timed = step_times[1000:2000] if total_steps > 1000 else step_times
        mean_step = sum(timed) / max(1, len(timed))
        out[name] = {"best_val": best, "generalization": gen, "mean_step_s": mean_step,
                     "system": _agg_sys(sys_samples)}
        if task in ("chain-mod5", "bracketed-steps"):
            with torch.no_grad():
                probe_len = 4 if task == "chain-mod5" else 16
                batch_fn = T.chain_mod5_batch if task == "chain-mod5" else T.bracketed_steps_batch
                s, l, p = batch_fn(1024, probe_len, T.DATA_SEED_TEST + probe_len)
                lg = get(model(s))
                valid = (p != -1) & (l != -100)
                pred = lg.gather(1, p.clamp(min=0).unsqueeze(-1).expand(-1, -1, lg.size(-1))).argmax(-1)
                fe = first_error_position(pred, l.clamp(min=0), valid)
                err = fe[fe >= 0]
                out[name]["first_error_pos"] = err.float().mean().item() if len(err) else None
        if task == "parity":
            with torch.no_grad():
                rob = {}
                for pr in (0.0, 0.01, 0.05, 0.1):
                    s, l = T.make_parity_batch(1024, 32, T.DATA_SEED_TEST + 32)
                    sn, ln = T.add_bitflip_noise(s, l, random.Random(T.DATA_SEED_TEST), pr)
                    rob[str(pr)] = (get(model(sn))[:, -1].argmax(-1) == ln).float().mean().item() * 100
                out[name]["robustness"] = rob
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / f"{task}-{name}-seed{init_seed}.pt")
    return out


def main(task="parity", split="5+3", total_steps=2000, global_batch=512,
         micro_batch=32, dim=512, heads=4, smoke=False,
         variant="rlt1", chunk_size=None, activation="swiglu",
         out="./logs/official.json"):
    assert task in TASKS, task
    assert variant in ("rlt1", "rlt0", "rlt2"), variant
    assert activation in ("swiglu", "gelu"), activation
    if smoke:
        total_steps, global_batch, micro_batch, dim, heads = 20, 32, 8, 64, 4
    if task.startswith("mod5") and not smoke and total_steps == 2000:
        total_steps = 5000  # official mod-5 budget
    per_seed = [run(task, split, total_steps, global_batch, micro_batch, dim, heads,
                    s, Path("./checkpoints/official"),
                    variant=variant, chunk_size=chunk_size,
                    activation=activation) for s in T.INIT_SEEDS]
    summary = {}
    names = per_seed[0].keys()
    for n in names:
        summary[n] = {}
        for L in test_lengths(task):
            vals = [r[n]["generalization"][L] for r in per_seed]
            m, sd = T.mean_sample_sd([v for v in vals if v == v])  # drop NaN
            summary[n][L] = {"mean": m, "sd": sd}
    steps = {n: sum(r[n]["mean_step_s"] for r in per_seed) / len(per_seed) for n in names}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"task": task, "split": split, "variant": variant,
                                     "chunk_size": chunk_size, "activation": activation,
                                     "per_seed": per_seed,
                                     "mean_step_s": steps,
                                     "summary": {k: {str(k2): v2 for k2, v2 in v.items()} for k, v in summary.items()}}, indent=2))
    print(f"\n{task} {split} {variant} (mean +- SD over {T.INIT_SEEDS}):")
    for L in test_lengths(task):
        print(f"len {L:3d}: " + "  ".join(f"{n} {summary[n][L]['mean']:5.1f}+-{summary[n][L]['sd']:4.1f}" for n in names))
    print("mean step s: " + "  ".join(f"{n} {steps[n]:.2f}" for n in names))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
