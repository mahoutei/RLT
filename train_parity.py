# /// script
# dependencies = [
#   "rlt-pytorch",
#   "x-transformers",
#   "tqdm",
# ]
# ///
"""Parity experiment aligned to the official RLT protocol.

Official (depth-8 snapshot, App. J):
  width 512, FFN 1365, 4 heads, SWA 8, G=1, alpha=0.1, TBPTT 128
  global batch 512, micro 32, AdamW(0.9,0.95) wd=0.1 clip=1.0
  warmup 200 -> 1e-4, cosine -> 5e-6 at step 2000
  train lens 3-40, val {32,36,40} pooled, test to 256 bits
  init seeds 42,43,44 on fixed data stream; select best ID-loss checkpoint

`--smoke` runs a tiny CPU check (dim 64, few steps). Default args are the
paper values; lower them manually on small machines.
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

from RLT import RLT
from official_tasks import (
    DATA_SEED_TRAIN,
    INIT_SEEDS,
    make_parity_batch,
    mean_sample_sd,
)

VAL_LENGTHS = (32, 36, 40)
TEST_LENGTHS = (32, 40, 48, 64, 96, 128, 192, 256)


def lr_schedule(step: int, total: int, base_lr: float = 1e-4, min_lr: float = 5e-6, warmup: int = 200):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


def build_models(dim=512, heads=4, enc_depth=5, dec_depth=3, swa=8, alpha=0.1, tbptt=128):
    dim_head = dim // heads
    ffn_mult = 1365 / 512  # paper FFN width at d=512; scale linearly otherwise
    plain = TransformerWrapper(
        num_tokens=2,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=dim, depth=enc_depth + dec_depth, heads=heads,
            attn_dim_head=dim_head, rotary_pos_emb=True,
            rotary_emb_dim=min(dim_head, 64),
            use_rmsnorm=True, ff_glu=True,
            ff_mult=ff_mult,
        ),
    )
    rlt = RLT(
        num_tokens=2, dim=dim, enc_depth=enc_depth, dec_depth=dec_depth,
        heads=heads, dim_head=dim_head,
        dec_sliding_window_size=swa,
        recurrent_transition_alpha=alpha,
        tbptt_step_size=tbptt,
        num_kv_layers=1,  # G=1 shared encoder memory
        ff_expansion_factor=ff_mult,
    )
    return plain, rlt


def final_token_loss(model, seq, labels):
    out = model(seq)
    logits = out[0] if isinstance(out, tuple) else out
    return F.cross_entropy(logits[:, -1], labels)


@torch.no_grad()
def final_acc(model, seq, labels):
    model.eval()
    out = model(seq)
    logits = out[0] if isinstance(out, tuple) else out
    return (logits[:, -1].argmax(-1) == labels).float().mean().item() * 100


def run_seed(
    init_seed: int, split=(5, 3), total_steps=2000, global_batch=512,
    micro_batch=32, dim=512, heads=4, ckpt_dir: Path | None = None,
):
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    enc_depth, dec_depth = split
    plain, rlt = build_models(dim=dim, heads=heads, enc_depth=enc_depth, dec_depth=dec_depth)
    models = {"plain-8": plain, f"rlt-{enc_depth}+{dec_depth}": rlt}
    results = {}
    for name, model in models.items():
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
        best_val, best_state = float("inf"), None
        accum = global_batch // micro_batch
        step_rng = random.Random(DATA_SEED_TRAIN + init_seed)
        for step in tqdm(range(total_steps), desc=f"{name}/seed{init_seed}", leave=False):
            model.train()
            opt.zero_grad()
            for _ in range(accum):
                L = step_rng.randint(3, 40)
                seq, lab = make_parity_batch(micro_batch, L, seed=step_rng.randint(0, 2**31 - 1))
                (final_token_loss(model, seq, lab) / accum).backward()
            # official schedule
            lr = lr_schedule(step, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step + 1) % 100 == 0 or step == total_steps - 1:
                model.eval()
                vloss, vacc = 0.0, 0.0
                for L in VAL_LENGTHS:
                    seq, lab = make_parity_batch(256, L, seed=DATA_SEED_TRAIN + 1000 + L)
                    with torch.no_grad():
                        out = model(seq)
                        logits = out[0] if isinstance(out, tuple) else out
                        vloss += F.cross_entropy(logits[:, -1], lab).item()
                        vacc += (logits[:, -1].argmax(-1) == lab).float().mean().item()
                vloss /= len(VAL_LENGTHS)
                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        # length generalization on shared test sets (1024 seqs/length, test seed)
        gen = {}
        for L in TEST_LENGTHS:
            seq, lab = make_parity_batch(1024, L, seed=20260916 + L)
            gen[L] = final_acc(model, seq, lab)
        results[name] = {"best_val": best_val, "generalization": gen}
        if ckpt_dir is not None:
            (ckpt_dir / f"{name}-seed{init_seed}.pt").parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_dir / f"{name}-seed{init_seed}.pt")
    return results


def main(
    split: str = "5+3",
    total_steps: int = 2000,
    global_batch: int = 512,
    micro_batch: int = 32,
    dim: int = 512,
    heads: int = 4,
    smoke: bool = False,
    out: str = "./logs/parity_official.json",
):
    if smoke:  # tiny CPU sanity check
        total_steps, global_batch, micro_batch, dim, heads = 20, 32, 8, 64, 4
    enc, dec = map(int, split.split("+"))
    all_seeds = []
    for s in INIT_SEEDS:
        all_seeds.append(run_seed(s, split=(enc, dec), total_steps=total_steps,
                                 global_batch=global_batch, micro_batch=micro_batch,
                                 dim=dim, heads=heads,
                                 ckpt_dir=Path("./checkpoints/parity_official")))
    # aggregate mean +- sample SD over inits (App. J.3)
    summary = {}
    for name in all_seeds[0]:
        summary[name] = {}
        for L in TEST_LENGTHS:
            m, sd = mean_sample_sd([r[name]["generalization"][L] for r in all_seeds])
            summary[name][L] = {"mean": m, "sd": sd}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"split": split, "per_seed": all_seeds, "summary": summary}, indent=2))
    print(f"\nparity generalization (mean +- SD over {INIT_SEEDS}):")
    for L in TEST_LENGTHS:
        row = "  ".join(f"{n} {summary[n][L]['mean']:5.1f}+-{summary[n][L]['sd']:4.1f}" for n in summary)
        print(f"len {L:3d}: {row}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
