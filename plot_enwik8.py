# /// script
# dependencies = [
#   "fire",
#   "matplotlib",
#   "numpy",
# ]
# ///

"""Visualize enwik8 training logs.

Reads the JSONL log written by train_enwik8.py (types: train / val / sample)
and optionally a legacy tqdm log (logs/enwik8_full.log).

Outputs:
  - loss.png: train loss + val loss vs step
  - bpc.png: bits-per-char vs step
  - samples.md: generation samples rendered as markdown
  - summary.json: best/final metrics

Usage:
  python plot_enwik8.py --log-file ./logs/enwik8_train.jsonl --out-dir ./plots/enwik8
  python plot_enwik8.py --log-file ./logs/enwik8_train.jsonl --legacy-log ./logs/enwik8_full.log --out-dir ./plots/enwik8
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import fire
import numpy as np


def read_jsonl(log_file: str | Path):
    train_steps, train_loss = [], []
    val_steps, val_loss = [], []
    samples = []
    run_config = {}
    log_file = Path(log_file)
    if not log_file.exists():
        return train_steps, train_loss, val_steps, val_loss, samples, run_config
    with log_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("type")
            if t == "train":
                train_steps.append(int(r["step"]))
                train_loss.append(float(r["train_loss"]))
            elif t == "val":
                val_steps.append(int(r["step"]))
                val_loss.append(float(r["val_loss"]))
            elif t == "sample":
                samples.append(r)
            elif t == "run_config":
                run_config = r
    return train_steps, train_loss, val_steps, val_loss, samples, run_config


LEGACY_RE = re.compile(r"(\d+)/\d+\s+\[.*?loss=([\d.]+)")

def read_legacy_tqdm(legacy_log: str | Path):
    """Best-effort parse of old tqdm-spammed log. Returns (steps, losses)."""
    steps, losses = [], []
    p = Path(legacy_log)
    if not p.exists():
        return steps, losses
    text = p.read_text(errors = "ignore").replace("\r", "\n")
    for m in LEGACY_RE.finditer(text):
        try:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
        except Exception:
            continue
    # dedupe keeping last loss per step
    dedup: dict[int, float] = {}
    for s, l in zip(steps, losses):
        dedup[s] = l
    steps = sorted(dedup)
    return steps, [dedup[s] for s in steps]


def moving_average(x: list[float], window: int):
    if window <= 1 or len(x) < 2:
        return np.asarray(x, dtype = float)
    w = min(window, len(x))
    kernel = np.ones(w) / w
    return np.convolve(np.asarray(x, dtype = float), kernel, mode = "same")


def main(
    log_file: str = "./logs/enwik8_train.jsonl",
    out_dir: str = "./plots/enwik8",
    legacy_log: str | None = None,
    smooth: int = 21,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents = True, exist_ok = True)

    train_steps, train_loss, val_steps, val_loss, samples, run_config = read_jsonl(log_file)

    legacy_steps, legacy_loss = [], []
    if legacy_log:
        legacy_steps, legacy_loss = read_legacy_tqdm(legacy_log)

    # fallback: if no jsonl yet, use legacy as train curve
    if not train_steps and legacy_steps:
        train_steps, train_loss = legacy_steps, legacy_loss

    if not train_steps and not val_steps:
        print(f"no data found in {log_file}" + (f" or {legacy_log}" if legacy_log else ""))
        print("hint: run train_enwik8.py first (it writes JSONL); legacy tqdm log can be passed via --legacy-log")
        return

    train_bpc = [l / math.log(2) for l in train_loss]
    val_bpc = [l / math.log(2) for l in val_loss]
    train_smooth = moving_average(train_loss, smooth)

    # --- loss plot ---
    fig, ax = plt.subplots(figsize = (9, 5))
    if train_steps:
        ax.plot(train_steps, train_loss, alpha = 0.25, label = "train (raw)")
        ax.plot(train_steps, train_smooth, label = f"train (ma{smooth})")
    if val_steps:
        ax.plot(val_steps, val_loss, "o-", label = "val")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("enwik8 training loss")
    ax.legend()
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi = 150)
    plt.close(fig)

    # --- bpc plot ---
    fig, ax = plt.subplots(figsize = (9, 5))
    if train_steps:
        ax.plot(train_steps, train_bpc, alpha = 0.25, label = "train bpc")
        ax.plot(train_steps, moving_average(train_bpc, smooth), label = f"train bpc (ma{smooth})")
    if val_steps:
        ax.plot(val_steps, val_bpc, "o-", label = "val bpc")
    ax.set_xlabel("step")
    ax.set_ylabel("bits per char")
    ax.set_title("enwik8 bits-per-char (loss / ln 2)")
    ax.legend()
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    fig.savefig(out / "bpc.png", dpi = 150)
    plt.close(fig)

    # --- samples markdown ---
    if samples:
        md = ["# Generation samples\n"]
        for s in samples[-20:]:
            md.append(f"## step {s.get('step')}\n")
            md.append(f"**prompt:** `{s.get('prompt', '')}`\n")
            md.append(f"**output:** `{s.get('output', '')}`\n")
        (out / "samples.md").write_text("\n".join(md))

    # --- summary ---
    summary = {
        "log_file": str(log_file),
        "legacy_log": legacy_log,
        "num_train_points": len(train_steps),
        "num_val_points": len(val_steps),
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_train_bpc": train_bpc[-1] if train_bpc else None,
        "best_val_loss": min(val_loss) if val_loss else None,
        "best_val_bpc": min(val_bpc) if val_bpc else None,
        "best_val_step": int(val_steps[int(np.argmin(val_loss))]) if val_loss else None,
        "final_val_loss": val_loss[-1] if val_loss else None,
        "final_val_ppl": math.exp(val_loss[-1]) if val_loss else None,
        "run_config": run_config,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent = 2))
    print(json.dumps(summary, indent = 2))
    print(f"\nwrote {out / 'loss.png'}, {out / 'bpc.png'}" + (f", {out / 'samples.md'}" if samples else ""))


if __name__ == "__main__":
    fire.Fire(main)
