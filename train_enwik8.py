# /// script
# dependencies = [
#   "rlt-pytorch",
#   "accelerate",
#   "fire",
#   "numpy",
#   "tqdm",
# ]
# ///

from __future__ import annotations

import gzip
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import tqdm
import fire
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from RLT import RLT

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents = True, exist_ok = True)
    return p

def reduce_mean(accelerator: Accelerator, value: torch.Tensor) -> float:
    # single-process fast path, distributed-safe otherwise
    if accelerator.num_processes <= 1:
        return value.detach().float().mean().item()
    gathered = accelerator.gather(value.detach().float())
    return gathered.mean().item()

def prune_checkpoints(checkpoint_dir: Path, keep_last: int | None, separator: str = "step_"):
    if not exists(keep_last) or keep_last <= 0:
        return
    # never prune the checkpoint that ./best points to
    protected: set[Path] = set()
    best_link = checkpoint_dir / "best"
    try:
        if best_link.is_symlink():
            protected.add((checkpoint_dir / best_link.readlink()).resolve())
    except Exception:
        pass
    ckpts = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith(separator)],
        key = lambda p: p.stat().st_mtime
    )
    # filter out protected before choosing victims
    victims = [p for p in ckpts if p.resolve() not in protected]
    for old in victims[:-keep_last] if len(victims) > keep_last else []:
        shutil.rmtree(old, ignore_errors = True)

def save_checkpoint(
    accelerator: Accelerator,
    model,
    checkpoint_dir: Path,
    step: int,
    train_loss: float,
    val_loss: float | None,
    elapsed_s: float,
    keep_last: int | None = 3
):
    ckpt_path = checkpoint_dir / f"step_{step:07d}"
    ckpt_path.mkdir(parents = True, exist_ok = True)

    accelerator.save_state(str(ckpt_path))

    meta = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "elapsed_s": elapsed_s,
        "bpc": (val_loss / math.log(2)) if exists(val_loss) else None,
        "ppl": math.exp(val_loss) if exists(val_loss) else None,
    }
    (ckpt_path / "meta.json").write_text(json.dumps(meta, indent = 2))

    # plain state_dict copy for easy loading without accelerate
    try:
        torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path / "model.pt")
    except Exception as e:
        (ckpt_path / "model_save_warning.txt").write_text(f"plain model.pt not saved: {e}\n")

    prune_checkpoints(checkpoint_dir, keep_last)
    return ckpt_path

def append_jsonl(log_file: Path, record: dict):
    log_file.parent.mkdir(parents = True, exist_ok = True)
    with log_file.open("a") as f:
        f.write(json.dumps(record) + "\n")

# dataset

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

# main training function

def train(
    num_batches: int = 10_000,
    batch_size: int = 4,
    grad_accum_every: int = 2,
    learning_rate: float = 1e-3,
    validate_every: int = 100,
    generate_every: int = 100,
    prime_length: int = 32,
    generate_length: int | None = None,
    seq_len: int = 128,
    dim: int = 256,
    enc_depth: int = 6,
    dec_depth: int = 2,
    dec_sliding_window_size: int = 8,
    recurrent_block_size: int = 16,
    recurrent_transition_alpha: float = 1.,
    filter_thres: float = 0.9,
    temperature: float = 1.,
    cpu: bool = False,
    data_path: str = "./data/enwik8.gz",
    # checkpointing + logging (new)
    checkpoint_dir: str = "./checkpoints/enwik8",
    save_every: int = 500,
    keep_last: int | None = 3,
    save_best: bool = True,
    resume_from: str | None = None,
    log_file: str = "./logs/enwik8_train.jsonl",
    log_every: int = 1,
    seed: int = 42,
    disable_tqdm: bool = False,
):
    generate_length = default(generate_length, seq_len)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    accelerator = Accelerator(cpu = cpu)
    is_main = accelerator.is_main_process

    ckpt_root = Path(checkpoint_dir)
    log_path = Path(log_file)
    if is_main:
        ensure_dir(ckpt_root)
        log_path.parent.mkdir(parents = True, exist_ok = True)
        append_jsonl(log_path, {
            "type": "run_config",
            "num_batches": num_batches,
            "batch_size": batch_size,
            "grad_accum_every": grad_accum_every,
            "learning_rate": learning_rate,
            "seq_len": seq_len,
            "dim": dim,
            "enc_depth": enc_depth,
            "dec_depth": dec_depth,
            "seed": seed,
            "resume_from": resume_from,
        })
        print(f"[config] checkpoints -> {ckpt_root} | log -> {log_path} | save_every={save_every} keep_last={keep_last} save_best={save_best}", flush = True)

    # prepare enwik8 data

    with gzip.open(data_path) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # model & optimizer

    model = RLT(
        num_tokens = 256,
        dim = dim,
        enc_depth = enc_depth,
        dec_depth = dec_depth,
        dec_sliding_window_size = dec_sliding_window_size,
        recurrent_block_size = recurrent_block_size,
        recurrent_transition_alpha = recurrent_transition_alpha
    )

    optim = Adam(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    start_step = 0
    best_val = float("inf")

    if exists(resume_from):
        accelerator.load_state(resume_from)
        meta_file = Path(resume_from) / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                start_step = int(meta.get("step", 0))
                if exists(meta.get("val_loss")):
                    best_val = float(meta["val_loss"])
            except Exception as e:
                if is_main:
                    print(f"[resume] warning: could not read meta.json: {e}", flush = True)
        if is_main:
            print(f"[resume] loaded {resume_from}, resuming from step {start_step}, best_val={best_val:.4f}", flush = True)

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    t0 = time.time()
    last_val_loss: float | None = None

    pbar = tqdm.tqdm(
        range(start_step + 1, num_batches + 1),
        initial = start_step,
        total = num_batches,
        mininterval = 2.0,
        desc = "training",
        disable = disable_tqdm,
        file = sys.stderr,
    )

    for step in pbar:
        model.train()
        total_loss = 0.

        for _ in range(grad_accum_every):
            data = next(train_loader)

            loss = model(data, return_loss = True)

            accelerator.backward(loss / grad_accum_every)
            total_loss += loss.item()

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

        avg_loss = total_loss / grad_accum_every
        elapsed = time.time() - t0

        if not disable_tqdm:
            pbar.set_postfix(loss = f"{avg_loss:.3f}")

        # line-buffered human log (no \r spam) + machine-readable jsonl
        if is_main and divisible_by(step, log_every):
            print(f"[step {step}/{num_batches}] train_loss={avg_loss:.4f} elapsed={elapsed:.0f}s", flush = True)
            append_jsonl(log_path, {
                "type": "train",
                "step": step,
                "train_loss": avg_loss,
                "bpc_train": avg_loss / math.log(2),
                "elapsed_s": elapsed,
            })

        if divisible_by(step, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)
                val_loss_t = model(valid_data, return_loss = True)
                val_loss = reduce_mean(accelerator, val_loss_t)
            last_val_loss = val_loss
            if is_main:
                print(f"[step {step}] val_loss={val_loss:.4f} bpc={val_loss / math.log(2):.4f} ppl={math.exp(val_loss):.2f}", flush = True)
                append_jsonl(log_path, {
                    "type": "val",
                    "step": step,
                    "val_loss": val_loss,
                    "bpc": val_loss / math.log(2),
                    "ppl": math.exp(val_loss),
                    "elapsed_s": elapsed,
                })

        if divisible_by(step, generate_every) or step == num_batches:
            model.eval()
            unwrapped_model = accelerator.unwrap_model(model)

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp)
            if is_main:
                print(f"--- [Step {step}] GENERATION ---", flush = True)
                print(f"PROMPT: {prime}", flush = True)

            sampled = unwrapped_model.generate(
                inp,
                max_len = generate_length,
                temperature = temperature,
                filter_kwargs = dict(thres = filter_thres)
            )

            decoded_output = decode_tokens(sampled)
            if is_main:
                print(f"OUTPUT: {decoded_output}\n", flush = True)
                append_jsonl(log_path, {
                    "type": "sample",
                    "step": step,
                    "prompt": prime,
                    "output": decoded_output,
                })

        # checkpointing
        saved_this_step = False
        if divisible_by(step, save_every) or step == num_batches:
            ckpt = save_checkpoint(
                accelerator, model, ckpt_root, step,
                train_loss = avg_loss,
                val_loss = last_val_loss,
                elapsed_s = elapsed,
                keep_last = keep_last
            )
            saved_this_step = True
            if is_main:
                print(f"[checkpoint] saved {ckpt}", flush = True)
                append_jsonl(log_path, {"type": "checkpoint", "step": step, "path": str(ckpt)})

        if save_best and exists(last_val_loss) and last_val_loss < best_val:
            best_val = last_val_loss
            if not saved_this_step:
                ckpt = save_checkpoint(
                    accelerator, model, ckpt_root, step,
                    train_loss = avg_loss,
                    val_loss = last_val_loss,
                    elapsed_s = elapsed,
                    keep_last = keep_last
                )
            else:
                ckpt = ckpt_root / f"step_{step:07d}"
            # mark a stable pointer to the best checkpoint
            if is_main:
                best_link = ckpt_root / "best"
                try:
                    if best_link.is_symlink() or best_link.exists():
                        if best_link.is_dir() and not best_link.is_symlink():
                            shutil.rmtree(best_link)
                        else:
                            best_link.unlink()
                    best_link.symlink_to(ckpt.name)
                except Exception:
                    # fallback for filesystems without symlink: copy meta
                    shutil.copy(ckpt / "meta.json", ckpt_root / "best_meta.json")
                print(f"[checkpoint] new best val={best_val:.4f} at step {step} -> {ckpt}", flush = True)
                append_jsonl(log_path, {"type": "best", "step": step, "val_loss": best_val, "path": str(ckpt)})

    # final model weights for easy inference
    if is_main:
        try:
            final_weights = ckpt_root / "final.pt"
            torch.save(accelerator.unwrap_model(model).state_dict(), final_weights)
            print(f"[done] saved final weights -> {final_weights}", flush = True)
        except Exception as e:
            print(f"[done] warning: could not save final.pt: {e}", flush = True)

    accelerator.end_training()
    accelerator.print("Training complete!")

if __name__ == "__main__":
    fire.Fire(train)
