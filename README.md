<img src="./rlt.png" width="400"></img>

# RLT — Official-Protocol Reproduction & Scope Expansion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](./pyproject.toml)
[![CI](https://github.com/mahoutei/RLT/actions/workflows/ci.yml/badge.svg)](https://github.com/mahoutei/RLT/actions)

Unofficial research reproduction of the [Recurrent Looped Transformer](https://yifanzhang-pro.github.io/recurrent-looped-tranformer/) (Zhang, Princeton) following the paper's official evaluation protocol (App. J): fixed data streams, init seeds 42/43/44, best-ID-loss checkpointing, plain-Transformer baselines — plus new compositional and robustness tasks that stress latent recurrence beyond length generalization.

> Forked from [`lucidrains/RLT`](https://github.com/lucidrains/RLT) (model code). All training/evaluation harnesses here are original to this repo.

## Contents

- [Status](#status) · [Tasks](#tasks) · [Prerequisites](#prerequisites) · [Quickstart](#quickstart) · [Repo map](#repo-map) · [Results](#results) · [Package usage](#package-usage) · [Appreciation](#appreciation) · [Citations](#citations)

## Status

- [x] 6 official algorithmic tasks (parity, addition, mod-5 flat/bracketed, S5 standard/swaps) with depth-8 splits
- [x] Compositional chain tasks with per-step supervision (`chain-mod5`, bracketed steps)
- [x] Robustness variants (parity bit-flip noise curves, mod-5 distractors)
- [x] Colab notebook (GPU + Drive) and enwik8 language-modeling trainer
- [ ] Paper-scale runs (currently smoke-validated; full summaries land in `logs/`)

## Tasks

| Family | Tasks | Train → test | Metric |
|--------|-------|--------------|--------|
| Official | parity, addition, mod5-flat, mod5-bracketed, s5-standard, s5-swaps | short → long (e.g. bits 3–40 → 256) | final-answer accuracy |
| Compositional (new) | chain-mod5, bracketed steps | 1–4 ops → 6–8 ops | step accuracy + first-error position |
| Robustness (new) | parity noise, mod-5 distractors | clean → noisy | accuracy-vs-noise curves |

Details per task: [`docs/new-tasks-protocol.md`](./docs/new-tasks-protocol.md).

## Prerequisites

- Python ≥ 3.10, PyTorch (CUDA build for paper-scale runs; CPU suffices for smoke)
- `data/enwik8.gz` for language-modeling runs (place it under `data/`)
- Google Colab with GPU runtime for the notebook path (Drive holds logs/checkpoints)

## Quickstart

```bash
pip install -e .          # or: pip install rlt-pytorch
pip install x-transformers tqdm fire   # trainers
```

```bash
# Smoke check (CPU, minutes)
python train_official.py --task parity --split 5+3 --smoke
python train_official.py --task chain-mod5 --split 5+3 --smoke

# Paper scale (GPU): d=512, 2000 steps (5000 for mod-5), batch 512
python train_official.py --task parity --split 5+3
python train_parity.py --split 5+3
```

Colab (GPU + Drive persistence): run `RLT_official_colab.ipynb` — set `SMOKE = True` first, then `False` for full runs.

```bash
pytest tests/ -q   # unit + protocol tests
```

## Repo map

| Path | What |
|------|------|
| `RLT/` | Model code (upstream) |
| `official_tasks.py` | Data generators: 6 official + chain/robustness tasks, fixed seeds |
| `train_official.py` | Generic official-protocol trainer (all tasks, all splits) |
| `train_parity.py` | Parity experiment script |
| `train_enwik8.py` | Enwik8 language-modeling trainer (accelerate) |
| `RLT_official_colab.ipynb` | Monolithic Colab notebook (model + tasks + trainer embedded) |
| `docs/new-tasks-protocol.md` | Per-task protocol notes (seeds, commands, outputs) |

## Results

Smoke validation (tiny CPU configs — chance-level is expected; proves the pipeline, not the paper):

| Run | plain | rlt1 |
|-----|-------|------|
| parity 1+1, 256-bit (chance ~50%) | 47.8 | 47.8 |
| chain-mod5 5+3, 8 ops (chance 20%) | 8.8 | 27.4 |

Full paper-scale numbers will be tabulated here from `logs/*_summary.json` (mean ± SD over seeds 42/43/44). Reference targets: parity 5+3/7+1 → 100% at 256 bits vs Transformer ~50%; swaps-S5 4+4 → ~56% final-state at 512 ops.

## Package usage

```python
import torch
from RLT import RLT

model = RLT(num_tokens=256, dim=512, enc_depth=4, dec_depth=4,
            dec_sliding_window_size=16, tbptt_step_size=16)
tokens = torch.randint(0, 256, (2, 1024))
loss = model(tokens, return_loss=True)
loss.backward()
```

Optional [Next-Latent Prediction](https://arxiv.org/abs/2511.05963): `RLT(..., next_lat_loss=True)`. Variable recurrent blocks via `recurrent_lengths`. See upstream README history for details.

## Appreciation

- [Pranoy](https://github.com/pranoyr) for the PR on weight tying / sharing between encoder and decoder (upstream)!

## Citations

```bibtex
@techreport{zhang2026recurrentlooped,
    title  = {Recurrent Looped Transformer},
    author = {Zhang, Yifan},
    year   = {2026},
    month  = {Sep},
    url    = {https://github.com/yifanzhang-pro/recurrent-looped-tranformer}
}
```

```bibtex
@misc{oncescu2026recurrenttransformergreatereffective,
    title     = {The Recurrent Transformer: Greater Effective Depth and Efficient Decoding},
    author    = {Costin-Andrei Oncescu and Depen Morwani and Samy Jelassi and Alexandru Meterez and Mujin Kwun and Sham Kakade},
    year      = {2026},
    eprint    = {2604.21215},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url       = {https://arxiv.org/abs/2604.21215},
}
```

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```

```bibtex
@misc{teoh2025nextlatentpredictiontransformerslearn,
    title     = {Next-Latent Prediction Transformers Learn Compact World Models},
    author    = {Jayden Teoh and Manan Tomar and Kwangjun Ahn and Edward S. Hu and Tim Pearce and Pratyusha Sharma and Akshay Krishnamurthy and Riashat Islam and Alex Lamb and John Langford},
    year      = {2025},
    eprint    = {2511.05963},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url       = {https://arxiv.org/abs/2511.05963}
}
```

```bibtex
@misc{wang2026fullbandwidthtransformer,
    title         = {Full-bandwidth transformer},
    author        = {Xi Wang and Ziyang Cai and Zheng Zhan and Harry Dong and Ying Fan and Gustavo de Rosa and Tim Pearce and John Langford},
    year          = {2026},
    eprint        = {2608.08888},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url       = {https://arxiv.org/abs/2608.08888}
}
```
