# RLT Scope Expansion — Design (2026-09-26)

## Intent

Expand the RLT reproduction beyond the 6 official algorithmic tasks with new
task families that stress latent recurrence, evaluated under styles beyond
length-generalization. Headline: compositional chains with process
supervision (B). Supporting: robustness/shift variants (C).

Success: new tasks run under the same experimental discipline as the
official protocol (fixed data streams, init seeds 42/43/44, plain-Transformer
baseline, best-checkpoint selection) with CPU-runnable smoke configs.

## Task families

### B — Compositional chains with step labels (headline)

2–3 tasks where every intermediate reasoning step has a ground-truth label,
e.g. chained modular arithmetic and bracketed-expression evaluation.
Tests whether latent iteration complements explicit steps — the core RLT
claim. Requires a new seq-gen harness emitting per-step labels.

### C — Robustness variants (supporting)

Perturbations of existing parity/mod-5 generators: distractor tokens,
symbol permutation, input noise. Small deltas on current code. Asks whether
recurrence buys robustness, not just length generalization.

## Evaluation

- B: step-level accuracy + final-answer accuracy + first-error position.
- C: accuracy-under-noise/perturbation curves.
- Both: fixed data seeds, init seeds 42/43/44, plain-Transformer baseline,
  best-checkpoint selection (same discipline as `train_official.py`).

## Code shape

- New generators next to `official_tasks.py`, same conventions (explicit
  seeds, minimal token schemes, fixed train/val/test splits).
- New step-label harness module for process supervision: current
  `batch_loss` supervises final/prefix tokens only and cannot express
  per-step labels.
- Training entry extends `train_official.py`; no second parallel trainer.
- Notebook (`RLT_official_colab.ipynb`) and scripts stay in sync via one
  short protocol note per new task.

## Validation

- Smoke config (tiny dim/steps) for every new task, CPU-runnable, before
  any paper-scale run.
- Determinism check: same seed produces identical batches.
- New tasks must not change existing tasks' data streams or results.

## Out of scope

- Stateful per-prefix tracking tasks beyond S5 (fallback only if B's
  harness proves too heavy).
- Changes to the RLT model itself or the official 6-task numbers.
