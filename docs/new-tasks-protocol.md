# New-tasks protocol (scope expansion, Tasks 1–6 + bracketed wiring)

Data seeds (`official_tasks.py`): train 20260914 / val 20260915 /
test 20260916; init seeds 42, 43, 44. Pads: positions `-1`, labels `-100`.

## chain-mod5 (Task 1: generator, Task 2: step harness, Task 5: wiring)
- `sample_chain_mod5(rng, n_ops)` → `(toks, step_labels, step_positions)`;
  `chain_mod5_batch(batch_size, n_ops, seed)` → `(seq, lab, pos)`.
- Train length `n_ops=4`; test op counts `[1, 2, 3, 4, 6, 8]`.
- Loss `step_token_loss(logits, step_pos, step_labels)`; metric
  `first_error_position(pred, true, valid)` (`-1` = all match).
- Smoke: `.venv/bin/python train_official.py --task chain-mod5 --split 5+3 --smoke --out ./logs/chain_smoke.json`

## bracketed-steps (Task 3: generator; wired like chain-mod5)
- `eval_bracketed(toks)`; `sample_bracketed_steps(rng, target_len)` → `(toks, pos, lab)`;
  `bracketed_steps_batch(batch_size, target_len, seed)` → `(seq, lab, pos)`.
- Train `target_len=16`; test lengths `[16, 24, 32, 48, 64, 96]`; same
  step loss and first-error probe as chain-mod5. Uses only the new
  builder (never the broken `encode_bracketed_mod5`).
- Smoke: `.venv/bin/python train_official.py --task bracketed-steps --split 5+3 --smoke --out ./logs/bsteps_smoke.json`
- Smoke artifact (not a bug): untrained RLT echoes input tokens, and the
  evaluated last step always sits on a `)` position, so rlt1 smoke reads
  0.0 while leaf steps score (see `first_error_pos`). Re-check at
  paper scale.

## parity-noise (Task 4: perturbation, Task 5: noise curves)
- `add_bitflip_noise(seq, labels, rng, p)` (`p <= 0` = identity; labels
  recomputed as `noisy.sum(-1) % 2`).
- Robustness probe in `run` (parity only): accuracy at `p in (0.0, 0.01, 0.05, 0.1)`,
  1024×L32, seed `DATA_SEED_TEST`.
- Smoke: same command as chain-mod5 with `--task parity`.

## mod5-distractors (Task 4: data-level only)
- `mod5_with_distractors(nums, ops, rng, k)` → `(toks, value)`; appends `k`
  trailing `+0` terms (length `+2*k`, value preserved). No trainer change.

## Summary JSON (`logs/<task>_smoke.json`, written by `main`)
- Per seed × model (`plain-<enc+dec>`, `<tag>-<split>`): `best_val`,
  `generalization` (test-length → accuracy), `mean_step_s`,
  `first_error_pos` (chain-mod5 + bracketed-steps, mean over errored seqs,
  `None` if perfect), `robustness` (parity only, noise-rate → accuracy).
- Top level: `summary` (mean/sd over init seeds per test length),
  `mean_step_s` (averaged over seeds), `per_seed` raw entries.
