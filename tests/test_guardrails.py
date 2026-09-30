"""Run guardrails: config validation and wall-clock-capped early exit."""
import pytest


def test_validate_config_rejects_bad_configs():
    from train_official import validate_config
    good = dict(task="parity", split="5+3", total_steps=2000, global_batch=512,
                micro_batch=32, dim=512, heads=4, variant="rlt1",
                chunk_size=None, activation="swiglu", max_hours=None)
    validate_config(**good)  # must not raise
    bad_dim = dict(good, dim=510)
    with pytest.raises(ValueError, match="dim"):
        validate_config(**bad_dim)
    bad_accum = dict(good, global_batch=500)
    with pytest.raises(ValueError, match="micro"):
        validate_config(**bad_accum)
    bad_steps = dict(good, total_steps=0)
    with pytest.raises(ValueError, match="total_steps"):
        validate_config(**bad_steps)
    bad_task = dict(good, task="nope")
    with pytest.raises(ValueError, match="task"):
        validate_config(**bad_task)


def test_zero_time_cap_exits_clean_with_partial_flag(monkeypatch):
    from pathlib import Path
    import train_official as TO
    monkeypatch.setattr(TO, "test_lengths", lambda task: [32])
    monkeypatch.setattr(TO, "test_acc", lambda model, task, length: 50.0)
    out = TO.run("parity", "5+3", 20, 32, 8, 64, 4, 42, Path("/tmp/opencode/guardckpt"),
                 variant="rlt1", max_hours=0)
    for name, r in out.items():
        assert r["completed"] is False
        assert r["steps_done"] == 0
        assert r["best_val"] is None
