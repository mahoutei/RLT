"""Reporting helpers: system stats sampling and GPU-trace plotting."""
import json


def test_system_stats_shape_and_agg():
    from train_official import system_stats, _agg_sys
    s = system_stats()
    assert isinstance(s, dict)
    assert set(s) <= {"cpu_pct", "ram_gb", "gpu_util_pct", "gpu_mem_mb"}
    agg = _agg_sys([{"gpu_mem_mb": 100.0, "cpu_pct": 10.0},
                    {"gpu_mem_mb": 300.0, "cpu_pct": 30.0}])
    assert agg["gpu_mem_mb"] == {"avg": 200.0, "peak": 300.0}
    assert agg["gpu_util_pct"] is None
    assert agg["num_samples"] == 2


def test_plot_reads_gputrace_and_writes_gpu_figure(tmp_path):
    import plot_enwik8 as P
    log = tmp_path / "t.jsonl"
    rows = [
        {"type": "train", "step": 1, "train_loss": 2.0,
         "system": {"gpu_mem_mb": 1000.0, "gpu_util_pct": 50.0}},
        {"type": "gputrace", "step": 1, "mem_alloc_mb": 800.0, "mem_reserved_mb": 900.0},
        {"type": "gputrace", "step": 2, "mem_alloc_mb": 850.0, "mem_reserved_mb": 900.0},
        {"type": "val", "step": 2, "val_loss": 1.9},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "plots"
    P.main(log_file=str(log), out_dir=str(out), smooth=1)
    assert (out / "gpu.png").exists()
    html = (out / "report.html").read_text()
    assert "avg=1000.0 peak=1000.0" in html
    d = json.loads((out / "summary.json").read_text())
    assert d["system"]["trace_mem_alloc_peak_mb"] == 850.0
    assert d["system"]["trace_points"] == 2
