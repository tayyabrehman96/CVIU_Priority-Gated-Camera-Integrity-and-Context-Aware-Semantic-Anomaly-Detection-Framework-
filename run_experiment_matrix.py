#!/usr/bin/env python3
"""
Run hybrid risk simulation experiment matrix and aggregate results.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import io
import contextlib
from pathlib import Path
from typing import Dict, List

import numpy as np

from test_pipeline import run_benchmark


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(sample.mean()))
    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1 - alpha / 2))
    return {"mean": float(arr.mean()), "ci_low": low, "ci_high": high}


def run_simulation(mode: str, severity: str, seed: int) -> Path:
    cmd = [
        sys.executable,
        "hybrid_risk_simulation.py",
        "--mode",
        mode,
        "--severity",
        severity,
        "--seed",
        str(seed),
        "--metadata-only",
    ]
    subprocess.run(cmd, check=True)

    sim_root = Path("simulations")
    candidates = sorted(sim_root.glob(f"run_{mode}_{severity}_seed{seed}_*/manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No manifest produced for {mode}/{severity}/seed={seed}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix for risk simulation.")
    parser.add_argument("--sample-frames", type=int, default=5)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--no-models", action="store_true", help="Skip YOLO loading in benchmark runs.")
    args = parser.parse_args()

    modes = ["dataset_only", "synthetic_only", "hybrid"]
    severities = ["low", "high"]
    seeds = [13, 23, 37]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_rows = []

    for mode in modes:
        for severity in severities:
            for seed in seeds:
                manifest = run_simulation(mode, severity, seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    res = run_benchmark(
                        videos_dir=str(manifest.parent / "videos"),
                        sample_every_n=args.sample_frames,
                        verbose=False,
                        manifest_path=str(manifest),
                        load_models=not args.no_models,
                    )
                print(f"[ok] evaluated mode={mode}, severity={severity}, seed={seed}")
                run_rows.append(
                    {
                        "mode": mode,
                        "severity": severity,
                        "seed": seed,
                        "accuracy": res["accuracy"],
                        "macro_f1": res["macro_f1"] * 100.0,
                        "scenario_coverage": res["risk_quality"]["scenario_coverage"] * 100.0,
                        "severity_balance": res["risk_quality"]["severity_balance"] * 100.0,
                        "temporal_stability": res["risk_quality"]["temporal_stability"] * 100.0,
                        "failure_case_rate": res["risk_quality"]["failure_case_rate"] * 100.0,
                        "report_file": res["report_file"],
                        "manifest": str(manifest),
                    }
                )

    raw_csv = out_dir / f"risk_matrix_raw_{stamp}.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    summary_rows = []
    for mode in modes:
        for severity in severities:
            subset = [r for r in run_rows if r["mode"] == mode and r["severity"] == severity]
            acc_ci = bootstrap_ci([r["accuracy"] for r in subset])
            f1_ci = bootstrap_ci([r["macro_f1"] for r in subset])
            tq_ci = bootstrap_ci([r["temporal_stability"] for r in subset])
            summary_rows.append(
                {
                    "mode": mode,
                    "severity": severity,
                    "accuracy_mean": acc_ci["mean"],
                    "accuracy_ci_low": acc_ci["ci_low"],
                    "accuracy_ci_high": acc_ci["ci_high"],
                    "macro_f1_mean": f1_ci["mean"],
                    "macro_f1_ci_low": f1_ci["ci_low"],
                    "macro_f1_ci_high": f1_ci["ci_high"],
                    "temporal_stability_mean": tq_ci["mean"],
                    "temporal_stability_ci_low": tq_ci["ci_low"],
                    "temporal_stability_ci_high": tq_ci["ci_high"],
                }
            )

    summary_csv = out_dir / f"risk_matrix_summary_{stamp}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    best = max(summary_rows, key=lambda x: x["macro_f1_mean"])
    markdown = out_dir / f"risk_results_{stamp}.md"
    with markdown.open("w", encoding="utf-8") as f:
        f.write("# Hybrid Risk Simulation Results\n\n")
        f.write("## Matrix design\n")
        f.write("- Modes: dataset_only, synthetic_only, hybrid\n")
        f.write("- Severities: low, high\n")
        f.write("- Seeds: 13, 23, 37\n\n")
        f.write("## Best setting by macro F1\n")
        f.write(
            f"- mode={best['mode']}, severity={best['severity']}, "
            f"macro_f1={best['macro_f1_mean']:.2f}% "
            f"(95% CI: {best['macro_f1_ci_low']:.2f} to {best['macro_f1_ci_high']:.2f})\n"
        )
        f.write(
            f"- accuracy={best['accuracy_mean']:.2f}% "
            f"(95% CI: {best['accuracy_ci_low']:.2f} to {best['accuracy_ci_high']:.2f})\n"
        )
        f.write("\n## Files\n")
        f.write(f"- Raw runs: `{raw_csv}`\n")
        f.write(f"- Summary: `{summary_csv}`\n")

    print(f"[ok] Raw matrix: {raw_csv}")
    print(f"[ok] Summary matrix: {summary_csv}")
    print(f"[ok] Markdown summary: {markdown}")


if __name__ == "__main__":
    main()
