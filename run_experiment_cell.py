#!/usr/bin/env python3
import argparse
import csv
import io
import contextlib
from pathlib import Path

from test_pipeline import run_benchmark
from run_experiment_matrix import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one matrix cell.")
    parser.add_argument("--mode", required=True, choices=["dataset_only", "synthetic_only", "hybrid"])
    parser.add_argument("--severity", required=True, choices=["low", "high"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-frames", type=int, default=50)
    parser.add_argument("--out-file", default="reports/risk_matrix_cells.csv")
    parser.add_argument("--no-models", action="store_true")
    args = parser.parse_args()

    manifest = run_simulation(args.mode, args.severity, args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_benchmark(
            videos_dir=str(manifest.parent / "videos"),
            sample_every_n=args.sample_frames,
            verbose=False,
            manifest_path=str(manifest),
            load_models=not args.no_models,
        )

    row = {
        "mode": args.mode,
        "severity": args.severity,
        "seed": args.seed,
        "accuracy": res["accuracy"],
        "macro_f1": res["macro_f1"] * 100.0,
        "scenario_coverage": res["risk_quality"]["scenario_coverage"] * 100.0,
        "severity_balance": res["risk_quality"]["severity_balance"] * 100.0,
        "temporal_stability": res["risk_quality"]["temporal_stability"] * 100.0,
        "failure_case_rate": res["risk_quality"]["failure_case_rate"] * 100.0,
        "report_file": res["report_file"],
        "manifest": str(manifest),
    }

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_file.exists()
    with out_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[ok] appended cell: {args.mode}/{args.severity}/seed={args.seed}")


if __name__ == "__main__":
    main()
