#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

from run_experiment_matrix import bootstrap_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matrix cell results.")
    parser.add_argument("--cells-file", default="reports/risk_matrix_cells.csv")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    cells = []
    with open(args.cells_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cells.append(r)

    modes = ["dataset_only", "synthetic_only", "hybrid"]
    severities = ["low", "high"]
    summary_rows = []
    for mode in modes:
        for severity in severities:
            subset = [r for r in cells if r["mode"] == mode and r["severity"] == severity]
            acc_ci = bootstrap_ci([float(r["accuracy"]) for r in subset])
            f1_ci = bootstrap_ci([float(r["macro_f1"]) for r in subset])
            tq_ci = bootstrap_ci([float(r["temporal_stability"]) for r in subset])
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_csv = out_dir / f"risk_matrix_summary_{stamp}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    best = max(summary_rows, key=lambda x: x["macro_f1_mean"])
    md = out_dir / f"risk_results_{stamp}.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Hybrid Risk Simulation Results\n\n")
        f.write(f"- Best mode: {best['mode']}\n")
        f.write(f"- Severity: {best['severity']}\n")
        f.write(f"- Macro F1: {best['macro_f1_mean']:.2f}% (95% CI {best['macro_f1_ci_low']:.2f} to {best['macro_f1_ci_high']:.2f})\n")
        f.write(f"- Accuracy: {best['accuracy_mean']:.2f}% (95% CI {best['accuracy_ci_low']:.2f} to {best['accuracy_ci_high']:.2f})\n")
        f.write(f"- Summary CSV: `{summary_csv}`\n")

    print(f"[ok] summary: {summary_csv}")
    print(f"[ok] markdown: {md}")


if __name__ == "__main__":
    main()
