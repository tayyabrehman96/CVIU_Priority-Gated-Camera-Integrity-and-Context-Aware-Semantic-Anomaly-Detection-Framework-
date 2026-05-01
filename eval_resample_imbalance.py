#!/usr/bin/env python3
"""
Recompute macro-F1 (and per-class F1) under extreme Benign:anomaly prevalence
by resampling a frame-level prediction CSV (reviewer C11).

Expected CSV columns: y_true, y_pred  (or truth, prediction — use --col-true/--col-pred)

Example:
  python eval_resample_imbalance.py --csv predictions.csv --benign-ratio 100 --seed 42

Does not require video data; use exports from your evaluation notebook or pipeline.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def f1_per_class(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return out


def resample(
    rows: List[Tuple[str, str]],
    benign_ratio: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    benign = [(t, p) for t, p in rows if t == "Benign"]
    anomaly = [(t, p) for t, p in rows if t != "Benign"]
    if not benign or not anomaly:
        raise ValueError("Need both Benign and non-Benign rows for ratio resampling.")
    n_anom = len(anomaly)
    n_ben = min(len(benign), benign_ratio * n_anom)
    idx_b = rng.choice(len(benign), size=n_ben, replace=True)
    idx_a = rng.choice(len(anomaly), size=n_anom, replace=True)
    out = [benign[i] for i in idx_b] + [anomaly[i] for i in idx_a]
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Frame-level predictions CSV")
    ap.add_argument("--col-true", default="y_true")
    ap.add_argument("--col-pred", default="y_pred")
    ap.add_argument("--benign-ratio", type=int, default=100, help="Target max Benign:per-anomaly-frame (approx)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows_raw: List[Tuple[str, str]] = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows_raw.append((row[args.col_true].strip(), row[args.col_pred].strip()))

    labels = sorted({t for t, _ in rows_raw} | {p for _, p in rows_raw})
    rng = np.random.default_rng(args.seed)
    resampled = resample(rows_raw, args.benign_ratio, rng)
    yt = [t for t, _ in resampled]
    yp = [p for _, p in resampled]
    f1s = f1_per_class(yt, yp, labels)
    macro = float(np.mean([f1s[c] for c in labels]))
    print(f"Resampled size: {len(resampled)} | benign_ratio param={args.benign_ratio}")
    print(f"Macro-F1: {macro*100:.2f}%")
    for c in labels:
        print(f"  F1 {c}: {f1s[c]*100:.2f}%")


if __name__ == "__main__":
    main()
