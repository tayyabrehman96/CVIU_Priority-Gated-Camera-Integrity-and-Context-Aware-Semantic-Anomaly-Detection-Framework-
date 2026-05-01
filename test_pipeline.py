#!/usr/bin/env python3
"""
Comprehensive Detection Pipeline Benchmark

Tests all detection modules (YOLO, fire/smoke CV, weapon, anomaly guards)
against the synthetic test videos and reports per-label accuracy.

Anomaly Labels Tested:
  1. camera_blocked    - Black/white/signal loss
  2. camera_covered    - Hand/cloth/tape/cardboard covering lens
  3. blur_frame        - Camera blurred / out of focus
  4. camera_moved      - Camera repositioned / tilted
  5. fire_detected     - Fire / flames visible
  6. smoke_detected    - Smoke / haze visible
  7. weapon_detected   - Weapon (gun/knife/etc) visible
  8. theft_suspect     - Armed robbery / theft in progress

Usage:
    python test_pipeline.py [--videos-dir videos] [--sample-frames 50] [--verbose]
"""

import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

os.environ["OPENCV_LOG_LEVEL"] = "OFF"

from config import *
from core.guards import (
    detect_block_blank_blur,
    occlusion_candidate,
    detect_fire_cv,
    detect_smoke_cv,
    detect_camera_covered,
    detect_scene_collapse,
    _frame_metrics,
)


ALL_ANOMALY_LABELS = [
    "camera_blocked",
    "camera_covered",
    "blur_frame",
    "camera_moved",
    "fire_detected",
    "smoke_detected",
    "weapon_detected",
    "theft_suspect",
    "Benign",
]

GROUND_TRUTH_MAP = {
    "TEST_blur_defocus": "blur_frame",
    "TEST_blocked_brown_cardboard": "camera_covered",
    "TEST_blocked_hand": "camera_covered",
    "TEST_blocked_black_tape": "camera_blocked",
    "TEST_camera_moved": "camera_moved",
    "TEST_fire_smoke": "fire_detected",
    "TEST_normal_reference": "Benign",
    "TEST_weapon_theft": "weapon_detected",
    "TEST_camera_covered_cloth": "camera_covered",
    "TEST_smoke_only": "smoke_detected",
}

ANOMALY_SCHEDULE = {
    "normal_start": 0,
    "anomaly_1_start": 6,
    "anomaly_1_end": 14,
    "normal_mid_start": 14,
    "normal_mid_end": 20,
    "anomaly_2_start": 20,
    "anomaly_2_end": 32,
    "normal_end_start": 32,
}

DEFAULT_PROTOCOL_PATH = "risk_simulation_protocol.json"


def is_anomaly_segment(frame_idx: int, fps: float) -> bool:
    t = frame_idx / fps
    return (6.0 <= t <= 14.0) or (20.0 <= t <= 32.0)


def classify_frame(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    yolo_model=None,
    yolo_class_names=None,
    weapon_model=None,
    weapon_class_names=None,
    baseline_metrics: Optional[Dict] = None,
) -> Tuple[str, str, float]:
    """
    Run all detection modules on a single frame.
    Returns (label, reason, confidence).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    hard_label, hard_reason = detect_block_blank_blur(gray)
    if hard_label == "camera_blocked":
        return "camera_blocked", hard_reason, 0.9

    covered, covered_reason = detect_camera_covered(frame_bgr, gray)
    if covered:
        return "camera_covered", covered_reason, 0.85

    occ, occ_reason = occlusion_candidate(gray)
    if occ:
        return "camera_covered", occ_reason, 0.6

    if hard_label == "blur_frame":
        return "blur_frame", hard_reason, 0.8

    is_fire, fire_conf, fire_reason = detect_fire_cv(frame_bgr)
    if is_fire and fire_conf >= 0.3:
        return "fire_detected", fire_reason, fire_conf

    is_smoke, smoke_conf, smoke_reason = detect_smoke_cv(frame_bgr, baseline_metrics=baseline_metrics)
    if is_smoke and smoke_conf >= 0.3:
        return "smoke_detected", smoke_reason, smoke_conf

    has_person = False
    if yolo_model is not None:
        results = yolo_model(frame_rgb, verbose=False)
        if results:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = yolo_class_names.get(cls_id, "")
                if cls_name == "person" and conf >= 0.3:
                    has_person = True

    if weapon_model is not None:
        w_results = weapon_model(frame_rgb, verbose=False)
        if w_results and len(w_results[0].boxes) > 0:
            img_h, img_w = frame_rgb.shape[:2]
            img_area = float(img_h * img_w)
            for box in w_results[0].boxes:
                conf = float(box.conf[0])
                if conf < WEAPON_CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                box_frac = box_area / img_area
                if box_frac < WEAPON_MIN_BOX_AREA_FRAC or box_frac > WEAPON_MAX_BOX_AREA_FRAC:
                    continue
                label = "theft_suspect" if has_person else "weapon_detected"
                return label, f"Weapon conf={conf:.2f}", conf

    return "Benign", "No anomaly detected", 0.0


def run_benchmark(
    videos_dir: str,
    sample_every_n: int = 5,
    verbose: bool = False,
    manifest_path: Optional[str] = None,
    protocol_path: str = DEFAULT_PROTOCOL_PATH,
    load_models: bool = True,
) -> Dict:
    """Run benchmark on all TEST_ videos in the directory."""
    videos_path = Path(videos_dir)
    if not videos_path.is_dir():
        print(f"ERROR: Videos directory not found: {videos_dir}")
        sys.exit(1)

    # If a manifest is provided (hybrid simulation mode), evaluate all manifest videos.
    scenario_manifest = None
    if manifest_path:
        mpath = Path(manifest_path)
        if not mpath.is_file():
            print(f"ERROR: Manifest not found: {manifest_path}")
            sys.exit(1)
        with open(mpath, "r", encoding="utf-8") as f:
            scenario_manifest = json.load(f)
        test_videos = []
        for sc in scenario_manifest.get("scenarios", []):
            candidate = Path(sc.get("path", ""))
            if not candidate.is_absolute():
                candidate = Path(sc.get("path", ""))
            if candidate.is_file() and candidate.stat().st_size > 10000:
                test_videos.append(candidate)
        test_videos = sorted(test_videos)
    else:
        test_videos = sorted([
            f for f in videos_path.glob("TEST_*.mp4")
            if f.stat().st_size > 10000
        ])

    if not test_videos:
        print(f"No TEST_*.mp4 files found in {videos_dir}")
        print("Run 'python generate_test_videos.py' first to create test videos.")
        sys.exit(1)

    print("=" * 70)
    print("  Detection Pipeline Benchmark")
    print("=" * 70)
    print(f"\n  Videos directory: {videos_dir}")
    print(f"  Test videos found: {len(test_videos)}")
    print(f"  Sample every {sample_every_n} frames")

    yolo_model = None
    yolo_class_names = {}
    weapon_model = None
    weapon_class_names = {}

    print("\n--- Loading Models ---")
    if load_models:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8s.pt")
            yolo_class_names = yolo_model.names
            print(f"  YOLO (general): loaded ({len(yolo_class_names)} classes)")
        except Exception as e:
            print(f"  YOLO (general): FAILED ({e})")

        try:
            from ultralytics import YOLO as YOLO2
            wp = WEAPON_MODEL_PATH
            if os.path.isfile(wp):
                weapon_model = YOLO2(wp)
                weapon_class_names = weapon_model.names
                print(f"  YOLO (weapon): loaded ({len(weapon_class_names)} classes: {weapon_class_names})")
            else:
                print(f"  YOLO (weapon): model file not found at {wp}")
        except Exception as e:
            print(f"  YOLO (weapon): FAILED ({e})")
    else:
        print("  YOLO loading skipped (--no-models).")

    print(f"\n--- Anomaly Labels Under Test ---")
    for label in ALL_ANOMALY_LABELS:
        display = DISPLAY_LABELS.get(label, label)
        color = LABEL_COLORS.get(label, "#000000")
        print(f"  [{color}] {label:20s} -> {display}")

    results = {}
    per_video_results = {}
    total_frames = 0
    total_correct = 0
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    total_tn = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    print(f"\n{'=' * 70}")
    print(f"  Running Detection Tests")
    print(f"{'=' * 70}\n")

    manifest_lookup = {}
    if scenario_manifest:
        manifest_lookup = {Path(s.get("path", "")).name: s for s in scenario_manifest.get("scenarios", [])}

    for video_path in test_videos:
        video_name = video_path.stem
        if scenario_manifest:
            sc = manifest_lookup.get(video_path.name, {})
            gt_label = sc.get("expected_label", "Unknown")
        else:
            gt_label = GROUND_TRUTH_MAP.get(video_name, "Unknown")

        if gt_label == "Unknown":
            print(f"  SKIP {video_name} (no ground truth mapping)")
            continue

        print(f"  Testing: {video_name}")
        print(f"    Expected: {gt_label} ({DISPLAY_LABELS.get(gt_label, gt_label)})")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if frame_idx % sample_every_n != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pred_label, reason, confidence = classify_frame(
                frame, gray,
                yolo_model=yolo_model,
                yolo_class_names=yolo_class_names,
                weapon_model=weapon_model,
                weapon_class_names=weapon_class_names,
            )

            in_anomaly = is_anomaly_segment(frame_idx, fps)

            if in_anomaly:
                expected = gt_label
            else:
                expected = "Benign"

            correct = (pred_label == expected)
            frame_results.append({
                "frame": frame_idx,
                "expected": expected,
                "predicted": pred_label,
                "correct": correct,
                "confidence": confidence,
                "reason": reason[:80],
            })

            confusion[expected][pred_label] += 1
            total_frames += 1
            if correct:
                total_correct += 1

            for lbl in ALL_ANOMALY_LABELS:
                pred_is = (pred_label == lbl)
                exp_is = (expected == lbl)
                if pred_is and exp_is:
                    total_tp[lbl] += 1
                elif pred_is and not exp_is:
                    total_fp[lbl] += 1
                elif not pred_is and exp_is:
                    total_fn[lbl] += 1
                else:
                    total_tn[lbl] += 1

        cap.release()

        n_correct = sum(1 for r in frame_results if r["correct"])
        n_total = len(frame_results)
        accuracy = (n_correct / n_total * 100) if n_total > 0 else 0.0

        anomaly_frames = [r for r in frame_results if r["expected"] != "Benign"]
        anomaly_correct = sum(1 for r in anomaly_frames if r["correct"])
        anomaly_total = len(anomaly_frames)
        anomaly_acc = (anomaly_correct / anomaly_total * 100) if anomaly_total > 0 else 0.0

        normal_frames = [r for r in frame_results if r["expected"] == "Benign"]
        normal_correct = sum(1 for r in normal_frames if r["correct"])
        normal_total = len(normal_frames)
        normal_acc = (normal_correct / normal_total * 100) if normal_total > 0 else 0.0

        print(f"    Frames tested: {n_total} (of {total_video_frames})")
        print(f"    Overall accuracy:  {accuracy:5.1f}% ({n_correct}/{n_total})")
        print(f"    Anomaly accuracy:  {anomaly_acc:5.1f}% ({anomaly_correct}/{anomaly_total})")
        print(f"    Normal accuracy:   {normal_acc:5.1f}% ({normal_correct}/{normal_total})")

        if verbose and anomaly_frames:
            mispredictions = [r for r in anomaly_frames if not r["correct"]]
            if mispredictions:
                print(f"    Mispredictions (anomaly segments):")
                for r in mispredictions[:5]:
                    print(f"      frame={r['frame']}: expected={r['expected']}, got={r['predicted']} ({r['reason']})")

        per_video_results[video_name] = {
            "gt_label": gt_label,
            "accuracy": accuracy,
            "anomaly_accuracy": anomaly_acc,
            "normal_accuracy": normal_acc,
            "total_frames": n_total,
        }
        print()

    # Summary
    print("=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)

    overall_acc = (total_correct / total_frames * 100) if total_frames > 0 else 0.0
    print(f"\n  Total frames tested:   {total_frames}")
    print(f"  Overall accuracy:      {overall_acc:.1f}%")

    print(f"\n  --- Per-Label Metrics ---")
    print(f"  {'Label':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*6}")

    for lbl in ALL_ANOMALY_LABELS:
        tp = total_tp[lbl]
        fp = total_fp[lbl]
        fn = total_fn[lbl]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {lbl:<20s} {precision:>9.1%} {recall:>9.1%} {f1:>9.1%} {tp:>6d} {fp:>6d} {fn:>6d}")

    print(f"\n  --- Per-Video Results ---")
    print(f"  {'Video':<38s} {'Expected':<18s} {'Accuracy':>10s} {'Anomaly%':>10s}")
    print(f"  {'-'*38} {'-'*18} {'-'*10} {'-'*10}")
    for vname, vr in sorted(per_video_results.items()):
        print(f"  {vname:<38s} {vr['gt_label']:<18s} {vr['accuracy']:>9.1f}% {vr['anomaly_accuracy']:>9.1f}%")

    print(f"\n  --- Confusion Matrix ---")
    all_labels = sorted(set(
        list(confusion.keys()) +
        [p for row in confusion.values() for p in row.keys()]
    ))
    header = f"  {'Actual\\Pred':<20s}"
    for lbl in all_labels:
        short = lbl[:12]
        header += f" {short:>12s}"
    print(header)
    for actual in all_labels:
        row = f"  {actual:<20s}"
        for pred in all_labels:
            count = confusion[actual].get(pred, 0)
            row += f" {count:>12d}"
        print(row)

    print(f"\n{'=' * 70}")

    # Risk-generation quality metrics
    risk_metrics = {
        "scenario_coverage": 0.0,
        "severity_balance": 1.0,
        "temporal_stability": 0.0,
        "failure_case_rate": 0.0,
    }
    if scenario_manifest and scenario_manifest.get("scenarios"):
        scenarios = scenario_manifest["scenarios"]
        expected_labels = [s.get("expected_label", "Unknown") for s in scenarios]
        unique_labels = len(set(expected_labels))
        risk_metrics["scenario_coverage"] = unique_labels / max(1, len(ALL_ANOMALY_LABELS) - 1)

        severities = [str(s.get("severity", "medium")) for s in scenarios]
        sev_counts = {k: severities.count(k) for k in set(severities)}
        if sev_counts:
            maxc = max(sev_counts.values())
            minc = min(sev_counts.values())
            risk_metrics["severity_balance"] = (minc / maxc) if maxc > 0 else 1.0

        anomaly_total = sum(
            sum(row.values()) for actual, row in confusion.items() if actual != "Benign"
        )
        anomaly_correct = sum(
            row.get(actual, 0) for actual, row in confusion.items() if actual != "Benign"
        )
        risk_metrics["temporal_stability"] = anomaly_correct / anomaly_total if anomaly_total > 0 else 0.0
        risk_metrics["failure_case_rate"] = 1.0 - (overall_acc / 100.0)

    report_path = Path("reports")
    report_path.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = report_path / f"benchmark_{stamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Detection Pipeline Benchmark Report\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Overall accuracy: {overall_acc:.1f}%\n\n")

        f.write("Per-Label Metrics:\n")
        f.write(f"{'Label':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'TP':>6s} {'FP':>6s} {'FN':>6s}\n")
        for lbl in ALL_ANOMALY_LABELS:
            tp = total_tp[lbl]
            fp = total_fp[lbl]
            fn = total_fn[lbl]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f.write(f"{lbl:<20s} {p:>9.1%} {r:>9.1%} {f1:>9.1%} {tp:>6d} {fp:>6d} {fn:>6d}\n")

        f.write(f"\nPer-Video Results:\n")
        for vname, vr in sorted(per_video_results.items()):
            f.write(f"  {vname}: expected={vr['gt_label']}, accuracy={vr['accuracy']:.1f}%, anomaly_acc={vr['anomaly_accuracy']:.1f}%\n")

        if scenario_manifest:
            f.write("\nRisk-generation quality metrics:\n")
            for k, v in risk_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")

    print(f"  Report saved: {report_file}")
    # Summarize per-label metrics for downstream matrix runner.
    label_metrics = {}
    macro_f1 = []
    for lbl in ALL_ANOMALY_LABELS:
        tp = total_tp[lbl]
        fp = total_fp[lbl]
        fn = total_fn[lbl]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        label_metrics[lbl] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        macro_f1.append(f1)

    return {
        "accuracy": overall_acc,
        "macro_f1": float(np.mean(macro_f1)) if macro_f1 else 0.0,
        "per_video": per_video_results,
        "per_label": label_metrics,
        "risk_quality": risk_metrics,
        "report_file": str(report_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Detection Pipeline Benchmark")
    parser.add_argument("--videos-dir", default="videos", help="Directory containing TEST_*.mp4 files")
    parser.add_argument("--sample-frames", type=int, default=5, help="Sample every N frames (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed mispredictions")
    parser.add_argument("--manifest", default=None, help="Optional simulation manifest JSON file.")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL_PATH, help="Protocol file for simulation metadata.")
    parser.add_argument("--no-models", action="store_true", help="Skip YOLO model loading for fast reproducibility runs.")
    args = parser.parse_args()

    run_benchmark(
        videos_dir=args.videos_dir,
        sample_every_n=args.sample_frames,
        verbose=args.verbose,
        manifest_path=args.manifest,
        protocol_path=args.protocol,
        load_models=not args.no_models,
    )


if __name__ == "__main__":
    main()
