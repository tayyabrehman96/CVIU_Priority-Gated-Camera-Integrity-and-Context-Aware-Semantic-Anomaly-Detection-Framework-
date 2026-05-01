#!/usr/bin/env python3
"""
Hybrid risk simulation builder.

Creates a reproducible scenario pack that combines:
1) dataset-native scenarios (existing TEST_*.mp4)
2) synthetic risk injections derived from base normal videos

Outputs:
  - simulations/<run_id>/videos/*.mp4
  - simulations/<run_id>/manifest.json
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DEFAULT_PROTOCOL = "risk_simulation_protocol.json"
DEFAULT_INPUT_VIDEOS = "videos"
DEFAULT_OUTPUT_ROOT = "simulations"

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


def load_protocol(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_schedule_frames(schedule_seconds: Dict[str, List[float]], fps: float) -> List[Tuple[int, int, bool]]:
    # normal_start, anomaly_1, normal_mid, anomaly_2, normal_end
    windows = []
    for key, (s, e) in schedule_seconds.items():
        is_anomaly = key.startswith("anomaly_")
        windows.append((int(s * fps), int(e * fps), is_anomaly))
    return windows


def anomaly_intensity(frame_idx: int, windows: List[Tuple[int, int, bool]], scale: float) -> float:
    val = 0.0
    for s, e, is_anomaly in windows:
        if s <= frame_idx < e and is_anomaly:
            center = (s + e) / 2.0
            half = max(1.0, (e - s) / 2.0)
            triangular = max(0.0, 1.0 - abs(frame_idx - center) / half)
            val = max(val, triangular)
    return max(0.0, min(1.0, val * scale))


def inject_synthetic_risk(frame: np.ndarray, label: str, intensity: float, rng: random.Random) -> np.ndarray:
    if intensity <= 0.01:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]

    if label == "camera_blocked":
        alpha = min(1.0, 0.3 + intensity * 0.7)
        black = np.zeros_like(out)
        out = cv2.addWeighted(out, 1.0 - alpha, black, alpha, 0)
    elif label == "camera_covered":
        alpha = min(0.95, 0.2 + intensity * 0.6)
        cloth = np.full_like(out, (80, 90, 100), dtype=np.uint8)
        noise = np.random.randint(-10, 11, out.shape, dtype=np.int16)
        cloth = np.clip(cloth.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out = cv2.addWeighted(out, 1.0 - alpha, cloth, alpha, 0)
    elif label == "blur_frame":
        k = int(3 + intensity * 21)
        if k % 2 == 0:
            k += 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif label == "camera_moved":
        dx = int(intensity * w * 0.15)
        dy = int(intensity * h * 0.10)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif label == "smoke_detected":
        haze_alpha = min(0.7, intensity * 0.5)
        haze = np.full_like(out, (190, 190, 190), dtype=np.uint8)
        out = cv2.addWeighted(out, 1.0 - haze_alpha, haze, haze_alpha, 0)
    elif label == "fire_detected":
        cx = int(w * (0.35 + 0.3 * rng.random()))
        cy = int(h * (0.55 + 0.25 * rng.random()))
        radius = int((min(w, h) * 0.08) * (0.5 + intensity))
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), radius, (0, 110, 240), -1)
        out = cv2.addWeighted(out, 1.0 - min(0.7, intensity), overlay, min(0.7, intensity), 0)
    elif label in {"weapon_detected", "theft_suspect"}:
        x = int(w * (0.35 + 0.2 * rng.random()))
        y = int(h * (0.45 + 0.2 * rng.random()))
        ww = max(12, int(w * (0.05 + 0.04 * intensity)))
        hh = max(6, int(h * (0.015 + 0.02 * intensity)))
        cv2.rectangle(out, (x, y), (x + ww, y + hh), (20, 20, 20), -1)

    return out


def generate_synthetic_variant(
    source_video: Path,
    output_video: Path,
    target_label: str,
    severity_cfg: Dict,
    schedule_seconds: Dict[str, List[float]],
    rng_seed: int,
) -> Dict:
    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(45 * fps)
    windows = get_schedule_frames(schedule_seconds, fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))
    rng = random.Random(rng_seed)

    injected_frames = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        intensity = anomaly_intensity(frame_idx, windows, severity_cfg["intensity_scale"])
        out = inject_synthetic_risk(frame, target_label, intensity, rng)
        if intensity > 0.01:
            injected_frames += 1
        writer.write(out)
        frame_idx += 1

    cap.release()
    writer.release()

    return {
        "video_name": output_video.name,
        "path": str(output_video.as_posix()),
        "source_type": "synthetic_injection",
        "source_video": source_video.name,
        "expected_label": target_label,
        "severity": severity_cfg,
        "frames_total": frame_idx if frame_idx > 0 else total,
        "frames_injected": injected_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid risk simulation pack.")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, help="Path to protocol JSON.")
    parser.add_argument("--input-videos", default=DEFAULT_INPUT_VIDEOS, help="Input videos directory.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Simulation output root.")
    parser.add_argument("--seed", type=int, default=13, help="Global reproducibility seed.")
    parser.add_argument("--severity", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--mode", choices=["dataset_only", "synthetic_only", "hybrid"], default="hybrid")
    parser.add_argument("--metadata-only", action="store_true", help="Create synthetic scenarios from existing TEST videos without expensive rendering.")
    args = parser.parse_args()

    protocol = load_protocol(Path(args.protocol))
    input_dir = Path(args.input_videos)
    run_id = f"run_{args.mode}_{args.severity}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_id
    out_videos = run_dir / "videos"
    out_videos.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": protocol.get("version", "unknown"),
        "mode": args.mode,
        "severity": args.severity,
        "seed": args.seed,
        "datasets": protocol.get("datasets", []),
        "schedule_seconds": protocol.get("temporal_schedule_seconds", {}),
        "scenarios": [],
    }

    # Dataset-native scenarios: copy existing TEST_*.mp4 and attach metadata.
    if args.mode in {"dataset_only", "hybrid"}:
        for p in sorted(input_dir.glob("TEST_*.mp4")):
            if p.stat().st_size < 10000:
                continue
            gt = GROUND_TRUTH_MAP.get(p.stem, "Benign")
            out = out_videos / p.name
            shutil.copy2(p, out)
            manifest["scenarios"].append(
                {
                    "video_name": out.name,
                    "path": str(out.as_posix()),
                    "source_type": "dataset_native",
                    "source_video": p.name,
                    "expected_label": gt,
                    "severity": args.severity,
                    "parameters": {
                        "dataset_role": "public_risk_events" if gt != "Benign" else "normal_baseline"
                    },
                }
            )

    # Synthetic injections: derive from normal videos.
    if args.mode in {"synthetic_only", "hybrid"}:
        risk_labels = [
            "camera_blocked",
            "camera_covered",
            "blur_frame",
            "camera_moved",
            "fire_detected",
            "smoke_detected",
            "weapon_detected",
        ]
        severity_cfg = protocol["severity_levels"][args.severity]
        schedule = protocol["temporal_schedule_seconds"]

        if args.metadata_only:
            label_to_test_video = {}
            for stem, lbl in GROUND_TRUTH_MAP.items():
                label_to_test_video.setdefault(lbl, stem + ".mp4")
            for ridx, label in enumerate(risk_labels):
                src_name = label_to_test_video.get(label, "TEST_normal_reference.mp4")
                src = input_dir / src_name
                if not src.exists():
                    continue
                out_name = f"SIMMETA_{label}_{args.severity}_s{args.seed}.mp4"
                out = out_videos / out_name
                shutil.copy2(src, out)
                manifest["scenarios"].append(
                    {
                        "video_name": out_name,
                        "path": str(out.as_posix()),
                        "source_type": "synthetic_injection_metadata_only",
                        "source_video": src.name,
                        "expected_label": label,
                        "severity": args.severity,
                        "parameters": {
                            "severity_name": args.severity,
                            "intensity_scale": severity_cfg["intensity_scale"],
                            "frequency_scale": severity_cfg["event_frequency_scale"],
                            "duration_jitter_seconds": severity_cfg["duration_jitter_seconds"],
                            "metadata_only": True,
                        },
                    }
                )
        else:
            normal_sources = sorted(input_dir.glob("Normal_Videos_*_x264.mp4"))
            if not normal_sources:
                raise FileNotFoundError("No Normal_Videos_*_x264.mp4 files were found for synthetic generation.")
            selected_sources = normal_sources[:3]
            for idx, src in enumerate(selected_sources):
                for ridx, label in enumerate(risk_labels):
                    out_name = f"SIM_{src.stem}_{label}_{args.severity}_s{args.seed}.mp4"
                    out_path = out_videos / out_name
                    metadata = generate_synthetic_variant(
                        source_video=src,
                        output_video=out_path,
                        target_label=label,
                        severity_cfg=severity_cfg,
                        schedule_seconds=schedule,
                        rng_seed=args.seed + idx * 101 + ridx * 17,
                    )
                    metadata["parameters"] = {
                        "severity_name": args.severity,
                        "intensity_scale": severity_cfg["intensity_scale"],
                        "frequency_scale": severity_cfg["event_frequency_scale"],
                        "duration_jitter_seconds": severity_cfg["duration_jitter_seconds"],
                        "metadata_only": False,
                    }
                    manifest["scenarios"].append(metadata)

    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] Simulation pack generated: {run_dir}")
    print(f"[ok] Scenarios: {len(manifest['scenarios'])}")
    print(f"[ok] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
