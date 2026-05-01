#!/usr/bin/env python3
"""
Generate synthetic anomaly test videos for conference demo.

Each video contains MULTIPLE scene transitions (normal -> anomaly -> normal -> another anomaly)
rather than a single static anomaly, making them more realistic for testing.

Videos generated:
  1.  Blur / defocus  (multi-scene)
  2.  Blocked with brown cardboard  (multi-scene)
  3.  Blocked with hand  (multi-scene)
  4.  Blocked with black tape  (multi-scene)
  5.  Camera moved / shifted  (multi-scene)
  6.  Fire / smoke  (advanced particle effects)
  7.  Normal reference
  8.  Weapon / theft scene (person silhouette + weapon overlay)
  9.  Camera covered with cloth  (multi-scene)
  10. Smoke only (no fire, just smoke haze)

Usage:
    python generate_test_videos.py
"""

import os
import sys
import math
import cv2
import numpy as np
from pathlib import Path

VIDEO_DIR = Path("videos")
DURATION_S = 45
FPS_DEFAULT = 25.0


def find_source_video() -> str:
    candidates = [
        "Normal_Videos_015_x264.mp4",
        "Normal_Videos_345_x264.mp4",
        "Normal_Videos_745_x264.mp4",
        "Normal_Videos_251_x264.mp4",
    ]
    for name in candidates:
        p = VIDEO_DIR / name
        if p.exists() and p.stat().st_size > 10000:
            return str(p)
    for p in sorted(VIDEO_DIR.glob("*.mp4")):
        if p.stat().st_size > 50000 and "TEST_" not in p.name:
            return str(p)
    return ""


def open_source(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def make_writer(output_path: str, fps: float, w: int, h: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def read_or_loop(cap) -> np.ndarray:
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame


def ease_in_out(t: float) -> float:
    return 0.5 * (1 - math.cos(math.pi * t))


# =========================================================
#  SCENE SCHEDULE: defines when anomalies activate/deactivate
# =========================================================

def build_multi_scene_schedule(fps: float) -> list:
    """
    Returns list of (start_frame, end_frame, is_anomaly) tuples.
    Pattern: normal -> anomaly -> normal -> anomaly -> normal tail
    """
    segs = []
    f = lambda s: int(s * fps)
    segs.append((0, f(6), False))           # 0-6s normal
    segs.append((f(6), f(14), True))        # 6-14s anomaly
    segs.append((f(14), f(20), False))      # 14-20s normal
    segs.append((f(20), f(32), True))       # 20-32s anomaly
    segs.append((f(32), f(DURATION_S), False))  # 32-end normal
    return segs


def get_anomaly_intensity(frame_idx: int, schedule: list, transition_frames: int = 30) -> float:
    """Returns 0.0 (normal) to 1.0 (full anomaly) with smooth transitions."""
    for start, end, is_anomaly in schedule:
        if start <= frame_idx < end:
            if not is_anomaly:
                return 0.0
            frames_in = frame_idx - start
            frames_to_end = end - frame_idx
            ramp_up = min(1.0, frames_in / transition_frames)
            ramp_down = min(1.0, frames_to_end / transition_frames)
            return ease_in_out(ramp_up) * ease_in_out(ramp_down)
    return 0.0


# =========================================================
#  1. BLUR / DEFOCUS  (multi-scene)
# =========================================================

def generate_blur(source: str, output: str):
    print(f"  [1/8] Generating blur video (multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        intensity = get_anomaly_intensity(i, schedule)
        if intensity > 0.01:
            ksize = int(3 + intensity * 55)
            if ksize % 2 == 0:
                ksize += 1
            frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  2. BLOCKED - BROWN CARDBOARD  (multi-scene)
# =========================================================

def generate_blocked_brown(source: str, output: str):
    print(f"  [2/8] Generating blocked (brown cardboard, multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        alpha = get_anomaly_intensity(i, schedule)
        if alpha > 0.01:
            brown = np.full_like(frame, (30, 75, 139), dtype=np.uint8)
            noise = np.random.randint(-8, 9, frame.shape, dtype=np.int16)
            brown = np.clip(brown.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            texture = np.random.randint(0, 15, (h, w), dtype=np.uint8)
            for c in range(3):
                brown[:, :, c] = np.clip(brown[:, :, c].astype(np.int16) + texture, 0, 255).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1.0 - alpha, brown, alpha, 0)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  3. BLOCKED - HAND  (multi-scene)
# =========================================================

def generate_blocked_hand(source: str, output: str):
    print(f"  [3/8] Generating blocked (hand, multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)
    skin_bgr = np.array([130, 160, 200], dtype=np.uint8)

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        progress = get_anomaly_intensity(i, schedule)
        if progress > 0.01:
            hand_mask = np.zeros((h, w), dtype=np.float32)
            cover_x = int(w * (1.0 - progress * 0.85))
            hand_mask[:, cover_x:] = 1.0
            edge_w = max(1, int(w * 0.08))
            for col in range(max(0, cover_x - edge_w), cover_x):
                hand_mask[:, col] = (col - (cover_x - edge_w)) / edge_w
            hand_layer = np.full_like(frame, skin_bgr, dtype=np.uint8)
            variation = np.random.randint(-12, 13, frame.shape, dtype=np.int16)
            hand_layer = np.clip(hand_layer.astype(np.int16) + variation, 0, 255).astype(np.uint8)
            mask3 = np.stack([hand_mask] * 3, axis=-1)
            frame = (frame.astype(np.float32) * (1.0 - mask3) + hand_layer.astype(np.float32) * mask3).astype(np.uint8)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  4. BLOCKED - BLACK TAPE  (multi-scene)
# =========================================================

def generate_blocked_black(source: str, output: str):
    print(f"  [4/8] Generating blocked (black tape, multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        alpha = get_anomaly_intensity(i, schedule)
        if alpha > 0.01:
            black = np.full_like(frame, (10, 8, 5), dtype=np.uint8)
            noise = np.random.randint(-3, 4, frame.shape, dtype=np.int16)
            black = np.clip(black.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1.0 - alpha, black, alpha, 0)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  5. CAMERA MOVED  (multi-scene)
# =========================================================

def generate_camera_moved(source: str, output: str):
    print(f"  [5/8] Generating camera moved (multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)
    max_dx = int(w * 0.18)
    max_dy = int(h * 0.12)
    max_angle = 8.0

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        t = get_anomaly_intensity(i, schedule)
        if t > 0.01:
            dx = int(max_dx * t)
            dy = int(max_dy * t)
            angle = max_angle * t
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += dx
            M[1, 2] += dy
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  ADVANCED FIRE PARTICLE SYSTEM
# =========================================================

class FireParticle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'temp')

    def __init__(self, cx, cy, spread):
        self.x = cx + np.random.uniform(-spread, spread)
        self.y = cy + np.random.uniform(-spread * 0.3, spread * 0.3)
        self.vx = np.random.uniform(-1.5, 1.5)
        self.vy = np.random.uniform(-4.0, -1.0)
        self.life = 0
        self.max_life = np.random.randint(15, 45)
        self.size = np.random.randint(4, 18)
        self.temp = np.random.uniform(0.6, 1.0)

    def update(self):
        self.life += 1
        self.x += self.vx + np.random.uniform(-0.3, 0.3)
        self.y += self.vy
        self.vy -= 0.05
        self.size = max(1, self.size - 0.2)

    @property
    def alive(self):
        return self.life < self.max_life

    @property
    def alpha(self):
        frac = self.life / self.max_life
        if frac < 0.2:
            return frac / 0.2
        return max(0.0, 1.0 - (frac - 0.2) / 0.8)

    def color_bgr(self):
        frac = self.life / self.max_life
        if frac < 0.3:
            return (40, int(220 * self.temp), int(255 * self.temp))
        elif frac < 0.6:
            return (20, int(120 * self.temp), int(230 * self.temp))
        else:
            return (10, int(40 * self.temp), int(180 * self.temp))


class SmokeParticle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'size')

    def __init__(self, cx, cy, spread):
        self.x = cx + np.random.uniform(-spread * 0.5, spread * 0.5)
        self.y = cy - np.random.uniform(0, spread * 0.5)
        self.vx = np.random.uniform(-0.8, 0.8)
        self.vy = np.random.uniform(-2.5, -0.5)
        self.life = 0
        self.max_life = np.random.randint(30, 80)
        self.size = np.random.randint(10, 35)

    def update(self):
        self.life += 1
        self.x += self.vx + np.random.uniform(-0.2, 0.2)
        self.y += self.vy
        self.vx *= 0.98
        self.size = min(60, self.size + 0.5)

    @property
    def alive(self):
        return self.life < self.max_life

    @property
    def alpha(self):
        frac = self.life / self.max_life
        if frac < 0.15:
            return frac / 0.15 * 0.4
        return max(0.0, 0.4 * (1.0 - (frac - 0.15) / 0.85))


def render_fire_smoke_particles(frame, fire_particles, smoke_particles):
    overlay = frame.copy().astype(np.float32)

    for p in smoke_particles:
        if not p.alive:
            continue
        a = p.alpha
        if a < 0.01:
            continue
        s = int(p.size)
        x1 = max(0, int(p.x) - s)
        y1 = max(0, int(p.y) - s)
        x2 = min(frame.shape[1], int(p.x) + s)
        y2 = min(frame.shape[0], int(p.y) + s)
        if x2 <= x1 or y2 <= y1:
            continue
        gray_val = np.random.randint(160, 200)
        smoke_color = np.array([gray_val, gray_val, gray_val], dtype=np.float32)
        region = overlay[y1:y2, x1:x2]
        overlay[y1:y2, x1:x2] = region * (1.0 - a) + smoke_color * a

    for p in fire_particles:
        if not p.alive:
            continue
        a = p.alpha
        if a < 0.01:
            continue
        s = int(p.size)
        x1 = max(0, int(p.x) - s)
        y1 = max(0, int(p.y) - s)
        x2 = min(frame.shape[1], int(p.x) + s)
        y2 = min(frame.shape[0], int(p.y) + s)
        if x2 <= x1 or y2 <= y1:
            continue
        color = np.array(p.color_bgr(), dtype=np.float32)
        region = overlay[y1:y2, x1:x2]
        glow_a = min(1.0, a * 1.3)
        overlay[y1:y2, x1:x2] = region * (1.0 - glow_a) + color * glow_a

    return np.clip(overlay, 0, 255).astype(np.uint8)


# =========================================================
#  6. FIRE / SMOKE  (advanced particles, multi-scene)
# =========================================================

def generate_fire_smoke(source: str, output: str):
    print(f"  [6/8] Generating fire/smoke (advanced particles, multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)

    fire_cx = int(w * 0.65)
    fire_cy = int(h * 0.78)
    fire_spread = int(min(w, h) * 0.12)

    fire_particles = []
    smoke_particles = []

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break

        intensity = get_anomaly_intensity(i, schedule)

        if intensity > 0.05:
            spawn_rate = int(intensity * 8)
            for _ in range(spawn_rate):
                fire_particles.append(FireParticle(fire_cx, fire_cy, fire_spread))
            smoke_spawn = int(intensity * 3)
            for _ in range(smoke_spawn):
                smoke_particles.append(SmokeParticle(fire_cx, fire_cy - fire_spread, fire_spread))

        for p in fire_particles:
            p.update()
        for p in smoke_particles:
            p.update()
        fire_particles = [p for p in fire_particles if p.alive]
        smoke_particles = [p for p in smoke_particles if p.alive]

        if fire_particles or smoke_particles:
            frame = render_fire_smoke_particles(frame, fire_particles, smoke_particles)

            if intensity > 0.3:
                glow_mask = np.zeros((h, w), dtype=np.float32)
                y_grid, x_grid = np.ogrid[:h, :w]
                dist = np.sqrt((x_grid - fire_cx) ** 2 + (y_grid - fire_cy) ** 2).astype(np.float32)
                glow_mask = np.clip(1.0 - dist / (fire_spread * 3), 0, 1) * intensity * 0.15
                glow_color = np.array([20, 80, 200], dtype=np.float32)
                for c in range(3):
                    frame[:, :, c] = np.clip(
                        frame[:, :, c].astype(np.float32) + glow_mask * glow_color[c],
                        0, 255
                    ).astype(np.uint8)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  7. NORMAL REFERENCE
# =========================================================

def generate_normal(source: str, output: str):
    print(f"  [7/8] Generating normal reference: {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  8. WEAPON / THEFT SCENE
# =========================================================

def draw_person_silhouette(frame, cx, cy, scale=1.0, color=(40, 40, 40)):
    """Draw a simple dark person silhouette at (cx, cy)."""
    s = scale
    pts_body = np.array([
        [cx - int(20*s), cy - int(60*s)],
        [cx + int(20*s), cy - int(60*s)],
        [cx + int(25*s), cy + int(40*s)],
        [cx - int(25*s), cy + int(40*s)],
    ], np.int32)
    cv2.fillPoly(frame, [pts_body], color)
    cv2.circle(frame, (cx, cy - int(75*s)), int(18*s), color, -1)

    pts_arm_r = np.array([
        [cx + int(20*s), cy - int(40*s)],
        [cx + int(55*s), cy - int(20*s)],
        [cx + int(50*s), cy - int(10*s)],
        [cx + int(18*s), cy - int(30*s)],
    ], np.int32)
    cv2.fillPoly(frame, [pts_arm_r], color)
    return frame


def draw_weapon(frame, cx, cy, scale=1.0):
    """Draw a simple gun shape extending from the person's hand."""
    s = scale
    hand_x = cx + int(55*s)
    hand_y = cy - int(15*s)

    gun_pts = np.array([
        [hand_x, hand_y - int(4*s)],
        [hand_x + int(40*s), hand_y - int(3*s)],
        [hand_x + int(40*s), hand_y + int(3*s)],
        [hand_x, hand_y + int(4*s)],
    ], np.int32)
    cv2.fillPoly(frame, [gun_pts], (30, 30, 30))

    grip_pts = np.array([
        [hand_x + int(5*s), hand_y + int(4*s)],
        [hand_x + int(12*s), hand_y + int(4*s)],
        [hand_x + int(10*s), hand_y + int(18*s)],
        [hand_x + int(3*s), hand_y + int(18*s)],
    ], np.int32)
    cv2.fillPoly(frame, [grip_pts], (25, 25, 25))
    return frame


def generate_weapon_theft(source: str, output: str):
    print(f"  [8/8] Generating weapon/theft scene (multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)

    person_cx = int(w * 0.35)
    person_cy = int(h * 0.55)
    person_scale = min(w, h) / 300.0

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        intensity = get_anomaly_intensity(i, schedule)
        if intensity > 0.05:
            sway = int(3 * math.sin(i * 0.1))
            pcx = person_cx + sway
            pcy = person_cy

            person_overlay = frame.copy()
            draw_person_silhouette(person_overlay, pcx, pcy, person_scale, color=(35, 35, 40))
            draw_weapon(person_overlay, pcx, pcy, person_scale)

            mask_alpha = intensity
            frame = cv2.addWeighted(frame, 1.0 - mask_alpha, person_overlay, mask_alpha, 0)

            if intensity > 0.5:
                danger_text = "ARMED PERSON"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7 * (min(w, h) / 480.0)
                thickness = max(1, int(2 * (min(w, h) / 480.0)))
                text_size = cv2.getTextSize(danger_text, font, font_scale, thickness)[0]
                tx = pcx - text_size[0] // 2
                ty = pcy - int(100 * person_scale)
                flash = 1.0 if (i % 20) < 10 else 0.5
                color = (0, 0, int(255 * flash))
                cv2.putText(frame, danger_text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  9. CAMERA COVERED WITH CLOTH  (multi-scene)
# =========================================================

def generate_covered_cloth(source: str, output: str):
    print(f"  [9/10] Generating camera covered (cloth, multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)

    cloth_base_color = np.array([95, 85, 75], dtype=np.uint8)

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break
        alpha = get_anomaly_intensity(i, schedule)
        if alpha > 0.01:
            cloth = np.full_like(frame, cloth_base_color, dtype=np.uint8)
            weave_x = np.sin(np.arange(w) * 0.3).astype(np.float32) * 8
            weave_y = np.cos(np.arange(h) * 0.25).astype(np.float32) * 6
            for row in range(h):
                shift = int(weave_y[row])
                cloth[row] = np.roll(cloth[row], shift, axis=0)
            noise = np.random.randint(-15, 16, frame.shape, dtype=np.int16)
            cloth = np.clip(cloth.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            thread_pattern = np.zeros((h, w), dtype=np.uint8)
            for row in range(0, h, 3):
                thread_pattern[row, :] = 12
            for col in range(0, w, 4):
                thread_pattern[:, col] += 8
            for c in range(3):
                cloth[:, :, c] = np.clip(
                    cloth[:, :, c].astype(np.int16) + thread_pattern.astype(np.int16), 0, 255
                ).astype(np.uint8)
            frame = cv2.addWeighted(frame, 1.0 - alpha, cloth, alpha, 0)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  10. SMOKE ONLY (no fire, just smoke haze)
# =========================================================

def generate_smoke_only(source: str, output: str):
    print(f"  [10/10] Generating smoke only (haze, multi-scene): {os.path.basename(output)}")
    cap, fps, w, h = open_source(source)
    writer = make_writer(output, fps, w, h)
    total = int(DURATION_S * fps)
    schedule = build_multi_scene_schedule(fps)

    smoke_particles = []

    for i in range(total):
        frame = read_or_loop(cap)
        if frame is None:
            break

        intensity = get_anomaly_intensity(i, schedule)

        if intensity > 0.05:
            spawn_rate = int(intensity * 6)
            cx = int(w * 0.5 + np.random.uniform(-w * 0.2, w * 0.2))
            cy = int(h * 0.7)
            spread = int(min(w, h) * 0.2)
            for _ in range(spawn_rate):
                smoke_particles.append(SmokeParticle(cx, cy, spread))

        for p in smoke_particles:
            p.update()
        smoke_particles = [p for p in smoke_particles if p.alive]

        if smoke_particles:
            overlay = frame.copy().astype(np.float32)
            for p in smoke_particles:
                if not p.alive:
                    continue
                a = p.alpha
                if a < 0.01:
                    continue
                s = int(p.size)
                x1 = max(0, int(p.x) - s)
                y1 = max(0, int(p.y) - s)
                x2 = min(w, int(p.x) + s)
                y2 = min(h, int(p.y) + s)
                if x2 <= x1 or y2 <= y1:
                    continue
                gray_val = np.random.randint(170, 210)
                smoke_color = np.array([gray_val, gray_val, gray_val], dtype=np.float32)
                region = overlay[y1:y2, x1:x2]
                overlay[y1:y2, x1:x2] = region * (1.0 - a) + smoke_color * a
            frame = np.clip(overlay, 0, 255).astype(np.uint8)

        if intensity > 0.3:
            haze_alpha = intensity * 0.25
            haze_color = np.full_like(frame, (190, 190, 190), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 1.0 - haze_alpha, haze_color, haze_alpha, 0)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    Done ({total} frames)")


# =========================================================
#  MAIN
# =========================================================

def main():
    print("=" * 60)
    print("  Conference Demo - Synthetic Anomaly Video Generator v3")
    print("  (Multi-scene + Fire/Smoke/Covered/Weapon)")
    print("=" * 60)

    source = find_source_video()
    if not source:
        print(f"\nERROR: No source video found in {VIDEO_DIR}/")
        print("Place normal .mp4 surveillance videos there first.")
        sys.exit(1)

    print(f"\nSource video: {source}")

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Resolution: {w}x{h} @ {fps:.1f}fps ({total_src} frames)")
    print(f"Output duration: {DURATION_S}s per video")
    print(f"Scene pattern: normal -> anomaly -> normal -> anomaly -> normal\n")

    videos = [
        ("TEST_blur_defocus.mp4", generate_blur),
        ("TEST_blocked_brown_cardboard.mp4", generate_blocked_brown),
        ("TEST_blocked_hand.mp4", generate_blocked_hand),
        ("TEST_blocked_black_tape.mp4", generate_blocked_black),
        ("TEST_camera_moved.mp4", generate_camera_moved),
        ("TEST_fire_smoke.mp4", generate_fire_smoke),
        ("TEST_normal_reference.mp4", generate_normal),
        ("TEST_weapon_theft.mp4", generate_weapon_theft),
        ("TEST_camera_covered_cloth.mp4", generate_covered_cloth),
        ("TEST_smoke_only.mp4", generate_smoke_only),
    ]

    for filename, gen_func in videos:
        out_path = str(VIDEO_DIR / filename)
        gen_func(source, out_path)

    print(f"\n{'=' * 60}")
    print(f"  All {len(videos)} test videos generated in {VIDEO_DIR}/")
    print(f"{'=' * 60}")
    print("\nGenerated files:")
    for filename, _ in videos:
        p = VIDEO_DIR / filename
        size_mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0
        print(f"  {filename:40s} {size_mb:6.1f} MB")
    print(f"\nRun 'python test_pipeline.py' to benchmark all detections.")
    print(f"Run 'python main.py' and click Start Processing to test in the UI.")


if __name__ == "__main__":
    main()
