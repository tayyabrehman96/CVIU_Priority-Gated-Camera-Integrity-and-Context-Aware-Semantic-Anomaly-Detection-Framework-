import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
import datetime

from config import *


# =============================================================================
# FIRE / SMOKE CV-BASED DETECTION
# =============================================================================

def detect_fire_cv(frame_bgr: np.ndarray, motion_mask: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:
    """
    Detect fire regions using HSV color analysis on the frame.
    Returns (is_fire, confidence, reason).
    Fire pixels are orange-red with high saturation and value.
    """
    try:
        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0, ""

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]
        total_pixels = float(h * w)

        mask1 = cv2.inRange(hsv, np.array(FIRE_HSV_LOWER_1), np.array(FIRE_HSV_UPPER_1))
        mask2 = cv2.inRange(hsv, np.array(FIRE_HSV_LOWER_2), np.array(FIRE_HSV_UPPER_2))
        fire_mask = cv2.bitwise_or(mask1, mask2)

        if motion_mask is not None:
            if motion_mask.shape[:2] != fire_mask.shape[:2]:
                motion_mask = cv2.resize(motion_mask, (fire_mask.shape[1], fire_mask.shape[0]))
            fire_mask = cv2.bitwise_and(fire_mask, motion_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        fire_pixels = float(np.count_nonzero(fire_mask))
        fire_frac = fire_pixels / total_pixels

        if fire_frac < FIRE_MIN_PIXEL_FRAC:
            return False, 0.0, ""

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) >= FIRE_MIN_CONTOUR_AREA]

        if not large_contours:
            return False, 0.0, ""

        max_area = max(cv2.contourArea(c) for c in large_contours)
        confidence = min(1.0, fire_frac / 0.05) * 0.5 + min(1.0, max_area / 2000.0) * 0.5

        reason = (
            f"Fire CV: {len(large_contours)} regions, "
            f"pixel_frac={fire_frac:.4f}, max_area={max_area:.0f}, conf={confidence:.2f}"
        )
        return True, confidence, reason

    except Exception as e:
        return False, 0.0, f"Fire CV error: {e}"


def detect_smoke_cv(
    frame_bgr: np.ndarray,
    baseline_metrics: Optional[Dict[str, float]] = None,
    motion_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, float, str]:
    """
    Detect smoke regions using color + contrast analysis.
    Smoke: grayish-white pixels with low saturation and reduced local contrast.
    Returns (is_smoke, confidence, reason).

    NOTE: Requires motion_mask to avoid false positives on static gray/neutral
    surfaces (walls, concrete, clothing, overcast sky). Without motion overlap
    we cannot distinguish real smoke from ordinary gray pixels.
    """
    try:
        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0, ""

        if motion_mask is None:
            return False, 0.0, ""

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = frame_bgr.shape[:2]
        total_pixels = float(h * w)

        low_sat = hsv[:, :, 1] <= SMOKE_SATURATION_MAX
        val = hsv[:, :, 2]
        gray_range = (val >= SMOKE_GRAY_RANGE[0]) & (val <= SMOKE_GRAY_RANGE[1])
        smoke_mask = (low_sat & gray_range).astype(np.uint8) * 255

        if motion_mask.shape[:2] != smoke_mask.shape[:2]:
            motion_mask = cv2.resize(motion_mask, (smoke_mask.shape[1], smoke_mask.shape[0]))
        smoke_mask = cv2.bitwise_and(smoke_mask, motion_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

        smoke_pixels = float(np.count_nonzero(smoke_mask))
        smoke_frac = smoke_pixels / total_pixels

        # Require at least 5% pixel fraction (up from 3%) to reduce false positives
        effective_min_frac = max(SMOKE_MIN_PIXEL_FRAC, 0.05)
        if smoke_frac < effective_min_frac:
            return False, 0.0, ""

        local_mean = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 0)
        local_var = cv2.GaussianBlur((gray.astype(np.float32) - local_mean) ** 2, (31, 31), 0)
        smoke_region_var = float(np.mean(local_var[smoke_mask > 0])) if np.any(smoke_mask > 0) else 999.0

        contrast_drop = False
        if baseline_metrics and baseline_metrics.get("std", 0) > 0:
            current_std = float(np.std(gray))
            baseline_std = float(baseline_metrics["std"])
            if current_std / (baseline_std + 1e-6) < SMOKE_CONTRAST_DROP_RATIO:
                contrast_drop = True

        confidence = min(1.0, smoke_frac / 0.15) * 0.4
        if smoke_region_var < 200:
            confidence += 0.3
        if contrast_drop:
            confidence += 0.3
        confidence = min(1.0, confidence)

        # Raise confidence bar to 0.45 (from 0.3) to reduce false positives
        if confidence < 0.45:
            return False, 0.0, ""

        reason = (
            f"Smoke CV: pixel_frac={smoke_frac:.4f}, region_var={smoke_region_var:.1f}, "
            f"contrast_drop={contrast_drop}, conf={confidence:.2f}"
        )
        return True, confidence, reason

    except Exception as e:
        return False, 0.0, f"Smoke CV error: {e}"


def detect_camera_covered(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    template_gray: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    """
    Detect lens-cover conditions (hand, cloth, tape, cardboard, etc.)
    Separate from camera_blocked (which is black/white/signal loss).
    Uses scene collapse + skin-tone / texture heuristics.
    """
    try:
        if frame_bgr is None or gray is None or gray.size == 0:
            return False, ""

        is_collapsed, collapse_reason, dbg = detect_scene_collapse(
            frame_bgr, gray, template_gray,
            require_triggers=2,
        )

        if not is_collapsed:
            return False, ""

        m = _frame_metrics(gray)
        mean_val = float(m["mean"])
        std_val = float(m["std"])

        if std_val < 4.0 and (mean_val < 18.0 or mean_val > 238.0):
            return False, ""

        if std_val >= 3.0 and 15.0 < mean_val < 240.0:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h_ch = hsv[:, :, 0]
            s_ch = hsv[:, :, 1]
            v_ch = hsv[:, :, 2]

            skin_mask = (
                (h_ch >= 5) & (h_ch <= 25) &
                (s_ch >= 30) & (s_ch <= 180) &
                (v_ch >= 60) & (v_ch <= 255)
            )
            skin_frac = float(np.count_nonzero(skin_mask)) / float(gray.size)

            brown_mask = (
                (h_ch >= 10) & (h_ch <= 30) &
                (s_ch >= 40) & (s_ch <= 200) &
                (v_ch >= 40) & (v_ch <= 180)
            )
            brown_frac = float(np.count_nonzero(brown_mask)) / float(gray.size)

            if skin_frac > 0.25:
                return True, f"Camera covered (hand/skin detected: skin_frac={skin_frac:.2f}) | {collapse_reason}"
            if brown_frac > 0.30:
                return True, f"Camera covered (cardboard/paper: brown_frac={brown_frac:.2f}) | {collapse_reason}"

            return True, f"Camera covered (scene collapse) | {collapse_reason}"

        return False, ""

    except Exception as e:
        return False, f"Camera covered check error: {e}"

# =============================================================================
# IMPORTANT LABEL POLICY (FINAL)
# - corrupted_frame: ONLY for true RTSP/decoder corruption (macroblocks/packet loss/invalid buffers/shape mismatch)
# - camera_blocked: includes blank/black/white screens and lens covers/occlusions
# - blur_frame: focus loss (baseline-aware, night-safe)
# =============================================================================


def _frame_metrics(gray: np.ndarray) -> Dict[str, float]:
    """Compute a compact set of metrics used for blur/blocked decisions."""
    g = _roi_for_metrics(gray)
    mean_val = float(np.mean(g))
    std_val  = float(np.std(g))
    lap      = float(cv2.Laplacian(g, cv2.CV_64F).var())
    ten      = _tenengrad(g)
    er       = _edge_ratio(g)
    # entropy for low-detail / fog
    g8 = g.astype(np.uint8, copy=False)
    hist = np.bincount(g8.ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, float(np.sum(hist)))
    p = p[p > 0]
    ent = float(-np.sum(p * np.log2(p))) if p.size else 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "lap": lap,
        "ten": float(ten),
        "edge": float(er),
        "entropy": ent,
    }



# =============================================================================
# NEW: Robust lens-cover / scene-collapse detector (hand/cloth/paper/paint/etc.)
# =============================================================================

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Safe correlation, returns 0.0 if variance is too small or invalid."""
    try:
        a = a.astype(np.float32, copy=False).ravel()
        b = b.astype(np.float32, copy=False).ravel()
        if a.size < 32 or b.size < 32:
            return 0.0
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isnan(c) or np.isinf(c):
            return 0.0
        return c
    except Exception:
        return 0.0


def detect_scene_collapse(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    template_gray: Optional[np.ndarray] = None,
    *,
    entropy_thr: float = 2.2,
    std_thr: float = 8.0,
    corr_thr: float = 0.98,
    kp_thr: int = 15,
    template_diff_thr: float = 4.0,
    require_triggers: int = 2,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Detect lens-covered / scene-collapse conditions.

    Why this exists:
    - Hand/cloth/paper covers are NOT always "blank" and NOT always "blur".
    - They often keep texture/edges but the scene becomes semantically collapsed.

    Strategy:
    Trigger when >= require_triggers of the following are true:
      1) entropy very low
      2) per-channel std very low
      3) RGB channel correlation very high (monotone surface)
      4) ORB keypoints collapse
      5) optional: frame very close to template (diff mean tiny)

    Returns:
      (is_collapsed, reason, debug_dict)
    """
    dbg: Dict[str, Any] = {}
    try:
        if frame_bgr is None or gray is None or gray.size == 0:
            return True, "Empty frame/gray (scene collapse)", {"empty": True}

        # Work on ROI (avoid OSD borders)
        g = _roi_for_metrics(gray)

        # Entropy (on ROI)
        g8 = g.astype(np.uint8, copy=False)
        hist = np.bincount(g8.ravel(), minlength=256).astype(np.float64)
        p = hist / max(1.0, float(np.sum(hist)))
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p))) if p.size else 0.0

        # Color stats + correlations on aligned ROI region in BGR
        h, w = gray.shape[:2]
        y1 = int(h * 0.08)
        y2 = int(h * 0.85)
        x1 = int(w * 0.05)
        x2 = int(w * 0.95)
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))

        roi_bgr = frame_bgr[y1:y2, x1:x2]
        if roi_bgr is None or roi_bgr.size == 0:
            roi_bgr = frame_bgr
        b, gch, r = cv2.split(roi_bgr)

        std_b = float(np.std(b))
        std_g = float(np.std(gch))
        std_r = float(np.std(r))

        corr_rg = _safe_corr(r, gch)
        corr_gb = _safe_corr(gch, b)

        # ORB keypoints collapse
        orb = cv2.ORB_create(500)
        kps = orb.detect(g, None)
        kp_count = int(len(kps)) if kps is not None else 0

        triggers = 0
        t_entropy = (entropy < entropy_thr)
        t_std = (std_r < std_thr and std_g < std_thr and std_b < std_thr)
        t_corr = (corr_rg > corr_thr and corr_gb > corr_thr)
        t_kp = (kp_count < kp_thr)

        triggers += int(t_entropy)
        triggers += int(t_std)
        triggers += int(t_corr)
        triggers += int(t_kp)

        t_tpl = False
        tpl_diff_mean = None
        if template_gray is not None and isinstance(template_gray, np.ndarray) and template_gray.size > 0:
            try:
                tg = _roi_for_metrics(template_gray)
                if tg.shape != g.shape:
                    tg = cv2.resize(tg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
                diff = cv2.absdiff(g, tg)
                tpl_diff_mean = float(np.mean(diff))
                t_tpl = (tpl_diff_mean < template_diff_thr)
                triggers += int(t_tpl)
            except Exception:
                t_tpl = False

        dbg.update({
            "entropy": entropy,
            "std": (std_r, std_g, std_b),
            "corr": (corr_rg, corr_gb),
            "kp": kp_count,
            "tpl_diff_mean": tpl_diff_mean,
            "triggers": triggers,
            "flags": {
                "entropy_low": t_entropy,
                "std_low": t_std,
                "corr_high": t_corr,
                "kp_low": t_kp,
                "tpl_diff_low": t_tpl,
            }
        })

        if triggers >= int(require_triggers):
            reason = (
                f"Scene collapse (triggers={triggers}/{require_triggers}): "
                f"entropy={entropy:.2f}, std={std_r:.1f}/{std_g:.1f}/{std_b:.1f}, "
                f"corr_rg={corr_rg:.3f}, corr_gb={corr_gb:.3f}, kp={kp_count}"
                + (f", tpl_diff={tpl_diff_mean:.2f}" if tpl_diff_mean is not None else "")
            )
            return True, reason, dbg

        return False, "", dbg

    except Exception as e:
        # Fail-safe: do not label blocked on exception; return debug only
        return False, f"Scene collapse check error: {e}", {"error": str(e)}


def detect_block_blank_blur(
    gray: np.ndarray,
    now_dt: Optional[datetime.datetime] = None,
    baseline_lap: Optional[float] = None,
    baseline_sharpness: Optional[float] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[str], str]:
    """
    Detect hard-failure frames: blocked (includes blank) / blur.

    - Blocked (includes blank/covered): very low std and mean near 0 or 255 OR very low texture/entropy
    - Blur: adaptive threshold vs baseline, with stricter gate at night to avoid IR false positives.
    """
    if now_dt is None:
        now_dt = datetime.datetime.now()
    hr = int(getattr(now_dt, "hour", 12))
    is_night = (hr >= NIGHT_START_HOUR) or (hr < NIGHT_END_HOUR)

    m = _frame_metrics(gray)
    mean_val = float(m["mean"])
    std_val  = float(m["std"])
    lap      = float(m["lap"])
    ten      = float(m["ten"])
    er       = float(m["edge"])
    ent      = float(m["entropy"])

    # blank: uniform black/white
    if std_val < 4.0 and (mean_val < 18.0 or mean_val > 238.0):
        return ("camera_blocked", f"Blank/covered-like frame (mean={mean_val:.1f}, std={std_val:.1f})")

    # blocked/occluded-ish: very low detail but not strictly blank
    if (er < 0.0025) and (lap < 20.0) and (std_val < 14.0) and (ent < 3.8) and (10.0 <= mean_val <= 245.0):
        return ("camera_blocked", f"Blocked/occluded (edge={er:.5f}, lap={lap:.1f}, std={std_val:.1f}, ent={ent:.2f})")

    # ---- adaptive blur threshold (baseline-aware) ----
    base = None
    if isinstance(baseline_metrics, dict) and float(baseline_metrics.get("lap", 0.0)) > 0:
        base = float(baseline_metrics.get("lap", 0.0))
    elif baseline_sharpness and baseline_sharpness > 0:
        base = float(baseline_sharpness)
    elif baseline_lap and baseline_lap > 0:
        base = float(baseline_lap)

    # Defaults
    blur_lap_thr = 10.0
    blur_ten_thr = 120.0
    std_gate = 10.0
    ent_gate = 4.0

    if base and base > 0:
        b = float(base)
        blur_lap_thr = max(3.0, b * 0.35)
        if isinstance(baseline_metrics, dict) and float(baseline_metrics.get("ten", 0.0)) > 0:
            bt = float(baseline_metrics.get("ten", 0.0))
            blur_ten_thr = max(60.0, bt * 0.35)

        if isinstance(baseline_metrics, dict):
            bstd = float(baseline_metrics.get("std", 0.0))
            bent = float(baseline_metrics.get("entropy", 0.0))
            std_gate = max(8.0, bstd * 0.55) if bstd > 0 else std_gate
            ent_gate = max(3.5, bent * 0.80) if bent > 0 else ent_gate

    # Apply a stricter blur gate at night to avoid IR/noise false positives.
    eff_blur_lap_thr = blur_lap_thr
    eff_blur_ten_thr = blur_ten_thr
    eff_edge_thr = 0.020
    if is_night:
        eff_blur_lap_thr = min(eff_blur_lap_thr, NIGHT_MAX_LAP_FOR_BLUR)
        eff_blur_ten_thr = min(eff_blur_ten_thr, NIGHT_MAX_TEN_FOR_BLUR)
        eff_edge_thr = min(eff_edge_thr, NIGHT_MAX_EDGE_FOR_BLUR)

    # =============================================================================
    # UPDATED BLUR DECISION (Night false positives fix)
    # =============================================================================
    # Night IR/noise scenes naturally have lower lap/edge values even when NOT blur.
    # So, at night we also require a DROP vs baseline (lap/edge/std ratios).
    #
    # This prevents: "clear but low-detail IR scene" being labeled as blur.
    # =============================================================================

    # Baseline edge/std for ratio checks (if available)
    base_edge = None
    base_std = None
    if isinstance(baseline_metrics, dict):
        try:
            be = float(baseline_metrics.get("edge", 0.0) or 0.0)
            bs = float(baseline_metrics.get("std", 0.0) or 0.0)
            if be > 0:
                base_edge = be
            if bs > 0:
                base_std = bs
        except Exception:
            base_edge = None
            base_std = None

    # Ratio gates (night stricter)
    lap_drop_ok = True
    edge_drop_ok = True
    std_drop_ok = True

    if base and base > 0:
        # At night, require more drop vs baseline to call blur
        lap_drop_ok = (lap / (base + 1e-6)) <= (0.55 if is_night else 0.70)

    if base_edge and base_edge > 0:
        edge_drop_ok = (er / (base_edge + 1e-9)) <= (0.55 if is_night else 0.75)

    if base_std and base_std > 0:
        std_drop_ok = (std_val / (base_std + 1e-9)) <= (0.80 if is_night else 0.90)

    # blur decision
    if (
        # mean gate prevents weird over/under exposure frames
        (10.0 <= mean_val <= 245.0)
        and (std_val >= 4.0)          # relaxed: allow low-contrast night scenes
        and (ent >= 2.6)              # relaxed entropy gate
        and (lap < eff_blur_lap_thr)
        and (ten < eff_blur_ten_thr)
        and (er <= eff_edge_thr)
        and lap_drop_ok
        and edge_drop_ok
        and std_drop_ok
    ):
        # If baseline says edges/contrast look OK, do not call blur.
        if isinstance(baseline_metrics, dict):
            try:
                bedge = float(baseline_metrics.get("edge", 0.0) or 0.0)
                bstd = float(baseline_metrics.get("std", 0.0) or 0.0)

                # If current edge or std is still close to baseline => likely NOT blur
                if bedge > 0 and er > (bedge * 0.60):
                    return (None, "")
                if bstd > 0 and std_val > (bstd * 0.92):
                    return (None, "")
            except Exception:
                pass

        return (
            "blur_frame",
            f"Blur suspected ({'night' if is_night else 'day'}; "
            f"lap={lap:.1f}<{eff_blur_lap_thr:.1f}, "
            f"edge={er:.5f}<={eff_edge_thr:.5f}, "
            f"ten={ten:.1f}<{eff_blur_ten_thr:.1f}, "
            f"std={std_val:.1f})",
        )

    return (None, "")


def update_and_check_transition(camera_state: dict, gray: np.ndarray, now_ts: float) -> bool:
    """Detect dusk/dawn IR-cut + auto-exposure transition and open a short suppression window.

    Why: at sunset/sunrise many cameras switch color->IR (or IR->color), focus shifts,
    contrast/edges drop, and auto-gain spikes. That often triggers false BLUR/BLOCKED/MOVED.

    Returns True if we are currently inside a suppression window.
    """
    try:
        if camera_state is None:
            return False

        m = _frame_metrics(gray)
        mean_val = float(m.get("mean", 0.0))
        lap_val  = float(m.get("lap", 0.0))

        prev_mean = float(camera_state.get("prev_mean", mean_val))
        prev_lap  = float(camera_state.get("prev_lap", lap_val))

        mean_jump = abs(mean_val - prev_mean)
        lap_drop_vs_prev = (lap_val / (prev_lap + 1e-6)) if prev_lap > 0 else 1.0

        # Already in suppression window? Extend if still unstable.
        until = float(camera_state.get("transition_until", 0.0) or 0.0)
        if now_ts < until:
            camera_state["prev_mean"] = float(mean_val)
            camera_state["prev_lap"]  = float(lap_val)

            unstable = (
                (mean_jump >= float(TRANSITION_MEAN_DELTA) * 0.60)
                or (lap_drop_vs_prev <= float(TRANSITION_LAP_DROP_FRAC) * 1.15)
                or (lap_val < 0.50 * max(prev_lap, lap_val))
            )

            if unstable:
                start_ts = float(camera_state.get("transition_start_ts", now_ts))
                max_until = start_ts + float(TRANSITION_SUPPRESS_S * 2.0)
                new_until = min(max_until, now_ts + float(TRANSITION_SUPPRESS_S * 0.5))
                if new_until > until:
                    camera_state["transition_until"] = new_until
                    camera_state["transition_last_reason"] = "Extended IR/AE transition (still unstable)"
            return True

        # If we don't have history, don't trigger
        if prev_lap <= 0:
            camera_state["prev_mean"] = float(mean_val)
            camera_state["prev_lap"]  = float(lap_val)
            return False

        # Compare vs baseline if available
        base_lap = None
        base_mean = None
        try:
            bm = camera_state.get("baseline_metrics") or {}
            bl = float(bm.get("lap", 0.0))
            if bl > 0:
                base_lap = bl
            # store baseline mean if available (helps distinguish IR transition from manual blur)
            bm_mean = bm.get("mean", None)
            if bm_mean is not None:
                try:
                    base_mean = float(bm_mean)
                except Exception:
                    base_mean = None
        except Exception:
            base_lap = None
            base_mean = None

        lap_drop_vs_base = None
        if base_lap and base_lap > 0:
            lap_drop_vs_base = (lap_val / (base_lap + 1e-6))

        # Transition signature:
        #  - mean brightness changes suddenly (AE/AGC)
        #  - laplacian drops sharply (IR-cut + focus shift + contrast loss)
        cond_prev = (mean_jump >= float(TRANSITION_MEAN_DELTA)) and (lap_drop_vs_prev <= float(TRANSITION_LAP_DROP_FRAC))
        cond_base = False
        if lap_drop_vs_base is not None:
            cond_base = (
                (lap_drop_vs_base <= float(TRANSITION_BASELAP_DROP_FRAC))
                and (35.0 <= mean_val <= 140.0)
                and (base_mean is not None)
                and (abs(mean_val - float(base_mean)) >= float(TRANSITION_MEAN_DELTA))
            )

        if cond_prev or cond_base:
            camera_state["transition_start_ts"] = float(now_ts)
            camera_state["transition_until"] = float(now_ts + float(TRANSITION_SUPPRESS_S))
            camera_state["transition_last_reason"] = (
                f"IR/AE transition: mean_jump={mean_jump:.1f}, lap={lap_val:.1f}, prev_lap={prev_lap:.1f}"
                + (f", base_lap={base_lap:.1f}" if base_lap else "")
            )
            camera_state["prev_mean"] = float(mean_val)
            camera_state["prev_lap"]  = float(lap_val)
            return True

        camera_state["prev_mean"] = float(mean_val)
        camera_state["prev_lap"]  = float(lap_val)
        return False
    except Exception:
        return False


def occlusion_candidate(gray: np.ndarray) -> Tuple[bool, str]:
    """Heuristic for lens occlusion candidates. Does NOT label; used only to force VLM."""
    m = _frame_metrics(gray)
    mean_val = m["mean"]
    std_val = m["std"]
    lap = m["lap"]
    er = m["edge"]
    ent = m["entropy"]
    if (er < 0.0030) and (lap < 25.0) and (std_val < 18.0) and (ent < 4.2) and (10.0 <= mean_val <= 245.0):
        return True, f"Occlusion candidate (edge={er:.5f}, lap={lap:.1f}, std={std_val:.1f}, ent={ent:.2f})"
    return False, ""


def partial_corruption_candidate(frame_rgb: np.ndarray, gray: np.ndarray) -> Tuple[bool, str]:
    """
    Detect partial stream corruption (macroblocks/bands/tearing).
    This is common in RTSP packet loss and MUST be treated as corrupted_frame.
    """
    try:
        if gray is None or gray.size == 0:
            return True, "Empty gray frame"

        h, w = gray.shape[:2]
        if h < 200 or w < 200:
            return True, f"Tiny frame {gray.shape}"

        # Focus on bottom band where artifacts often appear (timestamp OSD excluded by ROI)
        band_y1 = int(h * 0.55)
        band = gray[band_y1:, :]
        if band.size < 10:
            return False, ""

        # Blockiness ratio
        bscore = _blockiness_score(band, block=8)

        # Also detect extreme color "mosaic" by measuring high local variance on small downsample
        small = cv2.resize(band, (min(160, w), min(90, h - band_y1)), interpolation=cv2.INTER_AREA)
        lap = float(cv2.Laplacian(small, cv2.CV_64F).var())
        std = float(np.std(small))
        ent = _frame_metrics(band).get("entropy", 0.0)

        # Heuristic gates:
        #  - strong blockiness AND non-trivial texture/variance (not just blank)
        #  - OR extremely high laplacian variance on the band (noisy tiles)
        if (bscore >= 2.2 and std >= 6.0 and ent >= 3.0) or (lap >= 12000.0 and std >= 8.0):
            return True, f"Partial corruption (blockiness={bscore:.2f}, lap={lap:.0f}, std={std:.1f}, ent={ent:.2f})"

        return False, ""
    except Exception as e:
        return True, f"Corruption check error: {e}"


def _entropy(gray: np.ndarray) -> float:
    """Shannon entropy on ROI (0..8-ish). Low entropy can be blank/covered."""
    try:
        g = _roi_for_metrics(gray)
        g8 = g.astype(np.uint8, copy=False)
        hist = np.bincount(g8.ravel(), minlength=256).astype(np.float64)
        s = float(np.sum(hist))
        if s <= 0.0:
            return 0.0
        p = hist / s
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p))) if p.size else 0.0
    except Exception:
        return 0.0


def is_real_corrupted_frame(frame_rgb: np.ndarray, gray: np.ndarray) -> Tuple[bool, str]:
    """
    STRICT corruption detector.
    Returns True only for truly unusable frames / stream glitches.
    """
    if frame_rgb is None or gray is None:
        return True, "Empty frame/gray"
    if not isinstance(gray, np.ndarray) or gray.size == 0:
        return True, "Invalid gray buffer"
    if np.isnan(gray).any() or np.isinf(gray).any():
        return True, "NaN/Inf pixels"

    h, w = gray.shape[:2]
    if h < 200 or w < 200:
        return True, f"Tiny frame {gray.shape}"

    # Very strict degenerate-frame check (avoid night/IR false positives)
    std_val = float(np.std(gray))
    ent = _entropy(gray)
    if std_val < 0.35 and ent < 0.15:
        return True, f"Degenerate frame (std={std_val:.3f}, ent={ent:.3f})"

    return False, ""

def is_corrupted_scene(frame_rgb: np.ndarray, gray: np.ndarray, camera_name: str, camera_states: dict) -> bool:
    # -------------------------------------------------------------------------
    # STRICT corruption only (do NOT treat low-light / low-contrast as corruption)
    # -------------------------------------------------------------------------
    is_corr, reason = is_real_corrupted_frame(frame_rgb, gray)
    if is_corr:
        print(f"[CORRUPT] {reason} on {camera_name}")
        return True

    # Shape mismatch is a strong stream anomaly (keep as corruption)
    st = camera_states.get(camera_name)
    if st is not None:
        base_shape = st.get("frame_shape")
        if base_shape is not None and tuple(base_shape) != tuple(gray.shape):
            print(f"[CORRUPT] Shape mismatch on {camera_name}: got {gray.shape}, expected {base_shape}")
            return True

    # Detect partial RTSP corruption (macroblocks/bands). Must be treated as corrupted_frame.
    cand, creason = partial_corruption_candidate(frame_rgb, gray)
    if cand:
        print(f"[CORRUPT] {creason} on {camera_name}")
        return True

    # -------------------------------------------------------------------------
    # "Noisy codec garbage" detector (keep it, but make it stricter to avoid IR)
    # -------------------------------------------------------------------------
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    ent = _entropy(gray)

    # Exclude blank-like frames from corruption (blank is handled elsewhere)
    if std_val < 4.0 and (mean_val < 20.0 or mean_val > 235.0):
        return False

    # NOTE: Removed the old 'noisy codec garbage' heuristic because it can false-trigger on IR/noisy night scenes.
    # Partial RTSP corruption is already handled above by partial_corruption_candidate().

    # IMPORTANT:
    # Removed the old rule that flagged flat mid-grey frames as corrupted.
    # Night/IR/low-contrast scenes can look like that but are valid frames.

    return False



def _roi_for_metrics(gray: np.ndarray) -> np.ndarray:
    """Crop away timestamp/labels borders for more stable metrics.

    Defensive: if the frame is unusually small, return as-is.
    """
    if gray is None:
        return gray
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return gray
    y1 = int(h * 0.08)
    y2 = int(h * 0.85)  # cut bottom text band
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    return gray[y1:y2, x1:x2]


# ===================== CORRUPTION / MACROBLOCK ARTIFACT DETECTION =====================

def _blockiness_score(gray: np.ndarray, block: int = 8) -> float:
    """
    Estimate blockiness (macroblock artifacts) by comparing average absolute
    differences across block boundaries vs. non-boundary positions.
    Returns a ratio; higher means more blocky.
    """
    try:
        g = _roi_for_metrics(gray)
        h, w = g.shape[:2]
        if h < (block * 4) or w < (block * 4):
            return 0.0

        g = g.astype(np.int16)

        # Vertical boundaries at multiples of block (columns)
        cols = np.arange(block, w, block)
        if cols.size == 0:
            return 0.0
        vbd = np.mean(np.abs(g[:, cols] - g[:, cols - 1]))
        vnd = np.mean(np.abs(g[:, 1:] - g[:, :-1]))

        # Horizontal boundaries at multiples of block (rows)
        rows = np.arange(block, h, block)
        hbd = np.mean(np.abs(g[rows, :] - g[rows - 1, :]))
        hnd = np.mean(np.abs(g[1:, :] - g[:-1, :]))

        denom = (vnd + hnd) * 0.5 + 1e-6
        numer = (vbd + hbd) * 0.5
        return float(numer / denom)
    except Exception:
        return 0.0


# ===================== HARD GUARDS: blocked/blank/blur =====================

def _edge_ratio(gray: np.ndarray) -> float:
    e = cv2.Canny(gray, 60, 180)
    return float(np.mean(e > 0))


def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad(gray: np.ndarray) -> float:
    # Gradient magnitude energy (robust focus measure)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))






