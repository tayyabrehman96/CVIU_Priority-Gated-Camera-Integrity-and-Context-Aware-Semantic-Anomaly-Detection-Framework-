# config.example.py — copy to `config.py` before running (local `config.py` is not committed).
#   Windows PowerShell: Copy-Item config.example.py config.py
#   bash: cp config.example.py config.py
#
import os
from dotenv import load_dotenv

load_dotenv()
from typing import Dict, Any, List, Optional


os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"

# ================= CONFIG =================

# ---------- VIDEO FOLDER (OFFLINE MODE) ----------
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER", "videos")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm")
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
PROCESS_FPS = float(os.getenv("PROCESS_FPS", "10.0"))

UI_FRAME_PREVIEW_SKIP = 10
QUEUE_POLL_MS = 10

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "blaifa/InternVL3_5:8b")
try:
    OLLAMA_TEMP = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
except ValueError:
    OLLAMA_TEMP = 0.0

# Regolo (cloud classifier)
REGOLO_API_KEY = os.getenv("REGOLO_API_KEY")
REGOLO_MODEL = os.getenv("REGOLO_MODEL", "gpt-oss-120b")

# LIVE FEED CONTROL
FRAMES_TO_ANALYZE_PER_CAMERA = 20

MOTION_THRESHOLD_SCORE = 2.0
ANOMALY_THRESHOLD_SCORE = 60.0
QUALITY_DROP_THRESHOLD = 0.50
STABILITY_THRESHOLD = 0.50
ALERT_COOLDOWN_SECONDS = 120.0

SLOT_DEFAULT_MOTION_THR = {
    "night": 3.0,
    "morning": 2.0,
    "afternoon": 2.0,
    "evening": 2.5,
}

# ---- Adaptive motion threshold tuning ----
ADAPT_MOTION_THR_INTERVAL_S = 30 * 60
ADAPT_MOTION_HIST_MAX = 1200
ADAPT_TARGET_TRIGGER_RATE = 0.08
ADAPT_MIN_MOTION_THR = 1.0
ADAPT_MAX_MOTION_THR = 25.0

MOTION_EMA_ALPHA = 0.40

CALIBRATION_STATE_JSON = "camera_calibration_state.json"
CALIBRATION_MIN_STABLE_FRAMES = 30

YOLO_IGNORE_CONFIDENCE_THRESHOLD = 0.5

WEAPON_MODEL_PATH = os.getenv("WEAPON_MODEL_PATH", "models/weapon/weights/best.pt")
WEAPON_CONFIDENCE_THRESHOLD = 0.50
WEAPON_MIN_BOX_AREA_FRAC = 0.001
WEAPON_MAX_BOX_AREA_FRAC = 0.35
WEAPON_CONFIRM_FRAMES = 2

# ---------- Fire / Smoke CV Detection ----------
FIRE_HSV_LOWER_1 = (0, 80, 180)
FIRE_HSV_UPPER_1 = (25, 255, 255)
FIRE_HSV_LOWER_2 = (160, 80, 180)
FIRE_HSV_UPPER_2 = (180, 255, 255)
FIRE_MIN_PIXEL_FRAC = 0.005
FIRE_MIN_CONTOUR_AREA = 200

SMOKE_GRAY_RANGE = (140, 220)
SMOKE_SATURATION_MAX = 40
SMOKE_MIN_PIXEL_FRAC = 0.03
SMOKE_CONTRAST_DROP_RATIO = 0.60

# MEMORY / GRAPH
MAX_GRAPH_POINTS = 1000
MAX_ANOMALY_HISTORY = 50
MAX_QUEUE_SIZE = 30

# ---------- 4-times/day templates ----------
TEMPLATE_DIR = "calibration_templates"
TIMESLOTS = [("night", 0), ("morning", 6), ("afternoon", 12), ("evening", 18)]
TEMPLATE_SAMPLES_PER_SLOT = 6
GOOD_FRAME_REQUIRED_CONSECUTIVE = 2
MAX_TEMPLATE_WAIT_S = 24 * 3600

# ---------- Camera Movement Detection (Combined: Homography + VLM) ----------
CAMERA_MOVEMENT_ENABLED = True
CAMERA_MOVEMENT_CHECK_INTERVAL_S = 60
CAMERA_MOVEMENT_MIN_CHANGE_PERCENT = 50
CAMERA_MOVEMENT_USE_SIDE_BY_SIDE = True
REFERENCE_TEMPLATE_DIR = "reference_templates"

HOMOGRAPHY_ROTATION_THRESHOLD_DEG = 5.0
HOMOGRAPHY_TRANSLATION_THRESHOLD_PX = 30
HOMOGRAPHY_SCALE_CHANGE_THRESHOLD = 0.15

# "good frame" constraints
MIN_MEAN_BRIGHTNESS = 8
MAX_MEAN_BRIGHTNESS = 245
MIN_LAPLACIAN_VAR = 6.0
MIN_STDDEV = 3.5

# Night-time blur guard
NIGHT_START_HOUR = 19
NIGHT_END_HOUR   = 7
NIGHT_MAX_LAP_FOR_BLUR = 18.0
NIGHT_MAX_TEN_FOR_BLUR = 450.0
NIGHT_MAX_EDGE_FOR_BLUR = 0.010

# ---- Dusk/Dawn IR transition suppression ----
TRANSITION_SUPPRESS_S = 180.0
TRANSITION_MEAN_DELTA = 18.0
TRANSITION_LAP_DROP_FRAC = 0.45
TRANSITION_BASELAP_DROP_FRAC = 0.40

BLUR_MOTION_GUARD_FACTOR = 0.85
BLUR_TEMPLATE_MIN_SSIM_LIKE = 0.55
BLOCKED_MOTION_GUARD_FACTOR = 0.95
BLOCKED_TEMPLATE_MIN_SSIM_LIKE = 0.50

# ============== AUTO-UNBLOCK CONFIGURATION ==============
AUTO_UNBLOCK_BLOCKED_FRAMES = 3
AUTO_UNBLOCK_BLOCKED_SSIM_THR = 0.45
AUTO_UNBLOCK_BLOCKED_MIN_EDGE = 15.0
AUTO_UNBLOCK_BLOCKED_MIN_ENTROPY = 2.5

AUTO_UNBLOCK_BLUR_FRAMES = 3
AUTO_UNBLOCK_BLUR_SSIM_THR = 0.45
AUTO_UNBLOCK_BLUR_MIN_LAPLACIAN = 40.0

AUTO_UNBLOCK_MOVED_FRAMES = 5
AUTO_UNBLOCK_MOVED_SSIM_THR = 0.50

AUTO_UNBLOCK_MIN_DELAY_S = 10.0

AUTO_UNBLOCK_FALLBACK_S = 300.0
AUTO_UNBLOCK_FALLBACK_SSIM_THR = 0.35


VLM_PROMPT = (
    "You are a video surveillance assistant. Analyze this image and generate a concise operational description focused on the security-related aspects of the camera system.\n"
    "Explicitly assess:\n"
    "1. Camera operational status: Is the camera functional? Is the lens blocked, covered (by hand/cloth/tape/cardboard), blurred, or moved?\n"
    "2. Optical and image quality: clarity, focus, lighting, motion blur, noise, compression artifacts.\n"
    "3. Observed elements: objects, people, vehicles, their relative positions.\n"
    "4. WEAPONS: If any person is holding, pointing, or carrying a weapon (gun, pistol, rifle, knife, bat, machete) or if any weapon-like object is visible, EXPLICITLY state 'weapon visible' and describe it. If NO weapon is visible, state 'no weapon visible'.\n"
    "5. FIRE/SMOKE: If there are flames, fire, burning, or smoke visible anywhere in the scene, EXPLICITLY mention 'fire visible' or 'smoke visible'. If none, state 'no fire or smoke'.\n"
    "6. THREATS: If a person appears to be threatening others with a weapon or engaging in theft/robbery, describe the action.\n"
    "For each element, determine whether the scene provides sufficient visual information to reliably detect the presence or motion of an object.\n"
    "Describe what is happening in terms of detection without speculation.\n"
    "Output ONLY 5–7 sentences."
)


LABEL_COLORS = {
    "fire_or_smoke": "#EF4444",
    "fire_detected": "#DC2626",
    "smoke_detected": "#F87171",
    "smoke_only": "#F87171",
    "flame_only": "#B91C1C",
    "water_leak_flood": "#06B6D4",
    "camera_blocked": "#F97316",
    "camera_covered": "#FB923C",
    "blank_frame": "#6B7280",
    "blur_frame": "#3B82F6",
    "camera_moved": "#EAB308",
    "fight_or_violence": "#B91C1C",
    "abandoned_object": "#F59E0B",
    "lighting_change": "#22C55E",
    "loitering": "#10B981",
    "person_detected": "#3B82F6",
    "face_detected": "#60A5FA",
    "vehicle_detected": "#0EA5E9",
    "vehicle_moving": "#38BDF8",
    "slip_or_fall": "#A78BFA",
    "intrusion": "#EC4899",
    "weapon_detected": "#DC2626",
    "theft_suspect": "#991B1B",
    "Benign": "#16A34A",
    "Unknown": "#9CA3AF",
    "Pending": "#9CA3AF",
    "corrupted_frame": "#7F1D1D",
}

DISPLAY_LABELS = {
    "camera_blocked": "CAMERA BLOCKED",
    "camera_covered": "CAMERA COVERED",
    "blur_frame": "CAMERA BLURRED",
    "camera_moved": "CAMERA MOVED",
    "blank_frame": "BLANK FRAME",
    "corrupted_frame": "CORRUPTED FRAME",
    "fire_or_smoke": "FIRE / SMOKE",
    "fire_detected": "FIRE DETECTED",
    "smoke_detected": "SMOKE DETECTED",
    "water_leak_flood": "WATER LEAK / FLOOD",
    "weapon_detected": "WEAPON DETECTED",
    "theft_suspect": "THEFT / ROBBERY",
    "Unknown": "UNKNOWN",
    "Pending": "PENDING",
    "Benign": "NORMAL",
}

# ============== ANOMALY CATEGORIES ==============
REAL_ANOMALY_LABELS = {
    "camera_blocked",
    "camera_covered",
    "blur_frame",
    "camera_moved",
}

COMPUTED_ANOMALY_LABELS = {
    "fire_or_smoke",
    "fire_detected",
    "smoke_detected",
    "water_leak_flood",
    "blank_frame",
    "vehicle_moving",
    "vehicle_detected",
    "face_detected",
    "person_detected",
    "weapon_detected",
    "theft_suspect",
    "Benign",
    "Unknown",
}

AUTO_EXPORT_LABELS = set(REAL_ANOMALY_LABELS)

CORRUPTED_LABELS = {
    "corrupted_frame",
}

CAT_ALARMS = (
    "camera_blocked",
    "camera_covered",
    "blur_frame",
    "camera_moved",
    "fire_detected",
    "smoke_detected",
    "weapon_detected",
    "theft_suspect",
)

NON_ALARM_LABELS = {"Benign", "Pending", "Unknown"}


CLEAR_VIEW_TOKENS = [
    "clear view", "unobstructed view", "view is clear",
    "nothing blocking the camera", "no obstruction",
    "camera sees the scene clearly", "scene is fully visible",
]

BLOCKED_TOKENS = [
    "lens covered", "lens is covered", "lens blocked", "camera is blocked",
    "covered with tape", "tape on the lens", "tape over the lens",
    "covered with paper", "paper over the lens", "paper taped over the camera",
    "completely black frame", "black screen", "no video feed",
    "no image", "signal lost", "completely white frame", "white screen",
    "view is obscured", "close-up of a texture", "only a close-up surface",
    "cardboard", "colored surface", "solid surface", "uniform color",
    "featureless", "monotone", "no detail visible", "no scene visible",
    "obstructed view", "limited visibility", "nothing visible",
    "no objects visible", "cannot see scene", "no discernible elements",
    "insufficient visual information", "lacks clarity",
    "no discernible objects", "scene lacks clarity", "lack of clear visual",
    "does not provide sufficient visual", "insufficient detail",
    "very poor due to low lighting", "poor image quality", "no reliable detection",
    "indistinct shapes", "lacks distinct objects", "lack of clear definition",
    "limited visual information", "obscure finer details",
    "lack of sufficient detail", "outlines and specific attributes are unclear",
    "grainy texture that may obscure", "challenges in reliably detecting",
    "lacks sufficient detail", "difficulty in distinguishing",
    "limited visual information to confidently", "questionable for precise detection",
    "lack of sharpness, resulting in a grainy", "may obscure finer details",
]

CATS: Dict[str, Dict[str, Any]] = {
    "camera_blocked": {
        "w": 1.9,
        "pos": BLOCKED_TOKENS + [
            "very close surface", "extreme close-up", "only texture visible",
            "looking directly at a wall",
        ],
    },
    "camera_covered": {
        "w": 2.1,
        "pos": [
            "camera is covered", "hand covering the lens", "covered by hand",
            "covered by an object", "cloth over lens", "paper over lens",
            "tape on the lens", "tape over the lens", "covered with tape",
            "covered with paper", "cardboard over lens", "hand over camera",
            "palm covering", "fingers over lens", "covered by cloth",
            "fabric covering", "sticker on lens", "spray paint on lens",
        ],
    },
    "blank_frame": {
        "w": 2.3,
        "pos": [
            "blank frame", "fully black", "fully white", "pure black", "pure white",
            "uniform grey", "uniform gray", "no image", "signal lost",
            "no video signal", "black screen", "white screen",
        ],
    },
    "blur_frame": {
        "w": 1.6,
        "pos": [
            "blurry", "out of focus", "defocused", "soft focus",
            "very blurred", "image is blurry", "cannot see clearly",
        ],
    },
    "camera_moved": {
        "w": 1.6,
        "pos": ["camera moved", "repositioned", "view angle changed", "tilting", "tilted", "different viewpoint"],
    },
    "fire_detected": {
        "w": 2.0,
        "pos": ["fire", "flame", "flames", "burning", "blaze", "inferno", "engulfed in flames",
                "on fire", "fire visible", "active fire", "combustion"],
    },
    "smoke_detected": {
        "w": 1.8,
        "pos": ["smoke", "haze", "smoldering", "smoky", "smoke visible", "plume of smoke",
                "smoke rising", "thick smoke", "white smoke", "black smoke", "gray smoke"],
    },
    "fire_or_smoke": {
        "w": 1.7,
        "pos": ["fire and smoke", "fire with smoke", "flames and smoke"],
    },
    "water_leak_flood": {
        "w": 1.8,
        "pos": ["water leak", "leaking", "pool of water", "wet floor", "flood", "puddle"],
    },
    "face_detected": {"w": 1.1, "pos": ["face", "faces", "selfie", "close-up face"]},
    "person_detected": {"w": 1.0, "pos": ["person", "people", "man", "woman", "individual"]},
    "vehicle_detected": {"w": 1.0, "pos": ["vehicle", "car", "truck", "van", "bus", "motorcycle", "bicycle"]},
    "vehicle_moving": {"w": 1.2, "pos": ["moving car", "vehicle moving", "driving", "in motion"]},
    "weapon_detected": {
        "w": 2.5,
        "pos": [
            "gun", "pistol", "rifle", "firearm", "weapon", "knife", "blade",
            "armed", "holding a gun", "pointing a gun", "handgun", "shotgun",
            "machete", "sword", "assault rifle", "submachine gun", "revolver",
            "weapon visible", "carrying a weapon", "weapon in hand",
        ],
    },
    "theft_suspect": {
        "w": 2.3,
        "pos": [
            "robbery", "theft", "stealing", "robber", "thief", "hold-up",
            "threatening with weapon", "armed robbery", "stick-up", "mugging",
            "snatching", "burglary", "shoplifting", "looting",
        ],
    },
    "Benign": {"w": 0.7, "pos": ["normal", "routine", "quiet", "clear view", "no issues"]},
}

PRIORITY = [
    "corrupted_frame",
    "blank_frame",
    "camera_blocked",
    "camera_covered",
    "blur_frame",
    "camera_moved",
    "weapon_detected",
    "theft_suspect",
    "fire_detected",
    "smoke_detected",
    "fire_or_smoke",
    "water_leak_flood",
    "vehicle_moving",
    "vehicle_detected",
    "face_detected",
    "person_detected",
    "Benign",
    "Unknown",
    "Pending",
]
