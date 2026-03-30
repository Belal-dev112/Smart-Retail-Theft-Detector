"""
pose_estimator.py — MediaPipe Pose wrapper (auto-detects API version)
Smart Retail Theft Detection System

Supports:
  - mediapipe >= 0.10.21  → Tasks API (PoseLandmarker)
  - mediapipe 0.8 – 0.10  → Legacy mp.solutions.pose
  - mediapipe missing      → Graceful no-op
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path


# ── Detect available MediaPipe API ───────────────────────────────────────────
_USE_TASKS_API = False
_USE_LEGACY_API = False

try:
    import mediapipe as mp

    # Try new Tasks API first (mediapipe >= 0.10.21)
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        _USE_TASKS_API = True
    except (ImportError, AttributeError):
        pass

    # Fallback: legacy solutions API (mediapipe 0.8 – 0.10.14)
    if not _USE_TASKS_API:
        try:
            _ = mp.solutions.pose
            _USE_LEGACY_API = True
        except AttributeError:
            pass

except ImportError:
    mp = None

MP_AVAILABLE = _USE_TASKS_API or _USE_LEGACY_API


# ── Landmark indices (same across both APIs) ─────────────────────────────────
LANDMARKS = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# Connections for manual skeleton drawing (Tasks API path)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_lite/float16/latest/"
             "pose_landmarker_lite.task")
MODEL_FILENAME = "pose_landmarker_lite.task"


class PoseEstimator:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.available = MP_AVAILABLE
        self._api = None  # "tasks" | "legacy"

        if _USE_TASKS_API:
            self._init_tasks_api(min_detection_confidence)
        elif _USE_LEGACY_API:
            self._init_legacy_api(min_detection_confidence, min_tracking_confidence)
        else:
            print("[Pose] MediaPipe not available — pose features disabled.")

    # ── Initializers ─────────────────────────────────────────────────────────

    def _init_tasks_api(self, confidence):
        """Initialize with new mediapipe.tasks API (>= 0.10.21)."""
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / MODEL_FILENAME

        if not model_path.exists():
            print(f"[Pose] Downloading pose model to {model_path} ...")
            try:
                urllib.request.urlretrieve(MODEL_URL, str(model_path))
                print("[Pose] Download complete.")
            except Exception as e:
                print(f"[Pose] Model download failed: {e}")
                self.available = False
                return

        try:
            base_options = mp_tasks.BaseOptions(
                model_asset_path=str(model_path))
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_pose_detection_confidence=confidence,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self._api = "tasks"
            print("[Pose] MediaPipe Tasks API loaded.")
        except Exception as e:
            print(f"[Pose] Tasks API init failed: {e}")
            self.available = False

    def _init_legacy_api(self, det_conf, track_conf):
        """Initialize with legacy mp.solutions API (0.8 – 0.10.14)."""
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            model_complexity=0,
        )
        self._api = "legacy"
        print("[Pose] MediaPipe legacy (solutions) API loaded.")

    # ── Public API ───────────────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> dict:
        """
        Returns dict with 'landmarks' (list of (x,y,z,vis)) and 'angles'.
        Returns empty dict if unavailable or no pose detected.
        """
        if not self.available:
            return {}
        if self._api == "tasks":
            return self._estimate_tasks(frame)
        return self._estimate_legacy(frame)

    def draw(self, frame: np.ndarray, pose_data: dict) -> np.ndarray:
        """Draw skeleton overlay on frame."""
        if not self.available or not pose_data:
            return frame

        # Legacy API with raw landmarks — use built-in drawing
        if self._api == "legacy" and pose_data.get("raw") is not None:
            overlay = frame.copy()
            self.mp_draw.draw_landmarks(
                overlay, pose_data["raw"],
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2,
                                        circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 128, 0), thickness=2),
            )
            return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Tasks API or fallback: manual skeleton drawing
        lms = pose_data.get("landmarks", [])
        if len(lms) < 29:
            return frame

        overlay = frame.copy()
        for i, j in POSE_CONNECTIONS:
            if i < len(lms) and j < len(lms):
                pt1 = (lms[i][0], lms[i][1])
                pt2 = (lms[j][0], lms[j][1])
                cv2.line(overlay, pt1, pt2, (255, 128, 0), 2)

        for idx in LANDMARKS.values():
            if idx < len(lms):
                cv2.circle(overlay, (lms[idx][0], lms[idx][1]),
                           3, (0, 255, 128), -1)

        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # ── Internal estimation methods ──────────────────────────────────────────

    def _estimate_tasks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return {}

        h, w = frame.shape[:2]
        lms = []
        for lm in result.pose_landmarks[0]:
            lms.append((int(lm.x * w), int(lm.y * h), lm.z, lm.visibility))

        angles = self._compute_angles(lms)
        return {"landmarks": lms, "angles": angles, "raw": None}

    def _estimate_legacy(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return {}

        h, w = frame.shape[:2]
        lms = []
        for lm in results.pose_landmarks.landmark:
            lms.append((int(lm.x * w), int(lm.y * h), lm.z, lm.visibility))

        angles = self._compute_angles(lms)
        return {"landmarks": lms, "angles": angles, "raw": results.pose_landmarks}

    # ── Angle computation ────────────────────────────────────────────────────

    @staticmethod
    def _angle(a, b, c):
        a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    def _compute_angles(self, lms):
        try:
            return {
                "left_elbow": self._angle(lms[11], lms[13], lms[15]),
                "right_elbow": self._angle(lms[12], lms[14], lms[16]),
                "left_knee": self._angle(lms[23], lms[25], lms[27]),
                "right_knee": self._angle(lms[24], lms[26], lms[28]),
                "torso_lean": self._angle(lms[11], lms[23], lms[27]),
            }
        except Exception:
            return {}
