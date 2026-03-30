"""
behavior_analyzer.py — Rule-based suspicious behavior detection
Smart Retail Theft Detection System
"""

import math
import time
import numpy as np
from collections import defaultdict, deque


# ── Tuneable thresholds ──────────────────────────────────────────────────────
LOITER_SECONDS        = 8.0      # dwell time before loitering alert
LOITER_RADIUS_PX      = 80       # max displacement to count as "standing"
VELOCITY_SPIKE        = 18.0     # px/frame threshold for sudden movement
CROUCHING_RATIO       = 0.55     # bbox height/width ratio below which = crouching
QUICK_BEND_ANGLE      = 110      # knee angle below this = bending / crouching
CONCEALMENT_FRAMES    = 12       # frames hand stays at torso = concealment
HAND_TORSO_RATIO      = 0.30     # wrist y relative to shoulder-hip distance


class BehaviorAnalyzer:
    """
    Stateful per-track behavior analyser.
    Call update() every frame; get_alerts() returns active alerts.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._state = defaultdict(lambda: {
            "positions": deque(maxlen=300),
            "first_seen": None,
            "velocities": deque(maxlen=30),
            "hand_torso_count": 0,
            "alerts": set(),
            "alert_history": [],
        })

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, track_id: int, bbox: list, frame_id: int, pose_data: dict = None) -> list:
        """
        Process one track for one frame.
        Returns list of new alert dicts triggered this frame.
        """
        s = self._state[track_id]
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

        if s["first_seen"] is None:
            s["first_seen"] = frame_id

        if s["positions"]:
            prev = s["positions"][-1]
            vel = math.hypot(cx - prev[0], cy - prev[1])
            s["velocities"].append(vel)
        else:
            s["velocities"].append(0.0)

        s["positions"].append((cx, cy))

        new_alerts = []
        new_alerts += self._check_loitering(track_id, s, frame_id)
        new_alerts += self._check_sudden_movement(track_id, s)
        new_alerts += self._check_crouching(track_id, s, bbox)
        if pose_data:
            new_alerts += self._check_pose_concealment(track_id, s, pose_data, bbox)

        for a in new_alerts:
            s["alert_history"].append(a)

        return new_alerts

    def get_all_alerts(self) -> list:
        """Return all alerts across all tracks."""
        alerts = []
        for tid, s in self._state.items():
            alerts.extend(s["alert_history"])
        return sorted(alerts, key=lambda x: x["frame"])

    def get_active_alerts(self, frame_id: int, window: int = 60) -> list:
        """Return alerts that fired within the last `window` frames."""
        result = []
        for tid, s in self._state.items():
            for a in s["alert_history"]:
                if frame_id - a["frame"] <= window:
                    result.append(a)
        return result

    def track_risk_score(self, track_id: int) -> float:
        """0-100 composite risk score for a track."""
        s = self._state.get(track_id)
        if not s:
            return 0.0
        weights = {"LOITERING": 30, "SUDDEN_MOVEMENT": 20,
                   "CROUCHING": 25, "CONCEALMENT": 40}
        score = 0.0
        seen = set()
        for a in s["alert_history"]:
            k = a["type"]
            if k not in seen:
                score += weights.get(k, 10)
                seen.add(k)
        return min(score, 100.0)

    # ── Internal checks ─────────────────────────────────────────────────────

    def _check_loitering(self, tid, s, frame_id):
        dwell_frames = frame_id - s["first_seen"]
        dwell_secs = dwell_frames / self.fps
        if dwell_secs < LOITER_SECONDS:
            return []
        if len(s["positions"]) < 10:
            return []
        xs = [p[0] for p in list(s["positions"])[-int(LOITER_SECONDS * self.fps):]]
        ys = [p[1] for p in list(s["positions"])[-int(LOITER_SECONDS * self.fps):]]
        disp = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if disp <= LOITER_RADIUS_PX and "LOITERING" not in s["alerts"]:
            s["alerts"].add("LOITERING")
            return [self._make_alert(tid, "LOITERING", frame_id,
                                     f"Loitering {dwell_secs:.1f}s in restricted zone", "HIGH")]
        return []

    def _check_sudden_movement(self, tid, s):
        if len(s["velocities"]) < 5:
            return []
        recent = list(s["velocities"])[-5:]
        avg_prev = np.mean(list(s["velocities"])[:-5]) if len(s["velocities"]) > 5 else 0
        if max(recent) > VELOCITY_SPIKE and max(recent) > avg_prev * 3:
            frame_id = len(s["positions"])
            if "SUDDEN_MOVEMENT" not in s["alerts"]:
                s["alerts"].add("SUDDEN_MOVEMENT")
                return [self._make_alert(tid, "SUDDEN_MOVEMENT", frame_id,
                                          "Sudden rapid movement detected", "MEDIUM")]
        return []

    def _check_crouching(self, tid, s, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w == 0:
            return []
        ratio = h / w
        frame_id = len(s["positions"])
        if ratio < CROUCHING_RATIO and "CROUCHING" not in s["alerts"]:
            s["alerts"].add("CROUCHING")
            return [self._make_alert(tid, "CROUCHING", frame_id,
                                      f"Person crouching (h/w={ratio:.2f})", "HIGH")]
        return []

    def _check_pose_concealment(self, tid, s, pose_data, bbox):
        angles = pose_data.get("angles", {})
        lms = pose_data.get("landmarks", [])
        if not lms or len(lms) < 29:
            return []

        left_knee = angles.get("left_knee", 180)
        right_knee = angles.get("right_knee", 180)

        # Wrist near torso
        left_wrist = lms[15]
        right_wrist = lms[16]
        left_hip = lms[23]
        right_hip = lms[24]
        left_shoulder = lms[11]
        right_shoulder = lms[12]

        torso_h = abs(((left_shoulder[1] + right_shoulder[1]) / 2) -
                      ((left_hip[1] + right_hip[1]) / 2)) + 1e-6
        mid_torso_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4

        lw_near = abs(left_wrist[1] - mid_torso_y) / torso_h < HAND_TORSO_RATIO
        rw_near = abs(right_wrist[1] - mid_torso_y) / torso_h < HAND_TORSO_RATIO

        if lw_near or rw_near:
            s["hand_torso_count"] += 1
        else:
            s["hand_torso_count"] = max(0, s["hand_torso_count"] - 1)

        frame_id = len(s["positions"])
        if s["hand_torso_count"] >= CONCEALMENT_FRAMES and "CONCEALMENT" not in s["alerts"]:
            s["alerts"].add("CONCEALMENT")
            return [self._make_alert(tid, "CONCEALMENT", frame_id,
                                      "Possible item concealment (hands near torso)", "CRITICAL")]

        if (left_knee < QUICK_BEND_ANGLE or right_knee < QUICK_BEND_ANGLE) and \
                "CROUCHING" not in s["alerts"]:
            s["alerts"].add("CROUCHING")
            return [self._make_alert(tid, "CROUCHING", frame_id,
                                      "Crouching posture detected via pose", "HIGH")]
        return []

    @staticmethod
    def _make_alert(track_id, alert_type, frame_id, message, severity):
        return {
            "track_id": track_id,
            "type": alert_type,
            "frame": frame_id,
            "message": message,
            "severity": severity,
            "timestamp": time.strftime("%H:%M:%S"),
        }
