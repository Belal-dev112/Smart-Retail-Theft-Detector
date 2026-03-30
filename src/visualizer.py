"""
visualizer.py — OpenCV drawing utilities for CLI and UI output
Smart Retail Theft Detection System
"""

import cv2
import numpy as np
import time

SEVERITY_COLOR_BGR = {
    "CRITICAL": (0, 0, 220),
    "HIGH":     (0, 140, 255),
    "MEDIUM":   (0, 200, 255),
    "LOW":      (50, 200, 50),
}

TRACK_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 180, 100), (180, 100, 200),
]


def draw_tracks(frame: np.ndarray, tracks: list, risk_scores: dict = None) -> np.ndarray:
    for trk in tracks:
        tid = trk.track_id
        x1, y1, x2, y2 = trk.bbox
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        risk = risk_scores.get(tid, 0) if risk_scores else 0

        # Risk-based border thickness
        thick = 3 if risk > 60 else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        label = f"ID:{tid}"
        if risk > 0:
            label += f" R:{risk:.0f}"
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw movement trail
        pts = list(trk.history)[-30:]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(frame, pts[i - 1], pts[i], c, 1)

    return frame


def draw_alerts(frame: np.ndarray, active_alerts: list) -> np.ndarray:
    y = 30
    for a in active_alerts[-5:]:
        color = SEVERITY_COLOR_BGR.get(a["severity"], (200, 200, 200))
        text = f"Track {a['track_id']}: {a['type']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (8, y - th - 4), (18 + tw, y + 4), (20, 20, 20), -1)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 28
    return frame


def draw_stats(frame: np.ndarray, frame_id: int, fps: float,
               n_tracks: int, n_alerts: int) -> np.ndarray:
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (w - tw - 16, 4), (w - 4, th + 16), (20, 20, 20), -1)
    cv2.putText(frame, text, (w - tw - 10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    return frame


def draw_zone(frame: np.ndarray, zone: tuple, label: str = "WATCH ZONE") -> np.ndarray:
    if zone:
        x1, y1, x2, y2 = zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(frame, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return frame
