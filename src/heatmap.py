"""
heatmap.py — Cumulative position heatmap generator
Smart Retail Theft Detection System
"""

import cv2
import numpy as np


class HeatmapGenerator:
    def __init__(self, frame_shape: tuple, decay: float = 0.998):
        h, w = frame_shape[:2]
        self._map = np.zeros((h, w), dtype=np.float32)
        self._decay = decay
        self._frame_ref = None

    def update(self, positions: list):
        """positions: list of (cx, cy) tuples."""
        self._map *= self._decay
        for cx, cy in positions:
            if 0 <= cy < self._map.shape[0] and 0 <= cx < self._map.shape[1]:
                cv2.circle(self._map, (cx, cy), 20, 1.0, -1)

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        if self._map.max() < 0.01:
            return frame
        norm = (self._map / self._map.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask = (norm > 10).astype(np.uint8)[:, :, np.newaxis]
        blended = cv2.addWeighted(frame, 1.0, colored * mask, alpha, 0)
        return blended

    def save(self, path: str):
        if self._map.max() > 0:
            norm = (self._map / self._map.max() * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            cv2.imwrite(path, colored)
            return path
        return None
