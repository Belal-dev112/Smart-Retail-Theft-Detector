"""
tracker.py — DeepSORT-style multi-object tracker (pure NumPy fallback included)
Smart Retail Theft Detection System
"""

import numpy as np
import cv2
from collections import defaultdict


class SimpleTrack:
    """Single tracked object."""
    _id_counter = 0

    def __init__(self, bbox, frame_id):
        SimpleTrack._id_counter += 1
        self.track_id = SimpleTrack._id_counter
        self.bbox = bbox
        self.history = [self._center(bbox)]
        self.last_seen = frame_id
        self.age = 0
        self.hits = 1
        self.miss_streak = 0

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bbox, frame_id):
        self.bbox = bbox
        self.history.append(self._center(bbox))
        if len(self.history) > 150:
            self.history.pop(0)
        self.last_seen = frame_id
        self.hits += 1
        self.miss_streak = 0

    def predict(self):
        """Simple linear prediction."""
        return self.bbox

    @property
    def center(self):
        return self._center(self.bbox)


class MultiObjectTracker:
    """
    IoU-based tracker that mimics DeepSORT API.
    Replace with deep_sort_realtime for production use.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list = []
        self.frame_id = 0

    def update(self, detections: list) -> list:
        """
        Args:
            detections: list of {'bbox': [x1,y1,x2,y2], 'confidence': float}
        Returns:
            list of active Track objects (hits >= min_hits)
        """
        self.frame_id += 1
        det_bboxes = [d["bbox"] for d in detections]

        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate(det_bboxes)

        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(det_bboxes[det_idx], self.frame_id)

        for det_idx in unmatched_dets:
            self.tracks.append(SimpleTrack(det_bboxes[det_idx], self.frame_id))

        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].miss_streak += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.miss_streak <= self.max_age]

        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _associate(self, det_bboxes):
        if not self.tracks or not det_bboxes:
            return [], list(range(len(det_bboxes))), list(range(len(self.tracks)))

        iou_matrix = np.zeros((len(self.tracks), len(det_bboxes)))
        for t_idx, trk in enumerate(self.tracks):
            for d_idx, det in enumerate(det_bboxes):
                iou_matrix[t_idx, d_idx] = self._iou(trk.bbox, det)

        matched, unmatched_dets, unmatched_trks = [], [], []
        used_trks, used_dets = set(), set()

        # Greedy matching
        flat = np.argsort(-iou_matrix.ravel())
        for idx in flat:
            t_idx, d_idx = divmod(idx, len(det_bboxes))
            if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                break
            if t_idx not in used_trks and d_idx not in used_dets:
                matched.append((t_idx, d_idx))
                used_trks.add(t_idx)
                used_dets.add(d_idx)

        unmatched_dets = [i for i in range(len(det_bboxes)) if i not in used_dets]
        unmatched_trks = [i for i in range(len(self.tracks)) if i not in used_trks]
        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
