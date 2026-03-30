"""
detector.py — YOLOv8 object detection wrapper
Smart Retail Theft Detection System
"""

import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class PersonDetector:
    """
    YOLOv8-based person detector.
    Falls back to HOG detector if ultralytics is unavailable.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4, device: str = "cpu"):
        self.confidence = confidence
        self.device = device
        self.model = None
        self.use_yolo = False

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.use_yolo = True
                print(f"[Detector] YOLOv8 loaded: {model_path}")
            except Exception as e:
                print(f"[Detector] YOLOv8 load failed ({e}), falling back to HOG.")
        else:
            print("[Detector] ultralytics not installed, using HOG person detector.")

        if not self.use_yolo:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list:
        if self.use_yolo:
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list:
        results = self.model(frame, conf=self.confidence, classes=[0], verbose=False, device=self.device)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "class": "person"})
        return detections

    def _detect_hog(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, weights = self._hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
        detections = []
        for (x, y, w, h), conf in zip(rects, weights):
            if conf > 0.3:
                detections.append({"bbox": [x, y, x + w, y + h], "confidence": float(conf[0]), "class": "person"})
        return detections
