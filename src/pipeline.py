"""
pipeline.py — Shared processing pipeline used by BOTH CLI (main.py) and UI (app.py)
Smart Retail Theft Detection System
"""

import cv2
import time
import numpy as np
from pathlib import Path

from src.detector import PersonDetector
from src.tracker import MultiObjectTracker
from src.pose_estimator import PoseEstimator
from src.behavior_analyzer import BehaviorAnalyzer
from src.alert_system import AlertSystem
from src.heatmap import HeatmapGenerator
from src.visualizer import draw_tracks, draw_alerts, draw_stats, draw_zone


class RetailTheftPipeline:
    """
    End-to-end pipeline that accepts frames and produces annotated frames + alerts.
    Designed to be shared by CLI and UI.
    """

    def __init__(self, config: dict):
        self.config = config
        self.fps = config.get("fps", 30.0)
        self.show_heatmap = config.get("heatmap", False)
        self.show_pose = config.get("pose", False)
        self.zone = config.get("zone", None)          # (x1,y1,x2,y2) watch zone
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = config.get("headless", True)

        self.detector  = PersonDetector(
            model_path=config.get("model", "yolov8n.pt"),
            confidence=config.get("confidence", 0.4),
            device=config.get("device", "cpu"),
        )
        self.tracker   = MultiObjectTracker()
        self.pose      = PoseEstimator() if self.show_pose else None
        self.analyzer  = BehaviorAnalyzer(fps=self.fps)
        self.alerts    = AlertSystem(output_dir=str(self.output_dir))
        self.heatmap   = None   # lazily initialized on first frame

        self.frame_id  = 0
        self._fps_times = []
        self.active_alerts = []

    # ── Main entry ──────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Returns: (annotated_frame, new_alerts_this_frame)
        """
        self.frame_id += 1
        t0 = time.time()

        if self.heatmap is None:
            self.heatmap = HeatmapGenerator(frame.shape)

        # 1. Detect
        detections = self.detector.detect(frame)

        # 2. Track
        tracks = self.tracker.update(detections)

        # 3. Pose (optional, per track bbox crop)
        pose_results = {}
        if self.show_pose and self.pose:
            for trk in tracks:
                x1, y1, x2, y2 = [max(0, v) for v in trk.bbox]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pose_results[trk.track_id] = self.pose.estimate(crop)

        # 4. Behavior analysis
        raw_alerts = []
        risk_scores = {}
        positions = []
        for trk in tracks:
            pose_data = pose_results.get(trk.track_id, {})
            new_alerts = self.analyzer.update(trk.track_id, trk.bbox, self.frame_id, pose_data)
            raw_alerts.extend(new_alerts)
            risk_scores[trk.track_id] = self.analyzer.track_risk_score(trk.track_id)
            positions.append(trk.center)

        # 5. Alert system (cooldown + logging)
        new_alerts = self.alerts.process(raw_alerts, self.frame_id,
                                          verbose=not self.headless)
        self.active_alerts = self.analyzer.get_active_alerts(self.frame_id)

        # 6. Heatmap
        self.heatmap.update(positions)

        # 7. Annotate frame
        annotated = frame.copy()
        if self.show_heatmap:
            annotated = self.heatmap.overlay(annotated)
        if self.zone:
            annotated = draw_zone(annotated, self.zone)

        annotated = draw_tracks(annotated, tracks, risk_scores)
        annotated = draw_alerts(annotated, self.active_alerts)

        # FPS calculation
        self._fps_times.append(time.time() - t0)
        if len(self._fps_times) > 30:
            self._fps_times.pop(0)
        current_fps = 1.0 / (sum(self._fps_times) / len(self._fps_times) + 1e-6)

        annotated = draw_stats(annotated, self.frame_id, current_fps,
                                len(tracks), len(self.alerts.all_alerts))

        # Draw pose skeleton
        if self.show_pose and self.pose:
            for trk in tracks:
                pd = pose_results.get(trk.track_id, {})
                if pd:
                    annotated = self.pose.draw(annotated, pd)

        return annotated, new_alerts

    def finalize(self) -> dict:
        """Save heatmap, alerts JSON, and return summary."""
        heatmap_path = self.heatmap.save(str(self.output_dir / "heatmap.jpg")) if self.heatmap else None
        json_path = self.alerts.save_json()
        summary = self.alerts.summary()
        summary["heatmap"] = heatmap_path
        summary["json"] = json_path
        return summary

    def run_on_video(self, source, output_path: str = None, show_window: bool = False) -> dict:
        """
        Full video processing loop.
        source: path string or int (webcam index)
        """
        import sys
        if isinstance(source, int) and sys.platform == "win32":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        print(f"\n[Pipeline] Source : {source}")
        print(f"[Pipeline] Res    : {width}x{height}  FPS={self.fps:.1f}  Frames={total_frames}")
        print(f"[Pipeline] Output : {output_path or 'none'}\n")

        start = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, _ = self.process_frame(frame)

                if writer:
                    writer.write(annotated)

                if show_window:
                    cv2.imshow("Smart Retail Theft Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n[Pipeline] Interrupted by user.")
                        break

                if self.frame_id % 100 == 0:
                    elapsed = time.time() - start
                    progress = f"{self.frame_id}/{total_frames}" if total_frames > 0 else str(self.frame_id)
                    print(f"[Pipeline] Progress: {progress}  Elapsed: {elapsed:.1f}s  "
                          f"Alerts: {len(self.alerts.all_alerts)}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()

        elapsed = time.time() - start
        summary = self.finalize()
        print(f"\n[Pipeline] Done — {self.frame_id} frames in {elapsed:.1f}s")
        print(f"[Pipeline] Total alerts : {summary['total_alerts']}")
        print(f"[Pipeline] Alert log    : {summary['log_file']}")
        if summary.get("heatmap"):
            print(f"[Pipeline] Heatmap      : {summary['heatmap']}")
        return summary
