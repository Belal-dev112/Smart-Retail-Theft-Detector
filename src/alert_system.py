"""
alert_system.py — Alert logging, deduplication, and console/file output
Smart Retail Theft Detection System
"""

import csv
import json
import time
import os
from collections import defaultdict
from pathlib import Path

SEVERITY_COLOR = {
    "CRITICAL": "\033[91m",   # red
    "HIGH":     "\033[93m",   # yellow
    "MEDIUM":   "\033[94m",   # blue
    "LOW":      "\033[96m",   # cyan
    "RESET":    "\033[0m",
}

SEVERITY_BGR = {
    "CRITICAL": (0, 0, 220),
    "HIGH":     (0, 165, 255),
    "MEDIUM":   (255, 165, 0),
    "LOW":      (128, 255, 0),
}


class AlertSystem:
    def __init__(self, output_dir: str = "outputs", cooldown_frames: int = 90):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_frames = cooldown_frames
        self._last_alert = defaultdict(lambda: -9999)
        self.all_alerts = []
        self._log_file = self.output_dir / f"alerts_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_csv()

    def _init_csv(self):
        with open(self._log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame", "track_id", "type", "severity", "message"])

    def process(self, alerts: list, frame_id: int, verbose: bool = True) -> list:
        """Filter by cooldown, log, print. Returns deduplicated list."""
        active = []
        for a in alerts:
            key = (a["track_id"], a["type"])
            if frame_id - self._last_alert[key] >= self.cooldown_frames:
                self._last_alert[key] = frame_id
                self.all_alerts.append(a)
                self._append_csv(a)
                if verbose:
                    self._print_alert(a)
                active.append(a)
        return active

    def _append_csv(self, a):
        with open(self._log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([a["timestamp"], a["frame"], a["track_id"],
                              a["type"], a["severity"], a["message"]])

    def _print_alert(self, a):
        c = SEVERITY_COLOR.get(a["severity"], "")
        r = SEVERITY_COLOR["RESET"]
        print(f"{c}[ALERT]{r} Frame {a['frame']:>6} | Track {a['track_id']:>3} | "
              f"{c}{a['severity']:8}{r} | {a['type']:20} | {a['message']}")

    def get_active_alerts(self, current_frame: int, display_frames: int = 45) -> list:
        """Returns alerts that were triggered within the last `display_frames`."""
        return [a for a in self.all_alerts if current_frame - a["frame"] < display_frames]

    def summary(self) -> dict:
        counts = defaultdict(int)
        for a in self.all_alerts:
            counts[a["type"]] += 1
        return {
            "total_alerts": len(self.all_alerts),
            "by_type": dict(counts),
            "log_file": str(self._log_file),
        }

    def save_json(self, path: str = None):
        path = path or str(self.output_dir / "alerts_summary.json")
        with open(path, "w") as f:
            json.dump({"alerts": self.all_alerts, "summary": self.summary()}, f, indent=2)
        return path
