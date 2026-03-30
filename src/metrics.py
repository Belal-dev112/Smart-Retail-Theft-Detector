"""
metrics.py — Evaluation metrics computation (Precision, Recall, F1, etc.)
Smart Retail Theft Detection System
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class MetricsEvaluator:
    """
    Computes per-class and overall P/R/F1 from ground truth vs predicted alerts.

    Ground truth JSON format:
        [{"frame": int, "track_id": int, "type": str}, ...]
    """

    def __init__(self, tolerance_frames: int = 30):
        self.tolerance = tolerance_frames

    def evaluate(self, predictions: list, ground_truth: list) -> dict:
        classes = set(a["type"] for a in ground_truth) | set(a["type"] for a in predictions)
        results = {}

        for cls in classes:
            pred = [a for a in predictions if a["type"] == cls]
            gt   = [a for a in ground_truth if a["type"] == cls]
            tp, fp, fn = self._match(pred, gt)
            results[cls] = self._scores(tp, fp, fn)

        # Macro average
        if results:
            results["MACRO_AVG"] = {
                "precision": np.mean([v["precision"] for v in results.values()]),
                "recall":    np.mean([v["recall"]    for v in results.values()]),
                "f1":        np.mean([v["f1"]        for v in results.values()]),
            }
        return results

    def _match(self, pred, gt):
        matched_gt = set()
        tp = 0
        for p in pred:
            for i, g in enumerate(gt):
                if i not in matched_gt and abs(p["frame"] - g["frame"]) <= self.tolerance:
                    tp += 1
                    matched_gt.add(i)
                    break
        fp = len(pred) - tp
        fn = len(gt) - tp
        return tp, fp, fn

    @staticmethod
    def _scores(tp, fp, fn):
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        return {"precision": round(prec, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    @staticmethod
    def print_report(results: dict):
        print("\n" + "="*60)
        print(f"{'CLASS':<22} {'PREC':>8} {'REC':>8} {'F1':>8}")
        print("-"*60)
        for cls, m in results.items():
            print(f"{cls:<22} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
        print("="*60)

    @staticmethod
    def save_report(results: dict, path: str):
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Metrics] Report saved → {path}")
