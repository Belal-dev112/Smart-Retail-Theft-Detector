#!/usr/bin/env python3
"""
main.py — CLI entry point for Smart Retail Theft Detection System
VIT BYOP Submission

Usage:
    python main.py --input video.mp4 --output result.mp4
    python main.py --input 0 --show
    python main.py --input video.mp4 --output result.mp4 --heatmap --pose --zone 100 100 500 400
    python main.py --eval --predictions outputs/alerts.json --ground-truth data/gt.json
"""

import argparse
import sys
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart Retail Theft Detection System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Basic detection on a video file:
    python main.py --input video.mp4 --output result.mp4

  Webcam (index 0):
    python main.py --input 0 --show

  With heatmap + pose estimation:
    python main.py --input video.mp4 --output out.mp4 --heatmap --pose

  Custom watch zone (x1 y1 x2 y2):
    python main.py --input video.mp4 --zone 100 100 500 400

  Evaluate against ground truth:
    python main.py --eval --predictions outputs/alerts.json --ground-truth data/gt.json

  Use GPU:
    python main.py --input video.mp4 --device cuda
        """
    )

    # I/O
    p.add_argument("--input",  "-i", type=str, default="0",
                   help="Input video path or webcam index (default: 0)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output video path (optional)")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Directory for alerts CSV, heatmap, JSON (default: outputs)")

    # Model
    p.add_argument("--model",      type=str,  default="yolov8n.pt",
                   help="YOLOv8 weights path (default: yolov8n.pt)")
    p.add_argument("--confidence", type=float, default=0.4,
                   help="Detection confidence threshold (default: 0.4)")
    p.add_argument("--device",     type=str,  default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Inference device (default: cpu)")

    # Features
    p.add_argument("--heatmap", action="store_true", help="Overlay cumulative heatmap")
    p.add_argument("--pose",    action="store_true", help="Enable MediaPipe pose estimation")
    p.add_argument("--zone",    nargs=4, type=int, metavar=("X1","Y1","X2","Y2"),
                   default=None, help="Watch zone rectangle")

    # Display
    p.add_argument("--show",     action="store_true",
                   help="Show live window (requires display)")
    p.add_argument("--headless", action="store_true",
                   help="Suppress all window output (default behaviour on servers)")

    # Evaluation
    p.add_argument("--eval",           action="store_true",
                   help="Run evaluation metrics instead of detection")
    p.add_argument("--predictions",    type=str, default=None,
                   help="Path to predictions JSON (for --eval)")
    p.add_argument("--ground-truth",   type=str, default=None,
                   help="Path to ground truth JSON (for --eval)")
    p.add_argument("--metrics-output", type=str, default="outputs/metrics.json",
                   help="Where to save metrics report (default: outputs/metrics.json)")

    return p


def run_detection(args):
    from src.pipeline import RetailTheftPipeline

    # Parse input source
    try:
        source = int(args.input)
    except ValueError:
        source = args.input
        if not Path(source).exists():
            print(f"[ERROR] Input file not found: {source}")
            sys.exit(1)

    config = {
        "model":      args.model,
        "confidence": args.confidence,
        "device":     args.device,
        "heatmap":    args.heatmap,
        "pose":       args.pose,
        "zone":       tuple(args.zone) if args.zone else None,
        "output_dir": args.output_dir,
        "headless":   not args.show,
        "fps":        30.0,
    }

    print("=" * 60)
    print("  Smart Retail Theft Detection System")
    print("  VIT BYOP — CLI Mode")
    print("=" * 60)
    print(f"  Input      : {args.input}")
    print(f"  Output     : {args.output or 'not saved'}")
    print(f"  Model      : {args.model}")
    print(f"  Confidence : {args.confidence}")
    print(f"  Device     : {args.device}")
    print(f"  Heatmap    : {args.heatmap}")
    print(f"  Pose       : {args.pose}")
    print(f"  Watch zone : {args.zone}")
    print("=" * 60 + "\n")

    pipeline = RetailTheftPipeline(config)
    summary  = pipeline.run_on_video(
        source=source,
        output_path=args.output,
        show_window=args.show,
    )

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total alerts : {summary['total_alerts']}")
    for t, c in summary.get("by_type", {}).items():
        print(f"    {t:<22} {c:>4}")
    print(f"  Alert log    : {summary['log_file']}")
    if summary.get("heatmap"):
        print(f"  Heatmap      : {summary['heatmap']}")
    print("=" * 60)
    return 0


def run_evaluation(args):
    import json
    from src.metrics import MetricsEvaluator

    if not args.predictions or not args.ground_truth:
        print("[ERROR] --eval requires --predictions and --ground-truth")
        sys.exit(1)

    with open(args.predictions) as f:
        data = json.load(f)
        predictions = data.get("alerts", data)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    evaluator = MetricsEvaluator()
    results   = evaluator.evaluate(predictions, ground_truth)
    evaluator.print_report(results)

    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(results, args.metrics_output)
    return 0


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.eval:
        sys.exit(run_evaluation(args))
    else:
        sys.exit(run_detection(args))


if __name__ == "__main__":
    main()
