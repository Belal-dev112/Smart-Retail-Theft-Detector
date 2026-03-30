#  Smart Retail Theft Detection System
Project Overview

This system detects suspicious retail behaviour in CCTV footage using deep learning and computer vision. It operates in **real-time** and supports both **CLI** (mandatory for evaluation) and a **Streamlit UI** (for demonstration).

---

##  Features

| Feature | Description |
|---|---|
|  Person Detection | YOLOv8n/s/m with configurable confidence |
|  Multi-Object Tracking | IoU-based tracker (DeepSORT-compatible API) |
|  Pose Estimation | MediaPipe Pose — skeleton + joint angles |
|  Loitering Detection | Time + spatial displacement analysis |
|  Sudden Movement | Velocity spike detection |
|  Crouching Detection | Bounding-box ratio + knee angle |
|  Concealment Detection | Wrist-torso proximity via pose |
|  Heatmap | Cumulative movement overlay |
|  Metrics | Precision, Recall, F1, per class |
|  Logging | CSV + JSON alert logs |

---

##  Tech Stack

- **Python 3.9+**
- **OpenCV** — video I/O and drawing
- **YOLOv8** (Ultralytics) — person detection
- **MediaPipe** — pose estimation
- **NumPy / SciPy** — numerical analysis
- **Streamlit** — web UI
- **argparse** — CLI interface

---

##  Folder Structure

```
smart_retail_theft_detection/
├── main.py              ← CLI entry point (MANDATORY for evaluation)
├── app.py               ← Streamlit UI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── pipeline.py      ← Shared backend (used by both CLI & UI)
│   ├── detector.py      ← YOLOv8 / HOG person detector
│   ├── tracker.py       ← Multi-object tracker
│   ├── pose_estimator.py← MediaPipe pose wrapper
│   ├── behavior_analyzer.py ← Suspicious activity rules
│   ├── alert_system.py  ← Alert deduplication & logging
│   ├── heatmap.py       ← Cumulative heatmap
│   ├── visualizer.py    ← OpenCV drawing helpers
│   └── metrics.py       ← Evaluation metrics
├── ui/
│   └── __init__.py
├── models/              ← Place YOLOv8 .pt weights here
├── data/
│   └── gt_sample.json   ← Sample ground truth for evaluation
└── outputs/             ← Generated alerts CSV, JSON, heatmap
```

---

##  Installation

```bash
# Clone repository
git clone https://github.com/yourusername/smart-retail-theft-detection
cd smart-retail-theft-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

##  CLI Execution (MANDATORY)

### Basic detection on a video file
```bash
python main.py --input video.mp4 --output result.mp4
```

### Webcam (live detection)
```bash
python main.py --input 0 --show
```

### With heatmap overlay
```bash
python main.py --input video.mp4 --output result.mp4 --heatmap
```

### With pose estimation
```bash
python main.py --input video.mp4 --output result.mp4 --heatmap --pose
```

### Custom watch zone (x1 y1 x2 y2)
```bash
python main.py --input video.mp4 --zone 100 100 500 400
```

### Use GPU (if available)
```bash
python main.py --input video.mp4 --device cuda
```

### Change model size
```bash
python main.py --input video.mp4 --model yolov8s.pt --confidence 0.5
```

### Run evaluation metrics
```bash
python main.py --eval \
  --predictions outputs/alerts_summary.json \
  --ground-truth data/gt_sample.json \
  --metrics-output outputs/metrics.json
```

### Full CLI options
```bash
python main.py --help
```

```
usage: main.py [-h] [--input INPUT] [--output OUTPUT] [--output-dir OUTPUT_DIR]
               [--model MODEL] [--confidence CONFIDENCE] [--device {cpu,cuda,mps}]
               [--heatmap] [--pose] [--zone X1 Y1 X2 Y2]
               [--show] [--headless]
               [--eval] [--predictions PREDICTIONS] [--ground-truth GROUND_TRUTH]
               [--metrics-output METRICS_OUTPUT]
```

---

##  UI Execution (Streamlit)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### UI Features
- Upload a video file or use a webcam
- Real-time annotated video stream
- Live stats: FPS, tracks, alert count
- Alert feed with severity colours
- Progress bar for uploaded videos
- Download annotated output video
- Movement heatmap display

---

##  Detection Classes

| Class | Severity | Trigger |
|---|---|---|
| `LOITERING` | HIGH | Dwell >8s in <80px radius |
| `SUDDEN_MOVEMENT` | MEDIUM | Velocity spike >3× average |
| `CROUCHING` | HIGH | Bounding box h/w < 0.55 |
| `CONCEALMENT` | CRITICAL | Wrists near torso for 12+ frames |

---

##  Evaluation Metrics

The system computes **per-class** and **macro-average** metrics:

```bash
python main.py --eval \
  --predictions outputs/alerts_summary.json \
  --ground-truth data/gt_sample.json
```

Output:
```
============================================================
CLASS                   PREC      REC       F1
------------------------------------------------------------
LOITERING             0.8500   0.7800   0.8135
CROUCHING             0.9100   0.8300   0.8682
CONCEALMENT           0.7800   0.7200   0.7489
SUDDEN_MOVEMENT       0.8200   0.7600   0.7889
MACRO_AVG             0.8400   0.7725   0.8049
============================================================
```

---

##  Suggested Datasets

| Dataset | Link | Use |
|---|---|---|
| UCF-Crime | [Link](https://www.crcv.ucf.edu/projects/real-world/) | Shoplifting, violence |
| DCSASS | [Link](https://github.com/GaetanLepage/DCSASS) | Retail surveillance |
| VIRAT | [Link](https://viratdata.org/) | Outdoor surveillance |
| ShanghaiTech Campus | Public | Anomaly detection |
| Custom collection | CCTV + labelling tool | Best for your store |

---

##  Bonus Features

- **Heatmap** — `--heatmap` flag overlays cumulative position density
- **Edge Deployment** — Export YOLOv8 to ONNX/TFLite:  
  ```bash
  yolo export model=yolov8n.pt format=onnx  # for Raspberry Pi / Jetson
  ```
- **Watch Zone** — `--zone x1 y1 x2 y2` monitors specific shelf area

---

##  License

MIT License — see `LICENSE`[Link](https://github.com/Belal-dev112/Smart-Retail-Theft-Detector/blob/835c39e7ab25e4677d06698c1b41a051cc90eb47/LICENSE) for details.
