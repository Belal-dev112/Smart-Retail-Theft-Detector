# Datasets for Smart Retail Theft Detection

## Recommended Public Datasets

### 1. UCF-Crime Dataset
- **URL:** https://www.crcv.ucf.edu/projects/real-world/
- **Size:** 1900 long surveillance videos, 128 hours
- **Classes:** Shoplifting, Robbery, Burglary, + 10 other anomalies
- **Format:** MP4, 30 FPS, various resolutions
- **Use:** Primary benchmark for anomaly detection

### 2. DCSASS (Dangerous and Criminal Scene Anomaly Surveillance)
- **URL:** https://github.com/GaetanLepage/DCSASS
- **Size:** 533 videos
- **Use:** Bounding box level annotations for shoplifting

### 3. VIRAT Video Dataset
- **URL:** https://viratdata.org/
- **Size:** 8.5 hours, outdoor/indoor scenes
- **Use:** Movement pattern analysis, loitering

### 4. Avenue Dataset (CUHK)
- **URL:** http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
- **Use:** Anomalous behaviour in public spaces

### 5. ShanghaiTech Campus Dataset
- **Size:** 437 videos, 130 anomalous events
- **Use:** Anomaly detection benchmark

---

## Custom Data Collection Guide

### Step 1 — Recording Setup
```
Camera placement:
  - Height: 2.5–3.5m from floor
  - Angle: 15–30° downward tilt
  - Coverage: Full aisle or checkout zone
  - Resolution: Minimum 720p, prefer 1080p
  - FPS: 25–30 FPS
```

### Step 2 — Annotation Tools
- **CVAT** (https://cvat.ai) — free, web-based, supports video
- **LabelImg** — simple bounding box tool
- **VGG Image Annotator (VIA)** — browser-based

### Step 3 — Annotation Schema
```json
{
  "video": "shop_cam1_2024.mp4",
  "annotations": [
    {
      "frame_start": 120,
      "frame_end": 350,
      "track_id": 1,
      "type": "LOITERING",
      "bbox_sample": [245, 180, 310, 420]
    }
  ]
}
```

### Step 4 — Ground Truth JSON (for metrics evaluation)
```json
[
  {"frame": 120, "track_id": 1, "type": "LOITERING"},
  {"frame": 200, "track_id": 2, "type": "CROUCHING"},
  {"frame": 315, "track_id": 1, "type": "CONCEALMENT"}
]
```

### Step 5 — Data Augmentation
```python
# Using Albumentations
import albumentations as A
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
])
```

---

## Minimum Dataset Recommendation

For a reliable evaluation:
- ≥ 20 video clips per behaviour class
- At least 5 minutes of normal footage
- Mixed lighting conditions (day/night)
- Multiple camera angles
