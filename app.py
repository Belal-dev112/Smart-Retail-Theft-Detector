#!/usr/bin/env python3
"""
app.py — Streamlit UI entry point for Smart Retail Theft Detection System
VIT BYOP Submission

Run:
    streamlit run app.py
"""

import sys
import time
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Smart Retail Theft Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

:root {
  --bg: #0a0c10;
  --surface: #111520;
  --surface2: #161b28;
  --accent: #00d4ff;
  --accent2: #ff4757;
  --accent3: #ffd32a;
  --text: #e0e6f0;
  --subtext: #8892a4;
  --border: #1e2a3a;
  --critical: #ff4757;
  --high: #ff6b35;
  --medium: #ffd32a;
  --low: #2ed573;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace;
}

.stApp { background: var(--bg); }

.main-title {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 2.4rem;
  letter-spacing: -1px;
  background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #ff4757 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0;
}

.subtitle {
  color: var(--subtext);
  font-size: 0.8rem;
  letter-spacing: 4px;
  text-transform: uppercase;
  margin-top: -4px;
}

.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  text-align: center;
}

.metric-value {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: var(--accent);
}

.metric-label {
  font-size: 0.7rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--subtext);
}

.alert-card {
  border-radius: 6px;
  padding: 10px 14px;
  margin: 6px 0;
  border-left: 4px solid;
  font-size: 0.82rem;
}

.alert-CRITICAL { background: rgba(255,71,87,0.12); border-color: var(--critical); }
.alert-HIGH     { background: rgba(255,107,53,0.12); border-color: var(--high); }
.alert-MEDIUM   { background: rgba(255,211,42,0.10); border-color: var(--medium); }
.alert-LOW      { background: rgba(46,213,115,0.10); border-color: var(--low); }

.section-header {
  font-family: 'Syne', sans-serif;
  font-size: 0.75rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--subtext);
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
  margin: 16px 0 10px 0;
}

div[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
  color: white !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 700 !important;
  letter-spacing: 1px;
  padding: 10px 24px !important;
}

.stButton > button:hover {
  opacity: 0.85 !important;
  transform: translateY(-1px);
}

.stSlider > div { color: var(--text) !important; }

[data-testid="metric-container"] {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}

.stProgress > div > div { background: var(--accent) !important; }

code { color: var(--accent) !important; background: var(--surface2) !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">Smart Retail Theft Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">VIT BYOP · YOLOv8 + DeepSORT + MediaPipe</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Configuration</div>', unsafe_allow_html=True)

    source_type = st.radio("Input Source", ["Upload Video", "Webcam"], horizontal=True)

    st.markdown('<div class="section-header">🔍 Detection</div>', unsafe_allow_html=True)
    confidence  = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    model_size  = st.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    device      = st.selectbox("Device", ["cpu", "cuda", "mps"])

    st.markdown('<div class="section-header">🎛 Features</div>', unsafe_allow_html=True)
    use_heatmap = st.toggle("Heatmap Overlay", value=True)
    use_pose    = st.toggle("Pose Estimation", value=False)

    st.markdown('<div class="section-header">⚠ Thresholds</div>', unsafe_allow_html=True)
    loiter_time = st.slider("Loiter Time (s)", 2, 30, 8)
    risk_alert  = st.slider("Risk Score Alert", 20, 100, 50)

    st.markdown('<div class="section-header">💾 Output</div>', unsafe_allow_html=True)
    save_video   = st.toggle("Save Output Video", value=True)
    output_dir   = st.text_input("Output Directory", value="outputs")

# ── Main layout ───────────────────────────────────────────────────────────────
col_video, col_panel = st.columns([3, 1.2])

with col_video:
    st.markdown('<div class="section-header">📹 Video Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    progress_bar      = st.progress(0)
    status_text       = st.empty()

with col_panel:
    st.markdown('<div class="section-header">⚡ Alert Feed</div>', unsafe_allow_html=True)
    alert_ph = st.empty()

# ── File uploader / webcam ────────────────────────────────────────────────────
video_path = None
if source_type == "Upload Video":
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        video_path = tmp.name
else:
    cam_idx  = st.number_input("Webcam Index", 0, 10, 0, step=1)
    video_path = str(int(cam_idx))

# ── Run button ────────────────────────────────────────────────────────────────
run_col, stop_col = st.columns([1, 1])
run_btn  = run_col.button("▶  Start Detection", use_container_width=True)
stop_btn = stop_col.button("⏹  Stop", use_container_width=True)

if "running" not in st.session_state:
    st.session_state.running = False
if stop_btn:
    st.session_state.running = False

def severity_html(alert):
    sev = alert.get("severity", "LOW")
    color_map = {"CRITICAL": "#ff4757", "HIGH": "#ff6b35", "MEDIUM": "#ffd32a", "LOW": "#2ed573"}
    c = color_map.get(sev, "#8892a4")
    return (
        f'<div class="alert-card alert-{sev}">'
        f'<span style="color:{c};font-weight:700">Track {alert["track_id"]}</span> '
        f'· {alert["type"]}'
        f'</div>'
    )

if run_btn and video_path:
    st.session_state.running = True

    import importlib, sys as _sys
    # Allow dynamic import
    if "src.pipeline" not in _sys.modules:
        pass

    from src.pipeline import RetailTheftPipeline

    config = {
        "model":      model_size,
        "confidence": confidence,
        "device":     device,
        "heatmap":    use_heatmap,
        "pose":       use_pose,
        "output_dir": output_dir,
        "headless":   True,
    }

    try:
        src_int = int(video_path)
    except ValueError:
        src_int = video_path

    pipeline = RetailTheftPipeline(config)

    if isinstance(src_int, int) and _sys.platform == "win32":
        cap = cv2.VideoCapture(src_int, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(src_int)

    if not cap.isOpened():
        st.error(f"❌ Cannot open source: {video_path}")
        st.stop()

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline.fps = fps_v
    pipeline.analyzer.fps = fps_v

    out_writer = None
    if save_video:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path  = str(Path(output_dir) / "ui_output.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_v, (w, h))

    recent_alerts = []
    alert_counts  = {}
    frame_count   = 0

    # Show placeholder while loading
    video_placeholder.info("⏳ Loading pipeline... (first run downloads YOLOv8 weights)")

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, new_alerts = pipeline.process_frame(frame)
        frame_count += 1

        if out_writer:
            out_writer.write(annotated)

        for a in new_alerts:
            recent_alerts.insert(0, a)
            alert_counts[a["type"]] = alert_counts.get(a["type"], 0) + 1

        recent_alerts = recent_alerts[:20]

        # Update display every 3 frames
        if frame_count % 3 == 0:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            if total > 0:
                progress_bar.progress(min(frame_count / total, 1.0))

            status_text.text(f"Processing frame {frame_count}" +
                              (f" / {total}" if total > 0 else ""))

            html_blocks = "".join(severity_html(a) for a in recent_alerts[:6])
            alert_ph.markdown(html_blocks, unsafe_allow_html=True)

    cap.release()
    if out_writer:
        out_writer.release()

    summary = pipeline.finalize()
    st.session_state.running = False
    status_text.success(f"✅ Processing complete — {summary['total_alerts']} alerts detected")

    # Final summary
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Session Summary</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total Frames",  frame_count)
    sc2.metric("Total Alerts",  summary["total_alerts"])
    sc3.metric("Alert Types",   len(summary.get("by_type", {})))

    st.write("**Alerts by type:**")
    for t, c in summary.get("by_type", {}).items():
        st.write(f"  `{t}` — **{c}**")

    if save_video and Path(output_dir).joinpath("ui_output.mp4").exists():
        with open(str(Path(output_dir) / "ui_output.mp4"), "rb") as f:
            st.download_button("⬇ Download Annotated Video", f,
                                file_name="detection_output.mp4", mime="video/mp4")

    if summary.get("heatmap") and Path(summary["heatmap"]).exists():
        st.image(summary["heatmap"], caption="Movement Heatmap", use_container_width=True)

elif run_btn and not video_path:
    st.warning("⚠ Please upload a video or select webcam first.")
else:
    video_placeholder.markdown("""
    <div style="background:#111520;border:1px dashed #1e2a3a;border-radius:8px;
                padding:80px;text-align:center;color:#8892a4;">
        <div style="font-size:3rem;margin-bottom:12px">🎥</div>
        <div style="font-family:Syne,sans-serif;font-size:1.2rem;color:#00d4ff">
            Upload a video or connect webcam
        </div>
        <div style="font-size:0.8rem;margin-top:8px">
            then click ▶ Start Detection
        </div>
    </div>
    """, unsafe_allow_html=True)
