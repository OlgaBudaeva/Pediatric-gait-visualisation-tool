import os, shutil, threading, datetime
import streamlit as st
import onnxruntime as ort
from gait_core import process_video

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)
GPU_LOCK = threading.Lock()   # one video at a time

st.set_page_config(page_title="Gait Visualizer", page_icon="🚶", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
h1 {font-weight: 700; margin-bottom: 0;}
.sub {color: #9aa0a6; margin-bottom: 1.5rem;}
</style>""", unsafe_allow_html=True)

# Optional password: $env:GAIT_APP_PASSWORD="yourpassword"
pw = os.environ.get("GAIT_APP_PASSWORD")
if pw and st.session_state.get("ok") is not True:
    st.title("Gait Visualizer")
    entered = st.text_input("Password", type="password")
    if entered == pw:
        st.session_state["ok"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password")
    st.stop()

st.title("Gait Visualizer")
st.markdown('<div class="sub">Upload a video of a child walking to see the movement as a skeleton with trails.</div>',
            unsafe_allow_html=True)

if "CUDAExecutionProvider" not in ort.get_available_providers():
    st.warning("GPU not detected. Processing will be slow.")

c1, c2 = st.columns([1, 2])
with c1:
    patient_id = st.text_input("Patient ID (no names)", placeholder="e.g. P-014")
    show_face = st.checkbox("Draw face", value=True,
                            help="Turn off when the child walks away from the camera.")
video = c2.file_uploader("Video of the child walking", type=["mp4", "mov", "avi", "mkv"])

if st.button("Run analysis", type="primary", use_container_width=True, disabled=video is None):
    if not patient_id.strip():
        st.error("Please enter a Patient ID.")
        st.stop()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(RESULTS, f"{patient_id.strip()}_{stamp}")
    os.makedirs(session_dir, exist_ok=True)
    in_path = os.path.join(session_dir, "input" + os.path.splitext(video.name)[1])
    with open(in_path, "wb") as f:
        f.write(video.getbuffer())

    bar = st.progress(0.0, text="Waiting for the GPU...")
    try:
        with GPU_LOCK:
            def cb(p):
                label = "Finding keypoints" if p < 0.8 else "Rendering video"
                bar.progress(min(p, 1.0), text=f"{label}... {int(p * 100)}%")
            summary = process_video(in_path, session_dir, show_face=show_face, progress_cb=cb)
    except Exception as e:
        bar.empty()
        shutil.rmtree(session_dir, ignore_errors=True)
        st.error(f"Analysis failed: {e}")
        st.stop()
    bar.empty()
    st.session_state["result"] = {"dir": session_dir, "summary": summary, "patient": patient_id.strip()}

res = st.session_state.get("result")
if res:
    d, s = res["dir"], res["summary"]
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", f'{s["duration_s"]} s')
    m2.metric("Frames", s["frames"])
    m3.metric("Child detected", f'{s["detection_rate_pct"]}%')
    m4.metric("Processing time", f'{s["processing_time_s"]} s')
    if s["detection_rate_pct"] < 90:
        st.warning("The child was not detected in many frames. Check lighting and that the whole body is in view.")

    vid = os.path.join(d, "visualization.mp4")
    st.video(vid)
    st.caption("Left: the video with the skeleton and trails. Right: the same movement with the child kept in place, "
               "so the leg and arm motion is easy to see. Blue = left side, orange = right side.")
    with open(vid, "rb") as f:
        st.download_button("Download video", f, f'{res["patient"]}_gait.mp4', "video/mp4")
    with open(os.path.join(d, "keypoints.npz"), "rb") as f:
        st.download_button("Download keypoints (.npz)", f, f'{res["patient"]}_keypoints.npz')
        