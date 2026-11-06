import os, json, time, warnings
import cv2
import imageio
import numpy as np
from rtmlib import Wholebody

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rtmw-x-l", "end2end.onnx")

# ---------------- settings ----------------
SCORE_THR = 0.3           # body/feet points below this are treated as missing
FACE_THR = 0.3
SMOOTH = 5                # moving-average window for keypoints (odd number)
TRAIL_LEFT = 20           # trail length in frames, left panel
TRAIL_RIGHT = 30          # trail length in frames, right panel
BODY_HEIGHT_FRAC = 0.55   # skeleton height in the right panel, as a fraction of panel height
SCALE_SMOOTH = 31         # window (frames) for the size normalisation; larger = steadier size
# ------------------------------------------

L, R, C = (255, 120, 0), (0, 140, 255), (0, 200, 0)          # BGR: blue, orange, green
L_BRIGHT, R_BRIGHT = (255, 190, 70), (60, 190, 255)
EDGES = [
    (5, 6, C), (11, 12, C), (5, 11, L), (6, 12, R),
    (5, 7, L), (7, 9, L), (6, 8, R), (8, 10, R),
    (11, 13, L), (13, 15, L), (12, 14, R), (14, 16, R),
    (15, 17, L), (15, 18, L), (15, 19, L),
    (16, 20, R), (16, 21, R), (16, 22, R),
]
TRAIL_JOINTS = {15: L_BRIGHT, 16: R_BRIGHT, 9: L_BRIGHT, 10: R_BRIGHT}  # ankles, wrists

FACE_COL = (230, 230, 230)
F0 = 23
FACE_PARTS = [(0, 16, False), (17, 21, False), (22, 26, False), (27, 30, False),
              (31, 35, False), (36, 41, True), (42, 47, True), (48, 59, True), (60, 67, True)]
CHIN = F0 + 8

_model = None

def get_model():
    global _model
    if _model is None:
        kwargs = dict(to_openpose=False, mode="performance",
                      backend="onnxruntime", device="cuda")
        if os.path.exists(MODEL_PATH):
            kwargs.update(pose=MODEL_PATH, pose_input_size=(288, 384))
        _model = Wholebody(**kwargs)
    return _model

def _sub(cb, lo, hi):
    """Map a 0-1 progress of one stage onto the lo-hi part of the overall bar."""
    return (lambda p: cb(lo + (hi - lo) * p)) if cb else None

# ================= 1. keypoint extraction =================
def extract_keypoints(video_path, progress_cb=None):
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video. Try recording with 'Most Compatible' camera settings.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    all_k, all_s = [], []
    prev_center = None
    w = h = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        kpts, scores = model(frame)           # (people, 133, 2), (people, 133)

        if len(kpts) == 0:
            all_k.append(np.full((133, 2), np.nan))
            all_s.append(np.full(133, np.nan))
        else:
            if prev_center is None:           # first detection: the largest person
                areas = [np.ptp(k[:, 0]) * np.ptp(k[:, 1]) for k in kpts]
                p = int(np.argmax(areas))
            else:                             # afterwards: stay with the same person
                p = int(np.argmin([np.linalg.norm(k[:17].mean(axis=0) - prev_center) for k in kpts]))
            prev_center = kpts[p][:17].mean(axis=0)
            all_k.append(kpts[p])
            all_s.append(scores[p])

        i += 1
        if progress_cb:
            progress_cb(min(i / total, 1.0))
    cap.release()

    if not all_k:
        raise ValueError("No frames could be read from this video.")
    return np.array(all_k, np.float32), np.array(all_s, np.float32), float(fps), w, h

# ================= 2. cleaning: gaps + smoothing =================
def _fill(x):
    x = x.copy()
    bad = np.isnan(x)
    if bad.all() or not bad.any():
        return x
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x

def _smooth(x, n):
    if n <= 1 or np.isnan(x).all():
        return x
    return np.convolve(np.pad(x, (n // 2, n // 2), mode="edge"), np.ones(n) / n, mode="valid")

def clean_keypoints(K, S):
    kp = K.astype(np.float64).copy()
    mask = np.zeros(S.shape, dtype=bool)
    mask[:, :23] = S[:, :23] < SCORE_THR
    mask[:, 23:91] = S[:, 23:91] < FACE_THR
    kp[mask] = np.nan
    for j in range(91):
        for c in range(2):
            kp[:, j, c] = _smooth(_fill(kp[:, j, c]), SMOOTH)
    return kp

def centred_keypoints(kp, W, H):
    """Hip-centred ('treadmill') coordinates with a constant skeleton size."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mid_hip = np.nanmean(kp[:, [11, 12]], axis=1)
        mid_sho = np.nanmean(kp[:, [5, 6]], axis=1)
        mid_ank = np.nanmean(kp[:, [15, 16]], axis=1)
    span = np.linalg.norm(mid_sho - mid_ank, axis=1)        # apparent body height, per frame
    if np.isfinite(span).any():
        span = _smooth(_fill(span), min(SCALE_SMOOTH, 2 * (len(span) // 2) + 1))
        scale = BODY_HEIGHT_FRAC * H / np.maximum(span, 1.0)
    else:
        scale = np.ones(len(kp))
    anchor = np.array([W / 2, H * 0.42])
    return (kp - mid_hip[:, None, :]) * scale[:, None, None] + anchor

# ================= 3. drawing =================
def _ok(p):
    return not np.isnan(p).any()

def _pt(p):
    return int(p[0]), int(p[1])

def _draw_face(img, k, thick):
    for a, b, closed in FACE_PARTS:
        idx = list(range(F0 + a, F0 + b + 1))
        pairs = list(zip(idx[:-1], idx[1:]))
        if closed:
            pairs.append((idx[-1], idx[0]))
        for p, q in pairs:
            if _ok(k[p]) and _ok(k[q]):
                cv2.line(img, _pt(k[p]), _pt(k[q]), FACE_COL, thick, cv2.LINE_AA)
    for j in range(F0, F0 + 68):
        if _ok(k[j]):
            cv2.circle(img, _pt(k[j]), max(1, thick), (255, 255, 255), -1, cv2.LINE_AA)
    for j in (3, 4):
        if _ok(k[j]):
            cv2.circle(img, _pt(k[j]), thick + 3, FACE_COL, -1, cv2.LINE_AA)

def _draw_skeleton(img, k, thick, rad, face_thick, show_face):
    for a, b, col in EDGES:
        if _ok(k[a]) and _ok(k[b]):
            cv2.line(img, _pt(k[a]), _pt(k[b]), col, thick, cv2.LINE_AA)
    for j in range(5, 23):
        if _ok(k[j]):
            cv2.circle(img, _pt(k[j]), rad, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, _pt(k[j]), rad, (40, 40, 40), 1, cv2.LINE_AA)
    if _ok(k[5]) and _ok(k[6]):
        mid = (k[5] + k[6]) / 2
        if show_face and _ok(k[CHIN]):
            top = k[CHIN]
        elif show_face and _ok(k[0]):
            top = k[0]
        else:                                  # no face: short neck stub
            top = mid - np.array([0, 8 * thick])
        cv2.line(img, _pt(mid), _pt(top), C, thick, cv2.LINE_AA)
    if show_face:
        _draw_face(img, k, face_thick)

def _draw_trails(img, pts, i, length, width, outline=False):
    for j, col in TRAIL_JOINTS.items():
        for s in range(max(0, i - length), i):
            a, b = pts[s, j], pts[s + 1, j]
            if _ok(a) and _ok(b):
                fade = 0.35 + 0.65 * (s - (i - length)) / length
                c = tuple(int(v * fade) for v in col)
                if outline:
                    cv2.line(img, _pt(a), _pt(b), (0, 0, 0), width + 4, cv2.LINE_AA)
                cv2.line(img, _pt(a), _pt(b), c, width, cv2.LINE_AA)
        if _ok(pts[i, j]):
            p = _pt(pts[i, j])
            cv2.circle(img, p, width + 3, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, width + 3, (255, 255, 255), 1, cv2.LINE_AA)

def render_video(video_path, kp, kp_c, fps, W, H, out_path, show_face, progress_cb=None):
    W2, H2 = W + W % 2, H + H % 2             # even size, needed for H.264
    f = W2 / 720.0                             # keeps line widths proportional to resolution
    T = len(kp)
    cap = cv2.VideoCapture(video_path)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=1,
                                pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
    try:
        for i in range(T):
            # left: dimmed real video, real positions
            left = np.full((H2, W2, 3), 30, np.uint8)
            okv, frame = cap.read()
            if okv:
                left = (cv2.resize(frame, (W2, H2)) * 0.45).astype(np.uint8)
            _draw_trails(left, kp, i, TRAIL_LEFT, max(2, round(3 * f)))
            _draw_skeleton(left, kp[i], max(2, round(5 * f)), max(3, round(7 * f)), 1, show_face)
            cv2.putText(left, "In frame", (int(20 * f), int(40 * f)), cv2.FONT_HERSHEY_SIMPLEX,
                        f, (255, 255, 255), max(1, round(2 * f)), cv2.LINE_AA)

            # right: constant-size, hip-centred skeleton with trails
            right = np.full((H2, W2, 3), 30, np.uint8)
            cv2.line(right, (W2 // 2, 0), (W2 // 2, H2), (60, 60, 60), 1)
            _draw_trails(right, kp_c, i, TRAIL_RIGHT, max(3, round(6 * f)), outline=True)
            _draw_skeleton(right, kp_c[i], max(3, round(7 * f)), max(4, round(9 * f)),
                           max(1, round(2 * f)), show_face)
            cv2.putText(right, "blue = left   orange = right", (int(20 * f), H2 - int(25 * f)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * f, (200, 200, 200),
                        max(1, round(2 * f)), cv2.LINE_AA)

            writer.append_data(cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_BGR2RGB))
            if progress_cb:
                progress_cb((i + 1) / T)
    finally:
        writer.close()
        cap.release()

# ================= 4. full pipeline =================
def process_video(video_path, out_dir, show_face=True, progress_cb=None):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    K, S, fps, w, h = extract_keypoints(video_path, _sub(progress_cb, 0.0, 0.8))
    np.savez_compressed(os.path.join(out_dir, "keypoints.npz"),
                        keypoints=K, scores=S, fps=fps, width=w, height=h)

    kp = clean_keypoints(K, S)
    kp_c = centred_keypoints(kp, w, h)
    render_video(video_path, kp, kp_c, fps, w, h,
                 os.path.join(out_dir, "visualization.mp4"), show_face,
                 _sub(progress_cb, 0.8, 1.0))

    detected = int(np.sum(~np.isnan(K[:, 0, 0])))
    summary = {"frames": len(K), "fps": round(fps, 2), "width": w, "height": h,
               "duration_s": round(len(K) / fps, 2),
               "detection_rate_pct": round(100 * detected / len(K), 1),
               "processing_time_s": round(time.time() - t0, 1),
               "model": "RTMW performance / cuda"}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
