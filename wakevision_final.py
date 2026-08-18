
import cv2
import numpy as np
from scipy.spatial import distance
import time
from collections import deque
import pygame
import csv
from datetime import datetime
import os
import math

# ── MediaPipe ────────────────────────────────────────────────────────────────
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision

# ── Audio ─────────────────────────────────────────────────────────────────────
os.environ["SDL_AUDIODRIVER"] = "directsound"
pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()
pygame.mixer.init()

_sound = {}
for name, path in [("mild", "sounds/early.wav"),
                   ("moderate", "sounds/strong.wav"),
                   ("severe", "sounds/emergency.wav"),
                   ("critical", "sounds/emergency.wav")]:
    try:
        _sound[name] = pygame.mixer.Sound(path)
    except Exception:
        pass

ch = {k: pygame.mixer.Channel(i) for i, k in
      enumerate(["mild", "moderate", "severe", "critical"])}

# ── MediaPipe model ───────────────────────────────────────────────────────────
MODEL_PATH = "models/face_landmarker.task"
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
)

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_CONTOUR  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]   # MediaPipe iris landmarks (if available)
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP   = 1
CHIN       = 152
LEFT_CORNER  = 61
RIGHT_CORNER = 291

# 3-D reference points (canonical face model in mm)
MODEL_POINTS_3D = np.array([
    [0.0,   0.0,   0.0],    # Nose tip
    [0.0,  -330.0, -65.0],  # Chin
    [-225.0, 170.0, -135.0],# Left eye left corner
    [225.0, 170.0, -135.0], # Right eye right corner
    [-150.0,-150.0,-125.0], # Left mouth corner
    [150.0, -150.0,-125.0], # Right mouth corner
], dtype=np.float64)

LM_IDX_FOR_POSE = [NOSE_TIP, CHIN, 33, 263, LEFT_CORNER, RIGHT_CORNER]

# ── Parameters ────────────────────────────────────────────────────────────────
EAR_ALPHA       = 0.18       # smoothing factor for EAR
PERCLOS_WINDOW  = 30.0       # seconds for PERCLOS calculation
MICROSLEEP_SEC  = 1.5        # seconds closed → micro-sleep
MIN_BLINK_FRAMES = 3
MIN_YAWN_FRAMES  = 18        # frames mouth must be open to count as yawn
FPS_ESTIMATE    = 20

CALIBRATION_SEC = 25         # calibration duration
CAL_EAR_PCT     = 72         # ear threshold = this percentile of calibration data

# Drowsiness level thresholds (attention score 0-100)
LEVEL_THRESHOLDS = {
    "ALERT":    80,
    "MILD":     62,
    "MODERATE": 45,
    "SEVERE":   28,
    "CRITICAL":  0,
}

LEVEL_COLORS = {
    "ALERT":    (0,  200,  60),
    "MILD":     (0,  200, 200),
    "MODERATE": (0,  150, 255),
    "SEVERE":   (0,   80, 255),
    "CRITICAL": (0,   0,  220),
}

SOUND_COOLDOWN = {"mild": 6.0, "moderate": 3.5, "severe": 1.5, "critical": 0.5}

# ── Helpers ───────────────────────────────────────────────────────────────────

def ear(eye_pts):
    A = distance.euclidean(eye_pts[1], eye_pts[5])
    B = distance.euclidean(eye_pts[2], eye_pts[4])
    C = distance.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)


def mar(lm):
    """Mouth aspect ratio."""
    return distance.euclidean(lm[13], lm[17]) / (
           distance.euclidean(lm[61], lm[291]) + 1e-6)


def head_pose_angles(lm, w, h):
    """Return (pitch_deg, yaw_deg, roll_deg) using solvePnP."""
    focal  = w
    centre = (w / 2.0, h / 2.0)
    cam_matrix = np.array([[focal,0,centre[0]],[0,focal,centre[1]],[0,0,1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))
    pts_2d = np.array([lm[i] for i in LM_IDX_FOR_POSE], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, pts_2d, cam_matrix,
                               dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    pitch = math.degrees(math.atan2(-rmat[2,0], sy))
    yaw   = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
    roll  = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
    return pitch, yaw, roll


def gaze_score(lm, w, h):
    """
    Estimate gaze deviation: ratio of pupil offset within iris bounding box.
    Returns 0.0 (straight ahead) to 1.0 (looking far to side/up/down).
    Falls back to 0.0 if iris landmarks unavailable.
    """
    try:
        def _eye_gaze(eye_contour, iris_pts):
            cx_iris = np.mean([lm[i][0] for i in iris_pts])
            cy_iris = np.mean([lm[i][1] for i in iris_pts])
            xs = [lm[i][0] for i in eye_contour]
            ys = [lm[i][1] for i in eye_contour]
            cx_box = (min(xs) + max(xs)) / 2
            cy_box = (min(ys) + max(ys)) / 2
            dx = abs(cx_iris - cx_box) / (max(xs) - min(xs) + 1)
            dy = abs(cy_iris - cy_box) / (max(ys) - min(ys) + 1)
            return min(1.0, math.sqrt(dx**2 + dy**2) * 2.5)
        l = _eye_gaze(LEFT_EYE_CONTOUR, LEFT_IRIS)
        r = _eye_gaze(RIGHT_EYE_CONTOUR, RIGHT_IRIS)
        return (l + r) / 2
    except Exception:
        return 0.0


def attention_score(perclos, blink_rate_pm, yawn_rate_pm,
                    pitch, yaw, micro_sleep, gaze_dev):
    """
    Compute attention score 0-100.
    Each factor penalises from 100.
    """
    score = 100.0
    # PERCLOS (0-100 → 0-45 penalty)
    score -= perclos * 0.45
    # Blink rate: normal 15-20/min; excess is drowsiness
    excess_blinks = max(0, blink_rate_pm - 22)
    score -= excess_blinks * 0.7
    # Yawn rate: each yawn/min costs 8 points
    score -= yawn_rate_pm * 8.0
    # Head nod: forward pitch > 15 deg
    if pitch < -15:
        score -= min(25, abs(pitch + 15) * 0.8)
    # Head turn: yaw > 20 deg
    if abs(yaw) > 20:
        score -= min(15, (abs(yaw) - 20) * 0.5)
    # Gaze deviation (0-1) → 0-15 penalty
    score -= gaze_dev * 15.0
    # Micro-sleep instant penalty
    if micro_sleep:
        score -= 40
    return max(0.0, min(100.0, score))


def drowsiness_level(score):
    for level in ["CRITICAL", "SEVERE", "MODERATE", "MILD", "ALERT"]:
        if score >= LEVEL_THRESHOLDS[level]:
            return level
    return "CRITICAL"


# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("WakeVision Enhanced", cv2.WINDOW_NORMAL)
cv2.resizeWindow("WakeVision Enhanced", 1100, 680)

# ── State ─────────────────────────────────────────────────────────────────────
calibrated    = False
cal_start     = time.time()
ear_cal_buf   = []
mar_cal_buf   = []
EAR_THRESHOLD = 0.20
YAWN_THRESHOLD = 0.65

ear_smooth    = None
perclos_q     = deque()   # (timestamp, closed_flag) pairs
blink_times   = deque(maxlen=200)
yawn_times    = deque(maxlen=60)
blink_frames  = 0
blink_count   = 0
yawn_frames   = 0
eye_closed_start = None
micro_sleep   = False

pitch_smooth  = 0.0
yaw_smooth    = 0.0
roll_smooth   = 0.0
gaze_smooth   = 0.0

level_buffer  = deque(maxlen=12)
prev_level    = "ALERT"
last_sound    = {k: 0.0 for k in SOUND_COOLDOWN}

ride_start    = time.time()
max_perclos   = 0.0
level_counts  = {k: 0 for k in LEVEL_THRESHOLDS}

# ── Telemetry CSV ─────────────────────────────────────────────────────────────
ts_tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
telem_file = f"wakevision_telemetry_{ts_tag}.csv"
telem_fh   = open(telem_file, "w", newline="")
telem_writer = csv.writer(telem_fh)
telem_writer.writerow([
    "timestamp","elapsed_sec","ear","perclos","blink_rate_pm","yawn_rate_pm",
    "pitch","yaw","roll","gaze_dev","attention_score","level",
    "micro_sleep","blink_count","yawn_total"
])
last_telem_sec = -1

# ── Main loop ─────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    now = time.time()
    elapsed = now - ride_start
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_landmarker.detect(Image(ImageFormat.SRGB, rgb))

    if not result.face_landmarks:
        cv2.putText(frame, "No face detected", (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 80, 255), 3)
        cv2.imshow("WakeVision Enhanced", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    lm = np.array([(int(p.x*w), int(p.y*h)) for p in result.face_landmarks[0]])

    # ── EAR ─────────────────────────────────────────────────────────────────
    ear_raw = (ear(lm[LEFT_EYE_CONTOUR]) + ear(lm[RIGHT_EYE_CONTOUR])) / 2
    ear_smooth = ear_raw if ear_smooth is None else \
                 EAR_ALPHA * ear_raw + (1 - EAR_ALPHA) * ear_smooth

    # ── MAR (yawn) ───────────────────────────────────────────────────────────
    mar_val = mar(lm)

    # ── Head pose ────────────────────────────────────────────────────────────
    try:
        pitch_r, yaw_r, roll_r = head_pose_angles(lm, w, h)
        ALPHA_P = 0.15
        pitch_smooth = ALPHA_P * pitch_r + (1 - ALPHA_P) * pitch_smooth
        yaw_smooth   = ALPHA_P * yaw_r   + (1 - ALPHA_P) * yaw_smooth
        roll_smooth  = ALPHA_P * roll_r  + (1 - ALPHA_P) * roll_smooth
    except Exception:
        pass

    # ── Gaze ─────────────────────────────────────────────────────────────────
    try:
        g = gaze_score(lm, w, h)
        gaze_smooth = 0.12 * g + 0.88 * gaze_smooth
    except Exception:
        pass

    # ── Calibration phase ────────────────────────────────────────────────────
    if not calibrated:
        ear_cal_buf.append(ear_smooth)
        mar_cal_buf.append(mar_val)
        progress = min(1.0, (now - cal_start) / CALIBRATION_SEC)
        bar_w = int(progress * (w - 80))
        cv2.rectangle(frame, (40, h-60), (w-40, h-30), (80,80,80), 2)
        cv2.rectangle(frame, (40, h-60), (40+bar_w, h-30), (0,220,180), -1)
        cv2.putText(frame, f"Calibrating… {int(progress*100)}%  (look straight ahead)",
                    (40, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,180), 2)
        if progress >= 1.0:
            EAR_THRESHOLD  = np.percentile(ear_cal_buf, CAL_EAR_PCT) * 0.80
            YAWN_THRESHOLD = np.percentile(mar_cal_buf, 100 - 5) * 0.90
            calibrated = True
        cv2.imshow("WakeVision Enhanced", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ── Blink detection ──────────────────────────────────────────────────────
    closed = ear_smooth < EAR_THRESHOLD
    if closed:
        blink_frames += 1
        if eye_closed_start is None:
            eye_closed_start = now
        micro_sleep = (now - eye_closed_start) >= MICROSLEEP_SEC
    else:
        if blink_frames >= MIN_BLINK_FRAMES:
            blink_count += 1
            blink_times.append(now)
        blink_frames = 0
        eye_closed_start = None
        micro_sleep = False

    # ── PERCLOS (true 30-second window) ─────────────────────────────────────
    perclos_q.append((now, 1 if closed else 0))
    while perclos_q and now - perclos_q[0][0] > PERCLOS_WINDOW:
        perclos_q.popleft()
    perclos = (sum(v for _, v in perclos_q) / len(perclos_q)) * 100 if perclos_q else 0
    max_perclos = max(max_perclos, perclos)

    # ── Yawn detection ───────────────────────────────────────────────────────
    if mar_val > YAWN_THRESHOLD:
        yawn_frames += 1
    else:
        if yawn_frames >= MIN_YAWN_FRAMES:
            yawn_times.append(now)
        yawn_frames = 0

    # ── Rates ────────────────────────────────────────────────────────────────
    blink_rate_pm = len([t for t in blink_times if now - t <= 60]) / max(elapsed/60, 1/60)
    yawn_rate_pm  = len([t for t in yawn_times  if now - t <= 60]) / max(elapsed/60, 1/60)

    # ── Attention score & drowsiness level ──────────────────────────────────
    att = attention_score(perclos, blink_rate_pm, yawn_rate_pm,
                           pitch_smooth, yaw_smooth, micro_sleep, gaze_smooth)
    level_buffer.append(drowsiness_level(att))
    level = max(set(level_buffer), key=level_buffer.count)

    if level != prev_level:
        level_counts[level] += 1
        prev_level = level

    # ── Audio ────────────────────────────────────────────────────────────────
    lk = level.lower()
    if lk in _sound and lk in SOUND_COOLDOWN:
        if lk in ("severe", "critical"):
            if not ch[lk].get_busy():
                ch[lk].play(_sound[lk], loops=-1)
        else:
            if now - last_sound.get(lk, 0) > SOUND_COOLDOWN[lk]:
                ch[lk].play(_sound[lk])
                last_sound[lk] = now
    if lk not in ("severe", "critical"):
        for k in ("severe", "critical"):
            ch[k].stop()

    # ── Telemetry (once per second) ──────────────────────────────────────────
    sec_int = int(elapsed)
    if sec_int != last_telem_sec:
        telem_writer.writerow([
            datetime.now().isoformat(), f"{elapsed:.1f}",
            f"{ear_smooth:.4f}", f"{perclos:.2f}",
            f"{blink_rate_pm:.1f}", f"{yawn_rate_pm:.2f}",
            f"{pitch_smooth:.1f}", f"{yaw_smooth:.1f}", f"{roll_smooth:.1f}",
            f"{gaze_smooth:.3f}", f"{att:.1f}", level,
            int(micro_sleep), blink_count, len(yawn_times)
        ])
        telem_fh.flush()
        last_telem_sec = sec_int

    # ── Overlay UI ──────────────────────────────────────────────────────────
    overlay = frame.copy()
    col = LEVEL_COLORS[level]

    # Status banner
    cv2.rectangle(overlay, (0, 0), (w, 55), (col[2], col[1], col[0]), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(frame, f"LEVEL: {level}   |   Attention: {int(att)}",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 3)

    # Metric panel (top-left)
    mx, my = 20, 75
    def mtext(txt, dy=30, color=(255,255,255)):
        global my
        cv2.putText(frame, txt, (mx, my), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, color, 2)
        my += dy

    mtext(f"EAR         : {ear_smooth:.3f}  (thr {EAR_THRESHOLD:.3f})")
    mtext(f"PERCLOS     : {perclos:.1f}%  (max {max_perclos:.1f}%)")
    mtext(f"Blink rate  : {blink_rate_pm:.1f}/min  (total {blink_count})")
    mtext(f"Yawn rate   : {yawn_rate_pm:.2f}/min  (total {len(yawn_times)})")
    mtext(f"Head pitch  : {pitch_smooth:+.1f}°  yaw {yaw_smooth:+.1f}°")
    mtext(f"Gaze dev    : {gaze_smooth:.3f}")
    if micro_sleep:
        mtext("!! MICRO-SLEEP !!", color=(0, 0, 255))

    # Head direction label (top-right)
    if abs(pitch_smooth) < 12 and abs(yaw_smooth) < 15:
        hd = "CENTER"
    else:
        hd_v = "DOWN" if pitch_smooth < -12 else ("UP" if pitch_smooth > 12 else "")
        hd_h = "LEFT" if yaw_smooth > 15 else ("RIGHT" if yaw_smooth < -15 else "")
        hd = " ".join(filter(None, [hd_v, hd_h])) or "CENTER"
    cv2.putText(frame, f"HEAD: {hd}", (w - 280, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 240, 240), 2)

    # Attention bar
    bar_x, bar_y, bar_h = 20, h - 50, 22
    bar_len = w - 40
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_len, bar_y+bar_h), (60,60,60), -1)
    fill = int(att / 100 * bar_len)
    bar_col = (0,200,60) if att>80 else (0,200,200) if att>62 else \
              (0,150,255) if att>45 else (0,80,255) if att>28 else (0,0,220)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), bar_col, -1)
    cv2.putText(frame, f"Attention {int(att)}%", (bar_x+4, bar_y+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Eye landmark debug dots
    for idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
        cv2.circle(frame, tuple(lm[idx]), 2, (0, 255, 255), -1)

    cv2.imshow("WakeVision Enhanced", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Post-ride report ─────────────────────────────────────────────────────────
telem_fh.close()
ride_time = int(time.time() - ride_start)
avg_att   = 0
try:
    import pandas as pd
    df = pd.read_csv(telem_file)
    avg_att = int(df["attention_score"].mean())
except Exception:
    pass

report_file = f"wakevision_report_{ts_tag}.csv"
with open(report_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Ride Duration (sec)", ride_time])
    writer.writerow(["Average Attention Score", avg_att])
    writer.writerow(["Max PERCLOS (%)", f"{max_perclos:.1f}"])
    writer.writerow(["Blink Count", blink_count])
    writer.writerow(["Yawn Count", len(yawn_times)])
    for lv, cnt in level_counts.items():
        writer.writerow([f"Level '{lv}' entries", cnt])
    writer.writerow(["Telemetry file", telem_file])

print("\n╔══════════════════════════════════╗")
print("║   WAKEVISION SAFETY REPORT       ║")
print("╠══════════════════════════════════╣")
print(f"║  Ride Duration   : {ride_time:>6} sec     ║")
print(f"║  Avg Attention   : {avg_att:>6}         ║")
print(f"║  Max PERCLOS     : {max_perclos:>5.1f}%        ║")
print(f"║  Total Blinks    : {blink_count:>6}         ║")
print(f"║  Total Yawns     : {len(yawn_times):>6}         ║")
print(f"║  Report saved as : {report_file[:18]}... ║")
print("╚══════════════════════════════════╝\n")

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()