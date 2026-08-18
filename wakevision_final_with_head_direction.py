import cv2
import numpy as np
from scipy.spatial import distance
import time
from collections import deque
import pygame
import csv
from datetime import datetime
import os

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision

os.environ["SDL_AUDIODRIVER"] = "directsound"
pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()
pygame.mixer.init()

SOUND_EARLY = pygame.mixer.Sound("sounds/early.wav")
SOUND_STRONG = pygame.mixer.Sound("sounds/strong.wav")
SOUND_EMERGENCY = pygame.mixer.Sound("sounds/emergency.wav")

ch_early = pygame.mixer.Channel(0)
ch_strong = pygame.mixer.Channel(1)
ch_emergency = pygame.mixer.Channel(2)

last_early = 0
last_strong = 0

MODEL_PATH = "models/face_landmarker.task"
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1
    )
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
UPPER_LIP = [13,14]
LOWER_LIP = [17,18]
MOUTH_CORNERS = [61,291]
NOSE_TIP = 1

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(lm):
    return distance.euclidean(lm[13], lm[17]) / distance.euclidean(lm[61], lm[291])

def get_head_direction(landmarks):
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]

    face_cx = np.mean(xs)
    face_cy = np.mean(ys)

    nose_x, nose_y = landmarks[NOSE_TIP]

    dx = nose_x - face_cx
    dy = nose_y - face_cy

    thresh_x = (xs.max() - xs.min()) * 0.08
    thresh_y = (ys.max() - ys.min()) * 0.08

    if dx > thresh_x:
        horiz = "RIGHT"
    elif dx < -thresh_x:
        horiz = "LEFT"
    else:
        horiz = "CENTER"

    if dy > thresh_y:
        vert = "BOTTOM"
    elif dy < -thresh_y:
        vert = "TOP"
    else:
        vert = "CENTER"

    if vert == "CENTER" and horiz == "CENTER":
        return "CENTER"
    elif vert == "CENTER":
        return horiz
    elif horiz == "CENTER":
        return f"{vert} CENTER"
    else:
        return f"{vert} {horiz}"

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
cv2.namedWindow("WakeVision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("WakeVision", 900, 650)

# ===================== PARAMETERS =====================
EAR_ALPHA = 0.25
PERCLOS_ALPHA = 0.15
MICROSLEEP_TIME = 2.0

FPS = 20
WINDOW_SECONDS = 30
WINDOW_SIZE = FPS * WINDOW_SECONDS

MIN_BLINK_FRAMES = 3
MIN_YAWN_FRAMES = 15

# ===================== CALIBRATION =====================
CALIBRATION_TIME = 30
cal_start = time.time()
ear_base = []
mar_base = []
calibrated = False

EAR_THRESHOLD = 0.20
YAWN_THRESHOLD = 0.6

# ===================== STATE =====================
eye_closed_frames = deque(maxlen=WINDOW_SIZE)
eye_closed_start = None

blink_frames = 0
blink_count = 0
blink_times = deque(maxlen=100)

yawn_frames = 0
yawn_times = deque(maxlen=20)

ear_smooth = None
perclos_smooth = None
max_perclos = 0

status_buffer = deque(maxlen=15)
prev_status = "NORMAL"

emergency_start = None
emergency_triggered = False
emergency_events = 0

early_cnt = 0
strong_cnt = 0
emergency_cnt = 0

ride_start = time.time()
focus_log = []

# ===================== MAIN LOOP =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_landmarker.detect(Image(ImageFormat.SRGB, rgb))

    if result.face_landmarks:
        lm = np.array([(int(p.x*w), int(p.y*h)) for p in result.face_landmarks[0]])

        head_direction = get_head_direction(lm)

        ear_raw = (eye_aspect_ratio(lm[LEFT_EYE]) +
                   eye_aspect_ratio(lm[RIGHT_EYE])) / 2
        ear_smooth = ear_raw if ear_smooth is None else \
                     EAR_ALPHA * ear_raw + (1 - EAR_ALPHA) * ear_smooth

        mar = mouth_aspect_ratio(lm)

        # ---------- CALIBRATION ----------
        if not calibrated:
            ear_base.append(ear_smooth)
            mar_base.append(mar)

            cv2.putText(frame, "CALIBRATING... Keep normal face",
                        (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,255), 3)

            if time.time() - cal_start > CALIBRATION_TIME:
                EAR_THRESHOLD = np.mean(ear_base) * 0.75
                YAWN_THRESHOLD = np.mean(mar_base) * 1.4
                calibrated = True

            cv2.imshow("WakeVision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ---------- BLINK ----------
        closed = ear_smooth < EAR_THRESHOLD
        if closed:
            blink_frames += 1
        else:
            if blink_frames >= MIN_BLINK_FRAMES:
                blink_count += 1
                blink_times.append(time.time())
            blink_frames = 0

        # ---------- PERCLOS ----------
        eye_closed_frames.append(1 if closed else 0)
        perclos_raw = (sum(eye_closed_frames)/len(eye_closed_frames))*100
        perclos_smooth = perclos_raw if perclos_smooth is None else \
            PERCLOS_ALPHA*perclos_raw + (1-PERCLOS_ALPHA)*perclos_smooth
        max_perclos = max(max_perclos, perclos_smooth)

        # ---------- MICRO-SLEEP ----------
        micro_sleep = False
        if closed:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start >= MICROSLEEP_TIME:
                micro_sleep = True
        else:
            eye_closed_start = None

        # ---------- YAWN ----------
        if mar > YAWN_THRESHOLD:
            yawn_frames += 1
        else:
            if yawn_frames >= MIN_YAWN_FRAMES:
                yawn_times.append(time.time())
            yawn_frames = 0

        recent_yawns = len([t for t in yawn_times if time.time() - t <= 120])
        blink_rate = len([t for t in blink_times if time.time() - t <= 60])

        # ---------- FOCUS ----------
        focus = 100 \
                - (perclos_smooth * 0.6) \
                - (blink_rate * 1.2) \
                - (recent_yawns * 10)

        focus = max(0, min(100, focus))
        focus_log.append(focus)

        # ---------- STATUS ----------
        if micro_sleep:
            status = "MICRO-SLEEP"
        elif focus < 25:
            status = "EMERGENCY"
        elif focus < 40:
            status = "STRONG WARNING"
        elif focus < 60:
            status = "EARLY WARNING"
        else:
            status = "NORMAL"

        status_buffer.append(status)
        status = max(set(status_buffer), key=status_buffer.count)

        if status != prev_status:
            if status == "EARLY WARNING": early_cnt += 1
            elif status == "STRONG WARNING": strong_cnt += 1
            elif status in ["EMERGENCY","MICRO-SLEEP"]: emergency_cnt += 1
            prev_status = status

        # ---------- AUDIO ----------
        t = time.time()
        if status == "EARLY WARNING" and t-last_early > 5:
            ch_early.play(SOUND_EARLY); last_early = t
        elif status == "STRONG WARNING" and t-last_strong > 3:
            ch_strong.play(SOUND_STRONG); last_strong = t
        elif status in ["EMERGENCY","MICRO-SLEEP"]:
            if not ch_emergency.get_busy():
                ch_emergency.play(SOUND_EMERGENCY, loops=-1)
        else:
            ch_emergency.stop()

        # ---------- UI ----------
        y = 40
        cv2.putText(frame, f"Focus Score : {int(focus)}", (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        y+=35
        cv2.putText(frame, f"PERCLOS : {perclos_smooth:.1f}%", (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        y+=35
        cv2.putText(frame, f"Blinks : {blink_count}", (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        y+=35
        cv2.putText(frame, f"Yawns : {recent_yawns}", (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        y+=40
        cv2.putText(frame, f"STATUS : {status}", (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,
                    (0,200,0) if status=="NORMAL" else (0,0,255),3)

        cv2.putText(frame, f"HEAD: {head_direction}",
                    (w - 340, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 3)

    cv2.imshow("WakeVision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== POST-RIDE REPORT =====================
ride_time = int(time.time() - ride_start)
avg_focus = int(sum(focus_log)/len(focus_log)) if focus_log else 0

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = f"wakevision_report_{ts}.csv"

with open(report_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Ride Duration (sec)", ride_time])
    writer.writerow(["Average Focus Score", avg_focus])
    writer.writerow(["Max PERCLOS (%)", f"{max_perclos:.1f}"])
    writer.writerow(["Blink Count", blink_count])
    writer.writerow(["Yawns (recent)", len(yawn_times)])
    writer.writerow(["Early Warnings", early_cnt])
    writer.writerow(["Strong Warnings", strong_cnt])
    writer.writerow(["Emergency Warnings", emergency_cnt])
    writer.writerow(["Emergency Stop Events", emergency_events])

print("\nWAKEVISION SAFETY REPORT")
print("------------------------")
print(f"Ride Duration      : {ride_time} sec")
print(f"Average Focus      : {avg_focus}")
print(f"Max PERCLOS        : {max_perclos:.1f}%")
print(f"Total Blinks       : {blink_count}")
print(f"Total Yawns        : {len(yawn_times)}")
print(f"Emergency Events   : {emergency_events}")
print(f"Report saved as    : {report_file}\n")

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
