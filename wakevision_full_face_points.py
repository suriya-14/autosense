import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    if result.face_landmarks:
        landmarks = [
            (int(lm.x * w), int(lm.y * h))
            for lm in result.face_landmarks[0]
        ]

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        cv2.putText(
            frame,
            f"Facial Points: {len(landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("WakeVision - Full Facial Landmark Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
