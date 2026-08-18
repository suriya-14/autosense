# AutoSense 🚗👁️

**AutoSense** — Real-time Driver Drowsiness Detection System powered by MediaPipe Face Landmarker.

## Features

- **Eye Aspect Ratio (EAR)** monitoring with adaptive calibration
- **PERCLOS** (% of eye closure over 30-second window)
- **Blink & Yawn rate** tracking
- **Head pose estimation** (pitch, yaw, roll) via solvePnP
- **Gaze deviation** scoring using iris landmarks
- **Multi-level drowsiness alerts**: Alert → Mild → Moderate → Severe → Critical
- **Audio warnings** with escalating severity
- **Micro-sleep detection**
- **Per-second telemetry logging** to CSV
- **Post-ride safety report**
- **Streamlit dashboard** for session review and analytics

## Project Structure

```
├── wakevision_final.py                    # Main detection engine (enhanced)
├── wakevision_final_with_head_direction.py # Earlier version with head direction
├── wakevision_full_face_points.py          # Face landmark visualiser
├── wakevision_ui.py                        # Streamlit analytics dashboard
├── models/
│   └── face_landmarker.task               # MediaPipe face landmarker model
└── sounds/
    ├── early.wav                           # Mild warning sound
    ├── strong.wav                          # Moderate warning sound
    └── emergency.wav                       # Severe/critical alarm
```

## Quick Start

```bash
# Install dependencies
pip install opencv-python mediapipe scipy numpy pygame

# Run the detection system
python wakevision_final.py

# Launch the dashboard (after a session)
pip install streamlit plotly pandas
streamlit run wakevision_ui.py
```

## How It Works

1. **Calibration** (25 sec): Look straight ahead to establish baseline EAR and yawn thresholds.
2. **Detection**: Tracks eyes, mouth, head pose, and gaze in real-time.
3. **Scoring**: Computes an attention score (0–100) from multiple biometric signals.
4. **Alerts**: Plays escalating audio warnings based on drowsiness level.
5. **Logging**: Records per-second telemetry and generates a post-ride report.

Press **Q** to stop the detection session.

## License

MIT
