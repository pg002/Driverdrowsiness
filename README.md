# Driver Drowsiness Detection System

A real-time driver drowsiness monitoring project built with **Python**, **OpenCV**, and facial-landmark detection. It estimates eye closure and mouth opening from a webcam feed, displays visual warnings, and plays an audio alert when the detected patterns cross configured thresholds.

The repository includes two implementations:
- **DLIB** — uses dlib's frontal-face detector and 68-point facial landmark predictor.
- **MediaPipe** — uses MediaPipe Face Mesh landmarks.

> **Safety note:** This is an educational prototype, not a certified driver-safety device. It can miss drowsiness or trigger false alarms. Never rely on it as a substitute for rest, safe driving practices, or a validated in-vehicle safety system.

## Features

- Real-time webcam capture and face-landmark tracking.
- Eye Aspect Ratio (EAR) to help identify prolonged eye closure.
- Mouth Aspect Ratio (MAR) to help identify possible yawning.
- On-screen `Warning` and `Alert` indicators.
- Audio alert using `music.wav`.
- FPS display and frame-processing latency measurements.
- Saves processed frames locally.
- Plots EAR/MAR trends in the DLIB implementation and latency in both implementations.

## Project Structure

```text
Driverdrowsiness-main/
├── DLIB_Latest.py
├── MediaPipe_Latest.py
├── music.wav
├── README.md
└── models/
    └── shape_predictor_68_face_landmarks.dat  # required by DLIB version
```

The `models/` directory and predictor file are referenced by `DLIB_Latest.py`, but are not included in the supplied ZIP. Download the model separately as described below. The MediaPipe version does not require this dlib model file.

## Requirements

- Python 3.9–3.11 is a reasonable starting point; package compatibility depends on your operating system and Python version.
- A working webcam.
- Speakers or headphones for the sound alert.
- Internet access to install dependencies and download the DLIB landmark model.

## Installation

### 1. Extract the project

Extract the ZIP, then open a terminal in the project folder.

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

For the **MediaPipe implementation**:

```bash
pip install opencv-python mediapipe matplotlib pygame
```

For the **DLIB implementation**, install:

```bash
pip install opencv-python mediapipe matplotlib pygame numpy scipy imutils dlib
```

The DLIB implementation imports `dlib`, `imutils`, `scipy`, `numpy`, and `matplotlib`; the MediaPipe implementation imports `mediapipe`, `opencv-python`, `matplotlib`, and `pygame`. If `dlib` fails to install, use a Python version and installation method supported by the dlib release for your platform.

### 4. Configure the DLIB landmark model (DLIB version only)

Create a folder named `models` in the project directory and place the file below inside it:

`models/shape_predictor_68_face_landmarks.dat`

The file is available from the official dlib example model archive:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract the `.bz2` archive to obtain `shape_predictor_68_face_landmarks.dat`. Keep the filename and folder path exactly as shown, because the script loads it using a relative path.

### 5. Check the alert audio

Make sure `music.wav` is in the same directory as the Python scripts. Both implementations load this file by relative path.

## Run the Project

Run commands from the project root directory so the model and audio paths resolve correctly.

**MediaPipe version:**

```bash
python MediaPipe_Latest.py
```

**DLIB version:**

```bash
python DLIB_Latest.py
```

A window will open showing the webcam feed, facial landmarks, EAR/MAR values, and FPS. Press **`q`** while the video window is focused to exit. The scripts then release the webcam and close the OpenCV windows; the DLIB version also displays EAR, MAR, and latency plots, while the MediaPipe version displays a latency plot.

If the webcam does not open, close other applications that may be using it and check the camera permissions and camera index (`0` in the scripts).

## How It Works

### 1. Facial landmarks

The selected implementation detects facial landmarks on each webcam frame:
- **DLIB:** frontal-face detector and 68-point shape predictor.
- **MediaPipe:** Face Mesh landmarks.

### 2. Eye Aspect Ratio (EAR)

EAR compares vertical eye-landmark distances with the horizontal eye width. When the eye closes, EAR generally decreases. The code averages the left and right eye EAR values.

The implementations use threshold-based logic, including values around `0.25` for an alert and a higher threshold for a warning. These are heuristic defaults and may need calibration for a particular person, camera position, lighting, and frame rate.

### 3. Mouth Aspect Ratio (MAR)

MAR estimates mouth opening from vertical and horizontal mouth-landmark distances. A higher value can indicate a yawn, but it can also result from talking or other mouth movements. The scripts use MAR as an additional warning signal.

### 4. Warning and alert

The scripts count consecutive frames that meet the configured threshold conditions before displaying a warning or alert. The DLIB implementation uses `frame_check = 10`; the MediaPipe implementation uses a similar ten-frame counter. When the alert condition is met, the audio file is played.

### 5. Performance measurements

- **FPS:** estimated frames processed per second.
- **Latency:** elapsed time for processing a frame when a warning/alert condition is recorded.
- **Saved frames:** annotated images written to local output folders.

## Output Folders

The scripts create output folders as they run:

- `saved_frames_dlib/` — frames saved by the DLIB version.
- `saved_faces_dlib/` — directory created by the DLIB script; the current script does not save face crops into it.
- `saved_frames_mediapipe/` — frames saved by the MediaPipe version.

These folders can grow quickly because the scripts save frames continuously. Stop the program with `q` when finished and remove generated frames when they are no longer needed.

## Configuration

Thresholds are defined near the top of each script.

| Parameter | Purpose |
|---|---|
| `ear_alert_thresh` / `ear <= 0.25` | Lower EAR threshold associated with an alert |
| `ear_warning_thresh` / `ear <= 0.29` | EAR threshold associated with a warning |
| `mouth_thresh` / `mar >= 0.50` or `0.60` | Mouth-opening threshold used by the implementation |
| `frame_check` / frame counter `10` | Number of qualifying frames before a warning or alert |

The values differ slightly between implementations. Tune them only through controlled testing; they are not universal medical or safety thresholds.

## Troubleshooting

- **`FileNotFoundError` for the landmark model:** Confirm that `models/shape_predictor_68_face_landmarks.dat` exists in the project root's `models/` directory. This is only needed for DLIB.
- **`FileNotFoundError` for `music.wav`:** Run the script from the project directory and confirm the file is present.
- **Camera window is blank or closes immediately:** Check camera permissions, close other apps using the camera, and verify that `cv2.VideoCapture(0)` matches your webcam index.
- **`dlib` installation errors:** Try a Python version supported by your platform's dlib package, then follow the installation guidance for that version.
- **MediaPipe installation errors:** Check the supported Python versions for the installed MediaPipe release and use a compatible virtual environment.
- **No sound:** Check system volume and output device, and confirm `music.wav` plays normally.

## Limitations and Future Improvements

- Threshold-based detection can vary with lighting, glasses, camera angle, and individual facial features.
- Mouth opening alone cannot reliably distinguish yawning from talking.
- The current scripts save every processed frame, which may consume substantial disk space.
- Thresholds and frame counts are fixed in the scripts rather than exposed as command-line options.
- This prototype has not been validated for real-world driving or safety-critical use.

Potential improvements include configurable thresholds, saving frames only on events, adding a rolling eye-closure duration metric, logging events to CSV, and testing under varied lighting and camera conditions.

## License

No license file was included in the supplied project archive. Add a license before redistributing the project if you want to specify the terms under which others may use it.
