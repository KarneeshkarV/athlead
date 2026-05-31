# Athlead

Computer-vision experiments for **athlete pose estimation, movement tracking, and activity classification** from video. Athlead combines multiple pose backends (MediaPipe, OpenPose, YOLOv8) with speed estimation and a small PyTorch classifier for running vs. sitting.

> Research prototype — scripts and notebooks were built iteratively for demos and data collection, not as a packaged library.

## Features

| Area | What it does |
|------|----------------|
| **Pose estimation** | Extract body keypoints via MediaPipe, OpenPose (`graph_opt.pb`), or YOLOv8-pose |
| **Speed tracking** | Track athletes in video with YOLO + ByteTrack; estimate speed from pixel motion and a calibrated real-world distance |
| **Joint angles** | Compute limb angles from pose keypoints; optional live matplotlib plots |
| **Activity classification** | Classify **running vs. sitting** from MediaPipe landmarks using a trained PyTorch model (`RunningSitting_v2.pth`) |
| **Sports2D integration** | `Config_demo.toml` configures joint/segment angle pipelines via [Sports2D](https://github.com/davidpagnon/Sports2D) (BlazePose / OpenPose) |
| **Data pipeline** | Download training videos from YouTube, batch-extract landmarks to JSON, stream sensor/pose data over TCP, visualize with Dash |

## Tech stack

- **Python** — OpenCV, NumPy, Matplotlib
- **Ultralytics YOLOv8** + **supervision** — detection, pose, ByteTrack, annotated output video
- **MediaPipe** — BlazePose landmarks and static-image batch processing
- **PyTorch** — running/sitting binary classifier
- **OpenPose (TensorFlow)** — `tf_open_pose.py` + `graph_opt.pb` for COCO-style pose on images/video
- **Dash / Plotly** — real-time multi-device data dashboard (`getDataViz.py`)
- **Jupyter** — `openPose.ipynb`, `yoloPose.ipynb`, `mode_running_sitting.ipynb` for exploration

## Repository layout

```
├── combinedV1.py              # Combined speed + pose + angle tracking (main demo)
├── speed_tracking.py          # Athlete speed from video (YOLO + ByteTrack)
├── yoloPose.py                # YOLOv8-pose with optional angle plots
├── model_testing_running_sitting.py  # MediaPipe + PyTorch running/sitting classifier
├── mediapip_get_coornidates.py       # Batch landmark extraction to JSON
├── tf_open_pose.py            # OpenPose inference via TensorFlow graph
├── Config_demo.toml           # Sports2D pose/angle configuration
├── getData.py                 # Simple TCP server for incoming pose/sensor data
├── getDataViz.py              # Dash dashboard for streamed device readings
├── download.py / download_v2.py      # YouTube video download (pytube / yt-dlp)
├── get_link.py                # YouTube search helpers
├── graph_opt.pb               # OpenPose TensorFlow model weights
├── *.ipynb                    # Notebooks for pose and activity experiments
├── run_test*.jpg, test*.jpg   # Sample still frames for pose tests
└── result_*_coco.png          # Example OpenPose / pose output images
```

## Getting started

### Prerequisites

- Python 3.10+
- GPU recommended for YOLO and PyTorch (CUDA or Apple MPS)
- FFmpeg (for video read/write via OpenCV)

### Install dependencies

There is no `requirements.txt` yet; install from the imports used across scripts:

```bash
pip install opencv-python numpy matplotlib tqdm ultralytics supervision mediapipe torch pytube yt-dlp dash plotly pandas
```

For Sports2D angle workflows, follow the [Sports2D install guide](https://github.com/davidpagnon/Sports2D) and use the bundled `Config_demo.toml`.

### Run the combined tracking demo

Place input video under `./vid/` (or edit paths in the script), then:

```bash
python combinedV1.py
```

Calibrate speed by setting `known_real_world_distance_meters` and `measured_pixel_distance` in `CombinedTracker` — measure a known distance in the frame (e.g. track lane width) in pixels.

### Other entry points

```bash
# Speed-only tracking
python speed_tracking.py

# YOLO pose on a video file (edit video_name in script)
python yoloPose.py

# Running vs. sitting classifier (requires ./models/RunningSitting_v2.pth)
python model_testing_running_sitting.py

# OpenPose on image or webcam
python tf_open_pose.py --input run_test1.jpg

# TCP data receiver
python getData.py

# Live Dash dashboard for streamed JSON readings
python getDataViz.py
```

## Sample outputs

The repo includes annotated stills (`result_test*_coco.png`, `result_tf_test*_coco.png`) and example tracking videos (`tarun_combined_tracking.mp4`, `tarun_moving_frame.mp4`) from development runs.

## Notes & limitations

- **`./vid/` and `./models/`** are not committed — add your own videos and place `RunningSitting_v2.pth` under `models/` for the classifier script.
- Paths, thresholds, and calibration values are **hardcoded** per experiment; adjust before reuse.
- `mediapip_get_coornidates.py` expects a local `Database/Running and sitting/` image set for batch JSON export.
- OpenPose via Sports2D requires a separate Windows/Linux install and `openpose_path` in `Config_demo.toml`.

## Status

Early-stage research code. Useful as a reference for pose + tracking pipelines; not production-ready.

## License

No license file is specified. Contact the repository owner before redistribution.
