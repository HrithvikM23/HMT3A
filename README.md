# Human Motion Tracking Pipeline (Kinara)

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

Kinara is a local video and webcam motion-tracking pipeline built around a YOLO-based body pose tracking and ONNX hand pose inference. It supports single-person tracking, single-camera multi-person tracking, optional multi-camera fusion, live UDP output for Unreal-side receivers, and stack-safe output generation.

Project documentation:

- [Architecture And Data Flow](./docs/architecture.md)
- [Detailed Runtime Walkthrough](./docs/runtime-walkthrough.md)
- [Function Reference](./docs/function-reference.md)

This branch is currently optimized for practical local runtime use:
- YOLO body tracking for both single-person and multi-person flows
- ONNX hand tracking with fallback hands and anatomical cleanup
- clothing-hint assisted multi-person identity stability
- rendered video, JSON, and FBX outputs

---

# Requirements

## Hardware

- Webcam or video file input
- 16 GB RAM recommended
- NVIDIA GPU strongly recommended for YOLO pose inference
- Multi-camera setup optional

## Required Software

| Tool | Version | Notes |
| --- | --- | --- |
| Unreal Engine | 5.4 | Target animation runtime |
| Python | 3.11.9 | Pipeline scripting |
| CUDA Toolkit | 12.x | Optional acceleration for NVIDIA systems |
| cuDNN | 9.x | Used indirectly by PyTorch-backed GPU inference |

---

# Installation

## 1. Install Python

Download from:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

Enable Python on `PATH` during installation if you want terminal launch support.

---

## 2. Install NVIDIA Runtime Stack (Optional but Recommended)

Install:

```txt
- recent NVIDIA driver
- CUDA 12.x or newer
- cuDNN 9.x
```

If ONNX Runtime reports a missing `cudnn64_9.dll`, add your cuDNN `bin` directory to `PATH`.

---

## 3. Install Python Dependencies

Example GPU setup:

```bash
pip install ultralytics torch torchvision numpy opencv-python onnxruntime-gpu
```

CPU-only hand inference fallback:

```bash
pip install ultralytics torch torchvision numpy opencv-python onnxruntime
```

---

## 4. Clone The Project

```bash
git clone https://github.com/HrithvikM23/Kinara.git
cd Kinara
```

---

# System Architecture

```txt
Webcam / Video File(s)
        ↓
YOLO Body Pose Detection
        ↓
Per-Person Hand Detection
        ↓
Temporal Smoothing + Hold + Fallback
        ↓
Identity Stabilization / Cross-Person Hand Guard
        ↓
Rendered Output / JSON / FBX / Live UDP
```

---

# Tracking System

## Body Tracking

Body tracking uses an Ultralytics YOLO pose model by default. If you select the MediaPipe backend, the body model is a MediaPipe pose landmark TFLite asset.

Default body model:

```txt
yolo11x-pose.pt
```

You can replace the YOLO weights with any compatible YOLO pose file through `--model`. In MediaPipe mode, use the actual MediaPipe model filename with `--model`, for example `pose_landmark_full.tflite`.

If you pass a known YOLO filename such as `yolo11x-pose.pt`, `yolo11l-pose.pt`, `yolo11m-pose.pt`, `yolo11s-pose.pt`, or `yolo11n-pose.pt` and it is missing, Kinara downloads it directly into:

```txt
models/body/
```

MediaPipe pose model names are:

```txt
pose_landmark_lite.tflite
pose_landmark_full.tflite
pose_landmark_heavy.tflite
```

Kinara stages MediaPipe pose TFLite assets in `models/body/`. MediaPipe hand TFLite assets are staged in `models/hand/mediapipe/` when MediaPipe hands are enabled.

## Hand Tracking

Hand tracking uses a YOLO26 hand-pose ONNX model.

Preset mapping:
- `low`, `mid` -> FP16 variant
- `high`, `max` -> FP32 variant

Downloaded hand models are stored in:

```txt
models/hand/
```

Each hand outputs 21 landmarks.

## Hand Robustness Layer

The hand pipeline includes:

```txt
- wrist-directed crop generation
- temporal hold when a hand briefly disappears
- wrist-attached default hand fallback
- anatomical cleanup and distance constraints
- cross-person hand ownership rejection in multi-person mode
```

---

# Output Types

## Single-Person

Single-person runs currently write:

```txt
- rendered tracking video
- motion JSON export
- FBX export
```

The JSON export now includes Blender-facing rig metadata:

```txt
- skeleton hierarchy
- normalized coordinate system (Z-up)
- stable rest-joint positions
```

## Multi-Person

Single-camera multi-person runs currently write:

```txt
- rendered tracking video
- multi-person JSON export
- live UDP packets
```

Multi-person FBX export is not the default output path yet.
Multi-person JSON export now carries the same normalized Blender metadata as single-person export.

---

# Blender Import

Import a Kinara JSON clip into Blender as a real armature animation with:

```bash
blender --python .\blender_kinematics\import_kinara_motion.py -- --input .\outputs\your_clip.json
```

For multi-person files, import a specific label with:

```bash
blender --python .\blender_kinematics\import_kinara_motion.py -- --input .\outputs\your_clip.json --person person1
```

---

# Multi-Person Tracking

Single-camera multi-person mode is currently the supported path.

It uses:

```txt
- YOLO pose detections and tracker IDs
- box continuity
- optional clothing color hints
- wrist ownership checks to reduce hand stealing during crossings
```

Example identity hints:

```bash
py main.py --source ".\two_people.mp4" --max-people 2 --identity person1=black,orange --identity person2=gray,silver
```

Multi-camera plus multi-person can now run together through the fused multi-person path.

---

# Multi-Camera Support

Multi-camera mode currently supports:

```txt
- FRONT
- BACK
- LEFT
- RIGHT
```

The current fusion path now supports view-aware depth estimation with optional per-camera calibration overrides from JSON.

---

# Model Management

Repo-managed model downloads go directly into the local `models/` tree:

```txt
models/body/
models/hand/
```

That means body and hand weights stay inside the project instead of being left in external caches.

---

# Live UDP Output

Kinara can stream live UDP packets for downstream receivers such as Unreal-side runtime tools.

Current packet content includes:

```txt
- person IDs
- person labels
- frame metadata
- fused camera-view labels
- body landmarks
- hand landmarks
- hand boxes
- joint maps
- derived head / neck / foot / toe joints
```

Live UDP is disabled unless you pass `--osc-enabled`.

The default live UDP target values are centralized in [config.py](./config.py):

```txt
HOST = 127.0.0.1
PORT = 9000
ENABLED = false
```

See [cli-args.md](./docs/cli-args.md) for host/port flags.

---

# Output Naming

Rendered and export outputs are stack-safe and never overwrite previous runs.

Example:

```txt
outputs/dance rendered-1.mp4
outputs/dance json-1.json
outputs/dance fbx-1.fbx
```

The next run becomes `-2`, then `-3`, and so on.

---

# How To Run

## Interactive Mode

```bash
cd [drive]:\[path]\Kinara
py main.py
```

You can also use the package entrypoint:

```bash
python -m kinara
```

Program flow:

```txt
Select input source
1 -> Webcam
2 -> Video file(s)
If video mode:
  Enter number of cameras
  Assign FRONT/BACK/LEFT/RIGHT roles
  Pick one video per assigned role
Pipeline starts
```

Press `ESC` to close preview windows.

## Example Commands

Single-person:

```bash
py main.py --source ".\video.mp4" --model yolo11x-pose.pt
```

Realtime preview mode:

```bash
py main.py --source ".\video.mp4" --profile fastest --fps-log-interval 1
```

Fast preview on high-resolution clips:

```bash
py main.py --source ".\video.mp4" --profile fastest --processing-width 640
```

The output video stays at the original video size; only the model's working frame is reduced. The run prints a one-time line such as `source 1920x1080 -> inference 640x360` so you can confirm the scale.

Saved videos include an FPS tracker overlay by default. Add `--no-fps-overlay` to hide it.

Balanced mode:

```bash
py main.py --source ".\video.mp4" --profile mid
```

MediaPipe body and hand mode:

```bash
py main.py --source ".\video.mp4" --landmark-backend mediapipe --model pose_landmark_full.tflite
```

Single-camera exports are flat in depth by default because one camera cannot reconstruct reliable metric 3D. To experiment with MediaPipe world-landmark relative Z, add:

```bash
py main.py --source ".\video.mp4" --landmark-backend mediapipe --single-camera-depth mediapipe
```

YOLO body with MediaPipe hands:

```bash
py main.py --source ".\video.mp4" --body-backend yolo --hand-backend mediapipe
```

Offline quality mode:

```bash
py main.py --source ".\video.mp4" --profile quality
```

Single-person with CPU hand fallback:

```bash
py main.py --source ".\video.mp4" --model yolo11x-pose.pt --provider CPUExecutionProvider
```

Faster preview/render pass with prediction between model frames:

```bash
py main.py --source ".\video.mp4" --profile fastest --yolo-device cuda:0 --fps-log-interval 1
```

Two-person tracking:

```bash
py main.py --source ".\two_people.mp4" --model yolo11x-pose.pt --max-people 2
```

Two-person tracking with identity hints:

```bash
py main.py --source ".\two_people.mp4" --model yolo11x-pose.pt --max-people 2 --identity person1=black,orange --identity person2=gray,silver
```

Multi-camera fused tracking from the CLI:

```bash
py main.py --source ".\cam0.mp4" --source ".\cam1.mp4" --max-people 2
```

Create a calibrated camera TOML from Charuco calibration videos:

```bash
py main.py --calibrate-cameras --source ".\calib_cam0.mp4" --source ".\calib_cam1.mp4" --calibration-output ".\calibration.toml" --charuco-square-size 35
```

### Calibration Board

Use a Charuco board for camera calibration. The recommended starter board is:

```txt
Squares: 7 x 5
Square size: 35 mm
Marker size: 28 mm
Marker scale: 0.8
```

Print it as large and flat as practical. A4 can work for close webcams, but A3 is better because the board remains readable from more positions in the capture space. The most important value is the true measured edge length of one square. If the printed square edge is 30 mm, pass `--charuco-square-size 30`; if it is 35 mm, pass `--charuco-square-size 35`.

Good calibration video habits:

- keep the board flat and rigid
- show it clearly to every camera
- move it through near, far, high, low, left, right, and tilted angles
- avoid motion blur and glare
- record enough frames where the board is visible in multiple cameras at once

Fused tracking with real calibrated 3D triangulation:

```bash
py main.py --source ".\cam0.mp4" --source ".\cam1.mp4" --triangulate-3d --calibration-3d ".\calibration.toml" --sync-offset CAM_1=3
```

All CLI arguments are documented in [cli-args.md](./docs/cli-args.md).

---

# Current Development Status

| Component                           | Status   |
| ---                                 | ---      |
| Webcam input                        | Complete |
| Video file input                    | Complete |
| Interactive source selection        | Complete |
| Camera role selection               | Complete |
| Multi-camera synchronized input     | Complete |
| YOLO single-person body tracking    | Complete |
| YOLO multi-person body tracking     | Complete |
| ONNX hand tracking                  | Complete |
| Clothing color identity hints       | Complete |
| Cross-person hand guard             | Complete |
| Automatic model download to models/ | Complete |
| Stack-safe output naming            | Complete |
| Temporal smoothing                  | Complete |
| Rendered output video               | Complete |
| JSON export generation              | Complete |
| FBX export generation               | Complete |
| Live UDP streaming                  | Complete |
| Live root motion in Unreal/Blender  | Planned  |
| Live-only low-RAM mode with optional exports | Planned  |
| Android multi-phone LAN/WLAN streaming ingest | Planned  |
| Marker-glove hand tracking with joint dots | Planned  |
| Standalone desktop app build with .exe launcher | Planned  |
| Auto identity re-lock after person crossings | Planned  |
| Connection health monitor for ping, packet loss, latency, and FPS | Planned  |
| Hand inference decimation with interpolation between frames | Planned  |
| Occlusion recovery for missing joints across camera views | Planned  |
| Head and face tracking for head aim and facial landmarks | Planned  |
| Retarget presets for Unreal, MetaHuman, Mixamo, and Rigify | Planned  |
| Foot contact and ground lock to reduce foot sliding | Planned  |
| Multi-person FBX export             | Working |
| Calibration-aware 3D fusion         | Working  |
| Multi-camera + multi-person         | Complete |

---

# Roadmap

## 3D Fusion

Improve the current view-aware depth estimation into stronger calibration-driven reconstruction once measured camera rigs are available.

## Runtime Streaming

Keep the current UDP live-motion path and add an Unreal-side receiver/parser workflow around the v2 packet schema.

## Android Multi-Phone Capture

Add an Android client flow where up to four phones can stream camera video over LAN/WLAN to a PC receiver. The planned backend path includes port-based client listening, camera-role assignment such as FRONT/BACK/LEFT/RIGHT/UP, a connection handshake/health-check before capture begins, bounded buffering between ingest and inference, and confidence-based multi-camera fusion for final output generation.

---

# Technology Stack

## Inference

- Ultralytics YOLO pose
- ONNX Runtime
- YOLO26 hand pose ONNX

## Computer Vision

- OpenCV
- NumPy
- Python

## Stabilization

- Exponential landmark smoothing
- Landmark hold / decay
- Default hand fallback
- Cross-person hand ownership filtering

---

# License

See [LICENSE](./LICENSE.md).
