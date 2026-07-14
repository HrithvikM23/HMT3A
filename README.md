# Human Motion Tracking Pipeline (Kinara)

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

Kinara is a local video and webcam motion-tracking pipeline with MediaPipe defaults, RTMPose CUDA body tracking, legacy YOLO body pose tracking, and ONNX hand pose inference. It supports single-person tracking, single-camera multi-person tracking, optional multi-camera fusion, live UDP output for Unreal-side receivers, and stack-safe output generation.

Project documentation:

- [Architecture And Data Flow](./docs/architecture.md)
- [Detailed Runtime Walkthrough](./docs/runtime-walkthrough.md)
- [Function Reference](./docs/function-reference.md)

This branch is currently optimized for practical local runtime use:
- RTMPose body tracking for CUDA-first single-person and multi-person flows
- legacy YOLO body tracking for compatibility and comparisons
- ONNX hand tracking with fallback hands and anatomical cleanup
- clothing-hint assisted multi-person identity stability
- rendered video, JSON, and FBX outputs

---

# Requirements

## Hardware

- Webcam or video file input
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

For RTX 50-series / `sm_120` systems, use RTMPose through ONNX Runtime:

```txt
--landmark-backend rtmpose --rtmpose-device cuda --rtmpose-mode balanced
```

If you want RTMPose to own both body and hand landmarks, use WholeBody mode:

```txt
--landmark-backend rtmpose-wholebody --rtmpose-device cuda --rtmpose-mode balanced
```

Body tracking uses MediaPipe pose landmarks by default. If you select the RTMPose backend, Kinara uses `rtmlib` RTMPose body tracking plus the configured hand backend. If you select RTMPose WholeBody, Kinara uses `rtmlib` WholeBody for body and both 21-point hands. If you select the YOLO backend, Kinara uses the legacy Ultralytics YOLO pose model path.

Default legacy YOLO body model when YOLO mode is selected:

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

For speed-first runs, MediaPipe can enter the same multi-person runner/export path, but MediaPipe Pose only returns one body per frame. Use YOLO when you need true multiple-person detection in one camera view.

Example identity hints:

```bash
py app\main.py --source ".\two_people.mp4" --max-people 2 --identity person1=black,orange --identity person2=gray,silver
```

Multi-camera plus multi-person can now run together through the fused multi-person path.

---

# Multi-Camera Support

Multi-camera mode is based on generic camera labels:

```txt
- CAM_0
- CAM_1
- CAM_2
- CAM_3
```

Repeated unlabeled `--source` values are assigned `CAM_0`, `CAM_1`, and so on. You may still provide explicit labels, but the recommended path is to keep camera names generic and let calibration/sync data describe where each camera is in the rig.

The current fusion path supports lightweight per-camera depth overrides from JSON and calibrated 3D reconstruction from a TOML file whose camera names match the source labels. Without calibration data, generic cameras use neutral depth settings; real non-flat 3D should come from calibrated reconstruction.

---

# Windows App Build

Kinara includes a native Windows launcher that can be built into an app folder with an `.exe`.

Recommended build command from the repo root:

```bat
build.cmd
```

You can also run the Python build script directly:

```bash
py -3.11 scripts/build_exe.py
```

The Python script is the most portable build path. It avoids PowerShell execution-policy problems on laptops where `.ps1` scripts are blocked.

PowerShell is still supported:

```powershell
.\scripts\build_exe.ps1
```

The build writes the app here:

```txt
artifacts\windows\Kinara\Kinara.exe
```

Run it by double-clicking `Kinara.exe`. Keep the full `artifacts\windows\Kinara\` folder together when moving it to another machine, because the EXE depends on the `_internal` files beside it.

Do not commit the built app folder to git. The build output is ignored by `.gitignore`; share it separately through a GitHub Release or a zipped build artifact.

Build/runtime notes:

- Set `KINARA_BUILD_PYTHON` or `KINARA_PYTHON` only when you need to force a specific Python 3.11 runtime.
- Keep runtime dependencies, downloaded models, caches, logs, and rendered outputs out of git. The project ignores `models/`, `.vendor_py*/`, `.kinara_runtime/`, `.kinara_logs/`, `outputs/`, and `artifacts/`.
- The EXE is distributed as a folder, not as a single standalone file. Keep `Kinara.exe` with its `_internal` folder and any runtime folders created by Check Runtime.

Double-clicking `Kinara.exe` opens a native Windows desktop launcher instead of requiring terminal input. The UI uses a resizable control-deck layout: the preview/log area and the right control panel can be resized with splitter handles, with minimum and maximum constraints so the app cannot be crushed into an unusable shape.

Launcher tabs:

- Capture: local camera input for quick tests.
- Files: recorded source selection and output destination.
- Presets: one-click setups for demo, quality export, multi-person, RTMPose, and ChArUco calibration situations.
- Calibration: camera calibration workflow for synchronized ChArUco videos, including output path selection and A3/Rescue board presets.
- Triangulation: calibrated 3D triangulation workflow with a browse button for `.toml` or `.json` calibration files.
- Tune: advanced CLI-backed controls, including Runtime people count, identity color hints, Python runtime selection, model/backend options, ChArUco detector tuning, triangulation tuning, smoothing, cleanup, and output settings.

Global launcher controls:

- Sun/moon theme button: dark mode starts with a sun icon; clicking it switches to light mode and changes the icon to a moon.
- Reset button: restores launcher defaults without clearing selected source files.
- Check Runtime: installs/checks selected runtime dependencies and prepares selected model assets.
- Start: runs the selected workflow directly and skips runtime bootstrap so demo starts are faster after Check Runtime has been used.

The launcher runs the existing Kinara pipeline underneath, so CLI behavior and GUI behavior stay aligned.

During a launcher run, the processed frame is streamed back into the large preview area inside the app. The OpenCV preview window stays disabled for the packaged launcher, so the rendered skeleton view appears in the EXE UI instead of opening a separate terminal/OpenCV window.

MediaPipe is not bundled into the launcher build. The default session uses MediaPipe, so Check Runtime installs it through the normal pip flow (`python -m pip install mediapipe==0.10.21`) with the configured Python runtime into the app-local `.vendor_py311` folder. Sessions that explicitly choose YOLO body plus ONNX hands do not install MediaPipe.

If Check Runtime has already prepared `.vendor_py311`, Start can skip the full dependency check while still loading those app-local packages.

---

# Model Management

Repo-managed model downloads go directly into the local `models/` tree:

```txt
models/body/
models/hand/
```

That means body and hand weights stay inside the project instead of being left in external caches.

RTMPose downloads are cached under:

```txt
.kinara_runtime/cache/rtmlib/
```

This folder is ignored by git.

# Smoke Testing Models

Use a neutral output basename when testing local videos so the source filename does not become an output filename:

```bash
py -3.11 -m kinara --source "<VIDEO_PATH>" --no-preview --benchmark-frames 3 --output-dir ".tmp_test_runtime/video_matrix" --output-basename "mediapipe_full" --landmark-backend mediapipe --model pose_landmark_full.tflite
```

Recommended smoke matrix:

```txt
MediaPipe: pose_landmark_lite.tflite, pose_landmark_full.tflite, pose_landmark_heavy.tflite
YOLO: yolo11n-pose.pt, yolo11s-pose.pt, yolo11m-pose.pt, yolo11l-pose.pt, yolo11x-pose.pt
ONNX hand variants: low, mid, high, max
RTMPose: lightweight, balanced, performance
RTMPose WholeBody: balanced
```

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
py app\main.py
```

You can also use the package entrypoint:

```bash
python -m kinara
```

Recommended multi-camera CLI flow:

```txt
Pass one --source per camera
Unlabeled sources become CAM_0, CAM_1, ...
Use matching camera names in calibration/sync files
Pipeline starts
```

Press `ESC` to close preview windows.

## Example Commands

Single-person:

```bash
py app\main.py --source ".\video.mp4" --model yolo11x-pose.pt
```

Realtime preview mode:

```bash
py app\main.py --source ".\video.mp4" --profile fastest --fps-log-interval 1
```

Fast preview on high-resolution clips:

```bash
py app\main.py --source ".\video.mp4" --profile fastest --processing-width 640
```

The output video stays at the original video size. Body/person detection can use the reduced working frame, but hand crops are taken from the original source frame so small fingers do not get blurred by downscale-then-upscale processing. The run prints a one-time line such as `source 1920x1080 -> inference 640x360` so you can confirm the body/person inference scale.

Saved videos include an FPS tracker overlay by default. Add `--no-fps-overlay` to hide it.

Balanced mode:

```bash
py app\main.py --source ".\video.mp4" --profile mid
```

MediaPipe body and hand mode:

```bash
py app\main.py --source ".\video.mp4" --landmark-backend mediapipe --model pose_landmark_full.tflite
```

Single-camera exports are flat in depth by default because one camera cannot reconstruct reliable metric 3D. To experiment with MediaPipe world-landmark relative Z, add:

```bash
py app\main.py --source ".\video.mp4" --landmark-backend mediapipe --single-camera-depth mediapipe
```

YOLO body with MediaPipe hands:

```bash
py app\main.py --source ".\video.mp4" --body-backend yolo --hand-backend mediapipe
```

Offline quality mode:

```bash
py app\main.py --source ".\video.mp4" --profile quality
```

Single-person with CPU hand fallback:

```bash
py app\main.py --source ".\video.mp4" --model yolo11x-pose.pt --provider CPUExecutionProvider
```

Faster preview/render pass with prediction between model frames:

```bash
py app\main.py --source ".\video.mp4" --profile fastest --yolo-device cuda:0 --fps-log-interval 1
```

Two-person tracking:

```bash
py app\main.py --source ".\two_people.mp4" --model yolo11x-pose.pt --max-people 2
```

Two-person tracking with identity hints:

```bash
py app\main.py --source ".\two_people.mp4" --model yolo11x-pose.pt --max-people 2 --identity person1=black,orange --identity person2=gray,silver
```

Multi-camera fused tracking from the CLI:

```bash
py app\main.py --source ".\cam0.mp4" --source ".\cam1.mp4" --max-people 2
```

Create a calibrated camera TOML from ChArUco calibration videos:

```bash
py app\main.py --calibrate-cameras --source ".\calib_cam0.mp4" --source ".\calib_cam1.mp4" --calibration-output ".\calibration.toml" --charuco-squares-x 11 --charuco-squares-y 8 --charuco-square-size 36 --charuco-marker-scale 0.6667 --charuco-marker-bits 4 --charuco-dict-size 50 --charuco-detection-strictness balanced
```

For weak board videos, the detector retry can be tuned directly:

```bash
py app\main.py --calibrate-cameras --source ".\calib_cam0.mp4" --source ".\calib_cam1.mp4" --calibration-output ".\calibration.toml" --charuco-detection-strictness lenient --charuco-retry-scale 3.5 --charuco-min-markers 6 --charuco-retry-sharpen
```

### Calibration Board

Use a real OpenCV ChArUco board for camera calibration. The current recommended A3 landscape board is:

```txt
Squares: 11 x 8
Square size: 36 mm
Marker size: 24 mm
Marker scale: 0.6667
Dictionary: DICT_4X4_50
Legacy pattern: off for boards generated by current OpenCV
```

Generate the board with OpenCV `cv2.aruco.CharucoBoard.generateImage()` and print on A3 landscape at actual size / 100% scale. If the printed square edge measures differently, pass the measured value to `--charuco-square-size`. Do not use a random ArUco grid; Kinara needs a ChArUco board whose marker IDs match OpenCV's board layout.

Some online generators create OpenCV's older ChArUco marker layout. If OpenCV detects the ArUco markers but reports zero ChArUco corners, enable `--charuco-legacy-pattern`. The launcher exposes this through Tune > Calibration, while the Calibration tab provides the normal A3 and Rescue presets.

For compressed, distant, or lower-resolution board videos, set `--charuco-detection-strictness lenient`. Use `balanced` for normal recorded calibration clips and `strict` only for clean, sharp footage. If needed, override the retry with `--charuco-retry-scale`, `--charuco-min-markers`, and `--charuco-retry-sharpen`.

Good calibration video habits:

- keep the board flat and rigid
- show it clearly to every camera
- move it through near, far, high, low, left, right, and tilted angles
- avoid motion blur and glare
- record enough frames where the board is visible in multiple cameras at once

Fused tracking with real calibrated 3D triangulation:

```bash
py app\main.py --source ".\cam0.mp4" --source ".\cam1.mp4" --triangulate-3d --calibration-3d ".\calibration.toml" --sync-offset CAM_1=3
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
| Live root motion in Unreal/Blender  | Complete |
| Android multi-phone LAN/WLAN streaming ingest | MASKON |
| Marker-glove hand tracking with joint dots | MASKON |
| Standalone desktop app build with .exe launcher | Complete |
| Native launcher control-deck UI     | Complete |
| Launcher presets and reset defaults | Complete |
| Launcher calibration workflow tab   | Complete |
| Launcher triangulation workflow tab | Complete |
| Auto identity re-lock after person crossings | Complete |
| Connection health monitor for ping, packet loss, latency, and FPS | MASKON |
| Hand inference decimation with interpolation between frames | Complete |
| Occlusion recovery for missing joints across camera views | Complete |
| Head and face tracking for head aim and facial landmarks | Head joints complete; face landmark export requires a face backend |
| Foot contact and ground lock to reduce foot sliding | Complete |
| Multi-person FBX export             | Complete |
| Calibration-aware 3D fusion         | Complete |
| Multi-camera + multi-person         | Complete |

---

# MASKON-Owned Integration Notes

## 3D Fusion

Continue improving reconstruction quality, sync tooling, and diagnostics around the calibrated triangulation path. The core calibration-aware 3D fusion path is implemented through `--triangulate-3d --calibration-3d`.

## Runtime Streaming

Keep the current UDP live-motion path and add an Unreal-side receiver/parser workflow around the v2 packet schema. The launcher no longer exposes the old UDP camera placeholder; downstream receiver integration can be built separately without blocking the local capture UI.

## Android Multi-Phone Capture

Android multi-phone capture is handled by MASKON. That side can hand Kinara normal camera or video sources using generic labels such as `CAM_0` and `CAM_1`.

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

