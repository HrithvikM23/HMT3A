# Pose and Hand Landmark Pipeline Arguments

## General

`--profile`  
Function: Selects the app-facing runtime mode.  
Accepted values: `fastest`, `mid`, `quality`.  
Default: `fastest`  
Notes: `fastest` uses the light body model, FP16 body inference on CUDA when available, a 640px processing frame, and reduced hand cadence. `mid` balances speed and stability. `quality` keeps the heaviest defaults for offline renders. Any explicit argument you pass with a profile overrides that profile value.

`--calibrate-cameras`  
Function: Creates a calibrated camera TOML from synchronized Charuco calibration videos.  
Accepted values: flag only.  
Notes: Use with at least two `--source` video files and `--calibration-output`.

`--source`  
Function: Selects the input source.  
Accepted values: webcam index like `0`, `1`, `2`, a path to a video file, or repeated labeled values like `CAM_0=cam0.mp4` and `CAM_1=cam1.mp4`.  
Usage: repeat the argument for multi-camera runs. Unlabeled sources are auto-labeled as `CAM_0`, `CAM_1`, and so on.

`--output`  
Function: Sets the output video name template.  
Accepted values: any writable video path.  
Notes: The actual saved file is stacked as `"<name> rendered-N.<ext>"`.

`--output-dir`  
Function: Sets the directory where rendered, fbx, and json outputs are written.  
Accepted values: any writable directory path.

`--output-basename`  
Function: Sets the shared filename prefix for rendered, fbx, and json outputs.  
Accepted values: any non-empty text string.

`--calibration-output`  
Function: Sets the output TOML path for `--calibrate-cameras`.  
Accepted values: writable `.toml` path or output directory.

`--charuco-squares-x`  
Function: Sets the Charuco board square count along X for calibration.  
Accepted range: integer `> 0`.  
Default: `7`

`--charuco-squares-y`  
Function: Sets the Charuco board square count along Y for calibration.  
Accepted range: integer `> 0`.  
Default: `5`

`--charuco-square-size`  
Function: Sets the real square size for calibration in your chosen units.  
Accepted range: float `> 0`.  
Default: `1.0`

`--charuco-marker-scale`  
Function: Sets marker length as a fraction of square size.  
Accepted range: float `> 0`.  
Default: `0.8`

Recommended board: 7 x 5 Charuco squares, 35 mm square size, 28 mm marker size. If your printed square edge measures differently, use the measured value for `--charuco-square-size`.

## Model Selection

`--model`  
Function: Selects the body model for the active landmark backend. With YOLO it selects the Ultralytics pose weights. With MediaPipe it selects the MediaPipe pose landmark TFLite asset.  
Accepted values: YOLO filenames like `yolo11x-pose.pt`, `yolo11l-pose.pt`, `yolo11m-pose.pt`, `yolo11s-pose.pt`, `yolo11n-pose.pt`, a custom local YOLO path, or MediaPipe pose names `pose_landmark_lite.tflite`, `pose_landmark_full.tflite`, `pose_landmark_heavy.tflite`.  
Default: `yolo11n-pose.pt` in the default `fastest` profile for YOLO mode, `pose_landmark_full.tflite` for MediaPipe mode. `--profile quality` uses `yolo11x-pose.pt` unless you pass an explicit `--model`.  
Notes: Known YOLO models are stored in `models/body/`. MediaPipe pose TFLite assets are also staged in `models/body/`; MediaPipe hand TFLite assets are staged in `models/hand/mediapipe/` when that backend is used. MediaPipe may still need a runtime compatibility copy inside its Python package, but Kinara keeps the managed copy in `models/`.

`--landmark-backend`  
Function: Selects the high-level landmark backend family.  
Accepted values: `yolo`, `mediapipe`, `hybrid`.  
Default: `mediapipe`  
Notes: `yolo` maps to `--body-backend yolo --hand-backend onnx`. `mediapipe` maps to `--body-backend mediapipe --hand-backend mediapipe`. `hybrid` maps to MediaPipe body/hands with backend fallbacks enabled. Use `--model` to pick `pose_landmark_lite.tflite`, `pose_landmark_full.tflite`, or `pose_landmark_heavy.tflite` when MediaPipe body landmarks are active.

`--body-backend`  
Function: Selects the body landmark backend.  
Accepted values: `yolo`, `mediapipe`.  
Default: resolved from `--landmark-backend`; `mediapipe` unless you select `--landmark-backend yolo`.  
Notes: `yolo` supports multi-person tracking. `mediapipe` can provide richer single-person foot landmarks.
MediaPipe can run through the multi-person runner/export path, but MediaPipe Pose only returns one body per frame; choose YOLO for true multi-person detection.

`--hand-backend`  
Function: Selects the hand landmark backend.  
Accepted values: `onnx`, `mediapipe`.  
Default: resolved from `--landmark-backend`; `mediapipe` unless you select `--landmark-backend yolo`.

`--backend-fallbacks`  
Function: Enables alternate backend fallback when the selected body or hand backend misses a frame.  
Accepted values: flag only.  
Default: disabled unless `--landmark-backend hybrid` is used.

`--hand-model-variant`  
Function: Selects the hand model preset and auto-download target.  
Accepted variants: `low`, `mid`, `high`, `max`.  
Current mapping:  
`low`, `mid` -> YOLO26 hand pose FP16  
`high`, `max` -> YOLO26 hand pose FP32  
Default: `max`

`--hand-model`  
Function: Uses a specific ONNX hand model file instead of the preset downloader.  
Accepted values: path to an ONNX file.

## Model Runtime Settings

`--hand-input-name`  
Function: Sets the ONNX input tensor name for the hand model.  
Accepted values: any valid ONNX input name string.  
Default: `images`

`--body-input-size`  
Function: Sets the YOLO body model image size.  
Accepted range: integer `> 0`.  
Default: `640`

`--yolo-half`  
Function: Requests FP16 body inference on supported CUDA GPUs.  
Accepted values: flag only.  
Default: disabled in `quality`, enabled by `fastest` and `mid`.

`--no-yolo-fuse`  
Function: Disables YOLO Conv+BatchNorm fusion at model startup.  
Accepted values: flag only.  
Default: fusion enabled.

`--no-yolo-warmup`  
Function: Disables the one-time YOLO warmup pass.  
Accepted values: flag only.  
Default: warmup enabled.

`--no-yolo-person-class-filter`  
Function: Disables class filtering during YOLO pose inference.  
Accepted values: flag only.  
Default: class filtering enabled; Kinara asks YOLO for class `0`, the person class used by pose models.

`--hand-input-size`  
Function: Sets the square resize dimension for the hand crop input.  
Accepted range: integer `> 0`.  
Default: `640`

`--processing-width`  
Function: Runs body/person detection on a resized working frame, then scales body landmarks back to the original output frame.  
Accepted range: integer `>= 0`.  
Default: `0`  
Notes: `0` uses the source resolution. Values such as `480` or `720` are useful for fast preview and webcam runs, especially with 1080p or 2K input. Hand crops stay on the original source frame so downscaling body/person inference does not blur the hand detector input. The saved render keeps the original video size.
When enabled, startup logs print the actual source size and body/person inference size, for example `source 1920x1080 -> inference 640x360`.

## Detection Thresholds

`--body-conf-threshold`  
Function: Minimum YOLO body confidence used to keep body landmarks.  
Accepted range: float in `(0, 1]`.  
Default: `0.30`

`--body-iou-threshold`  
Function: YOLO body NMS IoU threshold.  
Accepted range: float in `(0, 1]`.  
Default: `0.45`

`--hand-det-threshold`  
Function: Minimum confidence needed to keep a hand detection candidate.  
Accepted range: float in `(0, 1]`.  
Default: `0.15`

`--hand-kp-threshold`  
Function: Minimum confidence needed to draw hand keypoints and hand skeleton links.  
Accepted range: float in `(0, 1]`.  
Default: `0.20`

## Hand Crop Settings

`--hand-box-min-size`  
Function: Minimum side length of the wrist-centered hand crop box.  
Accepted range: integer `> 0`.  
Default: `160`

`--hand-box-scale`  
Function: Scales the wrist-elbow distance to form the hand crop size.  
Accepted range: float `> 0`.  
Default: `2.0`

## Hand Model Providers

`--provider`  
Function: Adds an ONNX Runtime execution provider in priority order for the hand model.  
Accepted values: ONNX Runtime provider names like `CUDAExecutionProvider` or `CPUExecutionProvider`.  
Usage: repeat the argument to set fallback order.  
Default: `CUDAExecutionProvider`

## Multi-Person Tracking

`--max-people`  
Function: Enables multi-person tracking for a single camera or video source and sets the maximum number of tracked people.  
Accepted range: integer `> 0`.  
Default: `1`

`--identity personN=color1,color2`  
Function: Provides optional clothing color hints to keep person IDs stable when people cross or overlap.  
Accepted values: labels like `person1`, `person2` with one or more color names such as `black`, `orange`, `blue`, `gray`, `silver`, `red`, `green`, `yellow`, `purple`, `pink`, `brown`, `white`.  
Usage: repeat for multiple people.

`--person-box-scale`  
Function: Expands each detected person box before running hand inference so limbs near the box edges are less likely to be cut off.  
Accepted range: float `> 0`.  
Default: `1.15`

`--person-track-hold-frames`  
Function: Keeps a person track alive briefly when detections disappear for a few frames during crossings or occlusions.  
Accepted range: integer `> 0`.  
Default: `10`

`--person-match-threshold`  
Function: Minimum association score when matching a detected person to an existing track.  
Accepted range: float `> 0`.  
Default: `0.15`

`--person-cross-wrist-ratio`  
Function: Hand ownership switch ratio during overlaps. Lower values are stricter and more likely to reject a stolen hand.  
Accepted range: float in `(0, 1]`.  
Default: `0.90`

`--camera-calibration`  
Function: Loads an optional JSON file with per-camera fusion calibration overrides.  
Accepted values: path to a JSON object keyed by source label such as `CAM_0`, `CAM_1`, or another explicit label used in `--source`.  
Notes: Supported numeric fields are currently `depth_sign` and `depth_scale`. Cameras without JSON entries use neutral depth settings.

`--calibration-3d`  
Function: Loads a calibrated camera TOML for real multi-view 3D triangulation.  
Accepted values: path to a calibration `.toml` whose camera names match source labels like `CAM_0` and `CAM_1`.  
Notes: Used only with `--triangulate-3d`.

`--triangulate-3d`  
Function: Enables real calibrated triangulation for fused runs.  
Accepted values: flag only.  
Default: disabled  
Notes: Requires at least two synchronized camera sources and `--calibration-3d`.

`--triangulation-min-cameras`  
Function: Minimum number of camera views required to reconstruct one joint.  
Accepted range: integer `>= 2`.  
Default: `2`

`--triangulation-use-outlier-rejection`  
Function: Uses camera-view dropping when available to reduce bad-view influence.  
Accepted values: flag only.  
Default: disabled

`--triangulation-max-cameras-to-drop`  
Function: Maximum camera views that outlier rejection can drop for one point.  
Accepted range: integer `>= 0`.  
Default: `1`

`--triangulation-reprojection-error`  
Function: Target reprojection error for outlier-rejection triangulation.  
Accepted range: float `> 0`.  
Default: `0.01`

`--triangulation-max-error`  
Function: Drops triangulated joints above this reprojection error.  
Accepted range: float `> 0`.  
Default: disabled

`--triangulation-smoothing-alpha`  
Function: EMA smoothing factor applied after 3D triangulation. Higher values follow raw 3D more closely.  
Accepted range: float in `(0, 1]`.  
Default: `0.65`

`--sync-offset`  
Function: Applies manual per-camera frame offsets before fused processing.  
Accepted values: `LABEL=FRAMES`, for example `CAM_1=3` or `CAM_0=-2`.  
Notes: Positive values skip that camera's leading frames. Negative values shift other cameras forward relative to it.

`--fused-depth-scale`  
Function: Scales the estimated multi-camera depth used in fused 3D joint exports.  
Accepted range: float `> 0`.  
Default: `1.0`

`--single-camera-depth`  
Function: Selects how single-camera exports handle depth.  
Accepted values: `flat`, `mediapipe`.  
Default: `flat`  
Notes: `flat` keeps single-camera output on a stable Z plane. This avoids fake-depth twisting when no calibrated second camera exists. `mediapipe` uses MediaPipe world-landmark relative Z when available, but it is not calibrated real-world depth.

`--yolo-tracker`  
Function: Sets the Ultralytics tracker config used for multi-person tracking.  
Accepted values: tracker config names like `botsort.yaml` or `bytetrack.yaml`.  
Default: `bytetrack.yaml`

`--yolo-device`  
Function: Overrides the Ultralytics device selection.  
Accepted values: values like `0`, `cpu`, `cuda:0`.  
Default: automatic

## Performance Controls

`--body-detect-interval`  
Function: Runs body/person detection every N frames and predicts skipped frames from recent motion.  
Accepted range: integer `> 0`.  
Default: `1`  
Notes: `1` keeps the most stable body tracking. `2` can nearly halve body model calls, but fast motion may split limbs or tracks.

`--hand-detect-interval`  
Function: Runs hand inference every N frames and translates the last hand landmarks on skipped frames.  
Accepted range: integer `> 0`.  
Default: `1`  
Notes: Hands are usually the expensive part because each visible person can require left and right crops.

`--hand-crop-retries`  
Function: Sets how many extra hand crop attempts run after the primary crop.  
Accepted range: integer `>= 0`.  
Default: `3`  
Notes: Lower values are faster. Higher values are more robust when the first crop misses fingers.

`--fps-log-interval`  
Function: Prints render throughput every N seconds.  
Accepted range: float `>= 0`.  
Default: `0`  
Notes: `0` disables logging.

`--no-fps-overlay`  
Function: Disables the FPS tracker drawn into preview and saved output video frames.  
Accepted values: flag only.  
Default: FPS overlay enabled.

## Live UDP Output

`--osc-host`  
Function: Sets the live UDP target host.  
Accepted values: hostname or IP address.  
Default: `127.0.0.1` from `config.LiveUdpDefaults.HOST`

`--osc-port`  
Function: Sets the live UDP target port.  
Accepted range: integer from `1` to `65535`.  
Default: `9000` from `config.LiveUdpDefaults.PORT`

`--osc-enabled`  
Function: Enables live UDP sending.  
Accepted values: flag only.  
Default: disabled
Notes: Live UDP sends nothing unless this flag is present.
Notes: Live UDP now sends the `kinara-live-v2` schema with frame metadata, fused camera view labels, body landmarks, hand landmarks, hand boxes, and joint maps when available.
Notes: Derived joints currently include `Neck`, `Head`, `LeftFoot`, `LeftToeBase`, `RightFoot`, and `RightToeBase`.

## Preview and Video Writer

`--preview-title`  
Function: Sets the preview window title.  
Accepted values: any text string.  
Default: `Pose + Hand Landmarks`

`--fallback-fps`  
Function: FPS used when the video source reports `0` or invalid FPS.  
Accepted range: float `> 0`.  
Default: `30.0`

`--output-fourcc`  
Function: Sets the video writer codec code.  
Accepted values: text string with at least 4 characters. First 4 are used.  
Default: `mp4v`

## Drawing Colors

`--body-line-color`  
Function: Sets body skeleton line color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `255,0,0`

`--body-point-color`  
Function: Sets body landmark point color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `0,255,0`

`--hand-box-color`  
Function: Sets hand crop rectangle color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `80,80,255`

`--hand-line-color`  
Function: Sets hand skeleton line color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `0,255,255`

`--hand-point-color`  
Function: Sets hand landmark point color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `0,165,255`

## Drawing Sizes

`--body-line-thickness`  
Function: Sets body skeleton line thickness.  
Accepted range: integer `> 0`.  
Default: `2`

`--body-point-radius`  
Function: Sets body landmark point radius.  
Accepted range: integer `> 0`.  
Default: `4`

`--hand-box-thickness`  
Function: Sets hand crop rectangle thickness.  
Accepted range: integer `> 0`.  
Default: `1`

`--hand-line-thickness`  
Function: Sets hand skeleton line thickness.  
Accepted range: integer `> 0`.  
Default: `2`

`--hand-point-radius`  
Function: Sets hand landmark point radius.  
Accepted range: integer `> 0`.  
Default: `3`

## Temporal Stability

`--body-smoothing-alpha`  
Function: EMA smoothing factor for body landmarks. Higher values follow fresh detections more closely, lower values make motion steadier.  
Accepted range: float in `(0, 1]`.  
Default: `0.65`

`--hand-smoothing-alpha`  
Function: EMA smoothing factor for hand landmarks. Higher values follow fresh detections more closely, lower values make motion steadier.  
Accepted range: float in `(0, 1]`.  
Default: `0.55`

`--body-hold-frames`  
Function: Number of frames to keep the last valid body landmark before dropping it when detection confidence disappears.  
Accepted range: integer `> 0`.  
Default: `8`

`--hand-hold-frames`  
Function: Number of frames to keep the last valid hand landmark before dropping it when detection confidence disappears.  
Accepted range: integer `> 0`.  
Default: `6`

`--hold-confidence-decay`  
Function: Confidence multiplier applied each frame while a held landmark is being reused. Lower values fade held joints out faster.  
Accepted range: float in `(0, 1]`.  
Default: `0.85`

`--no-body-constraints`  
Function: Disables soft body length constraints.  
Accepted values: flag only.  
Default: constraints enabled.

`--body-length-smoothing-alpha`  
Function: Controls how quickly learned limb lengths adapt over time.  
Accepted range: float in `(0, 1]`.  
Default: `0.15`

`--body-length-correction`  
Function: Controls how strongly each frame is pulled toward learned limb lengths.  
Accepted range: float in `(0, 1]`.  
Default: `0.35`

`--no-export-cleanup`  
Function: Disables offline export interpolation, spike cleanup, smoothing, and foot lock.  
Accepted values: flag only.  
Default: cleanup enabled.

`--export-cleanup-smoothing-alpha`  
Function: EMA factor used while smoothing exported JSON/FBX motion.  
Accepted range: float in `(0, 1]`.  
Default: `0.55`

`--export-cleanup-max-velocity`  
Function: Maximum per-frame joint movement before export cleanup treats a point as a spike.  
Accepted range: float `> 0`.  
Default: `220.0`

`--no-foot-lock`  
Function: Disables export-time planted-foot stabilization.  
Accepted values: flag only.  
Default: foot lock enabled.

`--foot-lock-velocity`  
Function: Maximum per-frame foot movement considered planted during export cleanup.  
Accepted range: float `> 0`.  
Default: `8.0`

`--foot-lock-max-lift`  
Function: Maximum distance from the detected floor for export foot locking.  
Accepted range: float `>= 0`.  
Default: `16.0`

## Preview Toggle

`--no-preview`  
Function: Disables the live OpenCV preview window.  
Accepted values: flag only.  
Default: preview enabled
