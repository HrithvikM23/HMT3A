# Kinara Detailed Runtime Walkthrough

## Purpose

This file explains the project as if you were tracing it during execution. The architecture doc tells you the shape of the system. This file tells you what actually happens, in order, when the program runs.

## 1. Process Startup

When Python starts `main.py`, the first meaningful thing that happens is:

```python
from utils.bootstrap_dependencies import ensure_runtime_ready
ensure_runtime_ready()
```

That means the application does not wait until later to discover runtime problems. It tries to repair the environment before doing anything else.

### Why that matters

Kinara depends on a stack that is easy to misconfigure on Windows:

- OpenCV
- NumPy
- Ultralytics
- torch / torchvision
- ONNX Runtime
- CUDA DLLs
- cuDNN DLLs

On many laptops, all of those pieces are technically installed somewhere, but Python still cannot see the GPU runtime. The bootstrap module exists to fix that mismatch early.

## 2. Bootstrap Timeline

`ensure_runtime_ready()` conceptually does this:

1. Confirm the repo files exist.
2. Create local vendor/runtime folders if needed.
3. Add the local vendor folder to Python import paths.
4. Scan the machine for GPU runtime locations.
5. Add discovered CUDA/cuDNN folders to the process environment.
6. Register DLL directories on Windows.
7. Check whether Python packages import correctly.
8. Install anything missing into the local vendor folder.
9. Probe ONNX Runtime providers and print warnings if GPU acceleration still is not visible.

### Important side effect

This startup layer makes the repo much more portable because the project can carry its own Python dependencies in `.vendor_py311` instead of requiring a perfectly curated system-wide Python install.

## 3. CLI Resolution

After bootstrap, `main()` builds the CLI parser and parses args.

Then `resolve_sources()` transforms the raw input into a list of `InputAssignment` objects.

Examples:

- `--source 0` becomes `[InputAssignment(label="FRONT", source=0)]`
- `--source video.mp4` becomes `[InputAssignment(label="FRONT", source=Path("video.mp4"))]`
- `--source FRONT=front.mp4 --source LEFT=left.mp4` becomes two labeled assignments

### Why labels matter

Labels are not just cosmetic. They are used by:

- fused-camera logic
- output naming
- camera-group metadata
- reference-view selection

## 4. Config Construction

Next, the runtime converts args into `PipelineConfig`.

This config is the central contract for the whole execution. It decides:

- which model files to use
- what thresholds define a valid point or valid hand
- how aggressive temporal smoothing should be
- whether live UDP is enabled
- whether this run is single-person, multi-person, or fused
- what output files should be created

### Output naming behavior

The config computes stack-safe sibling outputs:

- `<base> rendered-N.mp4`
- `<base> json-N.json`
- `<base> fbx-N.fbx`

This prevents accidental overwrites and makes repeated experimentation safer.

## 5. Runtime Mode Selection

The application then branches:

- one source and `max_people == 1` -> single-person mode
- one source and `max_people > 1` -> single-camera multi-person mode
- multiple sources -> fused multi-camera mode

Each mode reuses the same lower-level pieces, but the orchestration differs.

## 6. Single-Person Mode Walkthrough

## Setup phase

`run_assignment()` performs:

1. `_prepare_runtime_config(config)`
2. `VideoCaptureSession(...)`
3. `ONNXPoseHandRunner(config)`
4. `LandmarkSmoother(config)`
5. `OSCSender(...)`
6. `PoseHandPipeline(config, runner, smoother, osc_sender)`

### Why this object layout is clean

- capture owns file/video IO
- runner owns model execution
- smoother owns temporal memory
- pipeline owns per-frame body+hand logic
- sender owns networking

No single object is asked to do everything.

## Frame loop

For every frame:

1. `session.read()` returns the next frame.
2. `pipeline.detect_pose(frame)` returns body landmarks and hands.
3. `build_joint_map(body_points, hands_by_side)` converts those landmarks into the canonical skeleton.
4. `pipeline.render_pose(...)` draws the overlay.
5. `osc_sender.send_pose(...)` optionally emits a live packet.
6. A motion frame containing `frame_index` and `joints` is appended.
7. `session.write(frame)` saves the rendered frame.

After the loop ends, `export_motion_bundle()` writes JSON and FBX.

## 7. What `detect_pose()` really does

This is the heart of the single-person runtime.

### Step A. Body detection

`runner.detect_body(frame)` runs YOLO pose inference and returns one 17-point body.

If no reliable body is found, the code returns an all-zero 17-point structure. That may sound odd at first, but it keeps the pipeline structurally stable. Downstream consumers do not need to branch on `None` everywhere.

### Step B. Body smoothing

`smoother.smooth_body()` applies exponential smoothing and short-term point holding.

This reduces:

- jitter
- tiny detector oscillations
- one-frame dropouts

### Step C. Hand detection

`detect_hands(frame, body_points)` handles each wrist separately.

For each side:

1. Read wrist and elbow from the body.
2. Require both to exceed the body confidence threshold.
3. Build a wrist-centered crop with `build_hand_box()`.
4. Run the hand model on that crop.
5. Reject implausible hand results with `is_hand_detection_valid()`.
6. Smooth the hand landmarks over time.
7. Anchor the hand wrist to the body wrist.
8. Apply anatomical cleanup with `enforce_hand_constraints()`.
9. Revalidate the corrected hand.
10. If still unreliable, generate a synthetic default hand.

### Why the fallback hand exists

Hand detectors fail much more often than body detectors. Without a fallback, the wrist and finger chains would pop in and out constantly. The synthetic hand keeps the export shape intact and gives downstream tools a stable chain to retarget, even when the hand model briefly loses the fingers.

## 8. Joint Map Construction

Once body and hand points are ready, `build_joint_map()` creates a named skeleton.

### Direct joints

These come straight from body landmarks:

- shoulders
- elbows
- wrists
- hips
- knees
- ankles

### Derived torso/head joints

These are synthesized:

- `HipsRoot`: average of left and right hip
- `Chest`: average of left and right shoulder
- `Neck` and `Head`: derived from face points if available, otherwise extended from torso direction

### Derived foot joints

These are synthesized from knee-to-ankle direction:

- `LeftFoot`
- `LeftToeBase`
- `RightFoot`
- `RightToeBase`

### Hand joints

The 21-point left and right hand detections are mapped into named finger chains such as:

- `LeftThumb1`..`LeftThumb4`
- `LeftIndex1`..`LeftIndex4`
- `RightPinky1`..`RightPinky4`

## 9. Export Timeline

When the frame loop is done, the runtime has a list of raw motion frames. Those frames are not yet ready for Blender or other consumers. The export layer normalizes them first.

### Normalization pipeline

`_normalize_export_frames()` conceptually does:

1. ground the motion timeline
2. convert the coordinate conventions into Z-up export space
3. re-ground the final Z-up result

### Metadata enrichment

The export path adds:

- skeleton hierarchy
- coordinate system declaration
- stable rest-joint positions
- run metadata such as source and mode

### Why rest joints are exported

Blender needs a stable rest rig. If the importer had to guess the rig from whatever the first frame looked like, noisy clips would create warped armatures. Exporting `rest_joints` makes the Blender side far more robust.

## 10. Single-Camera Multi-Person Walkthrough

Single-camera multi-person mode replaces “one pipeline instance” with “one tracker that owns many pipeline instances”.

## Setup phase

`run_multi_person_assignment()` creates:

- one shared capture session
- one shared body/hand runner
- one shared live UDP sender
- one `MultiPersonTracker`

The tracker then creates per-track helpers as needed.

## Frame loop

For every frame:

1. `tracker.update(frame)` runs the whole multi-person logic.
2. Active tracks are returned.
3. Each active track draws its own pose using its own `PoseHandPipeline`.
4. The top-level loop draws a label and box around each person.
5. `_build_person_payload()` builds a per-person export/live structure.
6. `send_people()` emits a multi-person UDP packet.
7. The people list is appended to `motion_frames`.
8. The rendered frame is written.

## 11. What `tracker.update()` really does

### Stage A. Person detection

`_detect_people(frame)` runs YOLO in tracking mode and returns `PersonDetection` objects. Each detection carries:

- expanded person box
- 17-point body
- confidence score
- tracker id if YOLO produced one
- clothing-color signature

### Stage B. Track association

`_associate_tracks()` tries to connect current detections to old tracks.

It first preserves direct tracker-id matches when possible because that is cheap and usually correct.

Then it scores remaining pairs using:

- overlap
- center closeness
- size similarity
- color similarity
- optional identity-hint score

If nothing matches strongly enough, it creates a new `PersonTrack`.

### Stage C. Track-local hand detection

When a track is updated from a detection, the tracker does not run a full body pipeline again. It already has the body points from YOLO. It only runs the track's `PoseHandPipeline.detect_hands(frame, body_points)` to get hands relative to that specific body.

This is a good example of code reuse: the same hand pipeline is used in both single-person and multi-person mode, but the track already supplies the body.

### Stage D. Cross-person hand cleanup

After all tracks are updated, `_resolve_cross_person_hands()` looks for obviously stolen hands. If one track's “left hand” is much more consistent with another person's left wrist and elbow, the current track loses that bad hand and receives a generated fallback instead.

This is not full global optimization, but it solves a common and visually obvious failure mode in a practical way.

## 12. Fused Multi-Camera Walkthrough

Fused mode is easier to understand if you think of it as “run one local vision pass per camera, then merge the results”.

## Setup phase

`run_fused_assignments()` creates:

- one `VideoInputSource` per camera
- one shared model runner
- one output writer based on the reference camera dimensions
- one `PoseHandPipeline` per camera for single-person fused mode
- one `MultiPersonTracker` per camera for fused multi-person mode

It also loads optional calibrations before the loop starts.

## Frame synchronization model

The current fused loop assumes one frame is read from each source per iteration. If any source ends, the loop ends. This is simple and deterministic, which is good for file-based synchronized runs.

## 13. Fused Single-Person Path

For every iteration:

1. Read one frame per camera.
2. Run camera-local `detect_pose()` for each frame.
3. Gather `camera_bodies` and `camera_hands`.
4. Fuse the body across views.
5. Fuse the left and right hands across views.
6. Estimate pseudo-depth from multi-view displacement.
7. Build one fused joint map.
8. Render the result on the reference frame copy.
9. Emit live packet and append export frame.

### Why the reference frame matters

Fusion does not produce a brand new abstract space first and render there. Instead it projects landmarks into one chosen reference view. That makes rendering intuitive, because the final overlay still sits on a real camera frame.

## 14. Fused Multi-Person Path

This mode combines the complexity of multi-person tracking with multi-camera grouping.

For every iteration:

1. Each camera tracker updates independently.
2. The runtime now has per-camera active track lists.
3. `_align_people_across_cameras()` groups those tracks into fused people.
4. For each group:
   - collect body landmarks from all views where that person exists
   - collect hand landmarks from all views where that person exists
   - fuse body and hands
   - estimate depth
   - build a joint map
5. Render each grouped person on the reference canvas.
6. Emit and export the fused multi-person result.

### Grouping strategy

The grouping logic prefers:

1. explicit labels such as `person1`
2. then color-profile similarity against reference-view tracks

That makes fused identity more stable when all cameras do not see the same person equally well.

## 15. Blender Import Walkthrough

The Blender importer is a second stage in the pipeline, not part of capture-time execution.

## Clip loading

`load_motion_clip()` reads the exported JSON and converts it into:

- clip metadata
- skeleton hierarchy
- coordinate system metadata
- one `PersonMotion` per person track

It supports both:

- `kinara-motion-json-v1`
- `kinara-multi-person-json-v1`

## Rest pose preparation

Each person needs `rest_joints`.

If the JSON already contains them, the importer uses them.

If not, `compute_rest_joints()` derives them from the motion frames by taking median parent-child offsets across the clip. Median is important because it resists noise and outlier frames much better than blindly trusting frame 0.

## Held-frame preparation

`build_held_joint_frames()` prepares a stable frame list by copying the last valid joint value forward when the current frame is weak or missing.

This protects the armature against:

- zero-length bones in one frame
- origin snaps
- missing-person gaps in multi-person clips

## Rig construction

The importer creates:

- one collection for the import session
- one armature per person
- one Blender `Action` per person
- a non-deforming `KinaraRoot`
- one edit bone per non-root Kinara joint

### Important rigging decision

The importer does not connect all bones as hard-connected edit bones. It parents them, but leaves them unconnected where appropriate. This matters because Kinara's skeleton branches at shared torso joints and a forced connected chain would distort several branches.

## Temporary targets and bake

The importer then creates hidden empties for each joint and keys their locations for every frame.

The rig is constrained to those targets and then baked into pose keys.

That bake step is crucial because the final result should be:

- a normal Blender armature
- with normal Blender animation data
- without permanent dependence on helper empties or live constraints

## 16. How Data Changes Shape Across The Project

The same motion changes representation several times:

1. Raw frame: OpenCV image
2. Raw body detections: detector boxes plus 17 keypoints
3. Body + hands: image-space landmarks
4. Joint map: named skeleton in normalized export space
5. Motion frame list: timeline of joint maps
6. Motion JSON: serialized timeline plus metadata
7. Blender `MotionClip`: parsed structured import object
8. Blender armature/action: final DCC animation asset

Understanding these shape transitions is the key to understanding the codebase.

## 17. Where To Debug Common Problems

### Problem: GPU falls back to CPU

Look at:

- `utils/bootstrap_dependencies.py`
- requested ONNX providers in the config
- discovered CUDA/cuDNN directories
- whether `torch/lib` was found and registered

### Problem: hands jitter or disappear

Look at:

- `utils/normalize.py`
- `PoseHandPipeline.detect_hands()`
- `utils/smoothing.py`
- `utils/hand_fallback.py`
- `utils/hand_constraints.py`

### Problem: people swap identities

Look at:

- `utils/multi_person.py`
- `_match_score()`
- `_refresh_track_label()`
- `color_profile_similarity()`

### Problem: fused result looks wrong

Look at:

- `utils/fusion.py`
- reference-view selection
- calibration input
- depth-scale setting

### Problem: Blender bones slide or collapse

Look at:

- `utils/exports.py` metadata generation
- `blender_kinematics/kinara_motion.py`
- `blender_kinematics/import_kinara_motion.py`

## 18. Mental Model To Keep

The simplest accurate mental model for Kinara is:

- `main.py` orchestrates
- `config.py` describes the run
- `inference` detects
- `pipeline` stabilizes and renders
- `multi_person` persists identities
- `fusion` merges cameras
- `exports` converts landmarks into a durable skeleton timeline
- `blender_kinematics` turns that timeline into a real armature animation

Once that model is clear, the repo becomes much easier to extend safely.
