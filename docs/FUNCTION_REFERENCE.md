# Kinara Function Reference

## How To Read This File

This reference is organized by module. Each bullet explains:

- what the symbol is responsible for
- where it is used
- what it calls or returns when that relationship matters

Private helpers are included because much of Kinara's logic is intentionally lightweight and function-driven rather than hidden behind large framework abstractions.

For end-to-end execution order and module interaction, pair this file with [Architecture And Data Flow](./ARCHITECTURE_AND_DATA_FLOW.md) and [Detailed Runtime Walkthrough](./DETAILED_RUNTIME_WALKTHROUGH.md).

## `main.py`

### Types

- `InputAssignment`: Immutable mapping of a logical camera label such as `FRONT` or `LEFT` to either a webcam index or a file path.

### CLI And Source Resolution

- `parse_color`: Parses `B,G,R` CLI strings into OpenCV color tuples; used by `build_parser`.
- `parse_identity_hint`: Parses `--identity person1=black,orange` into a normalized label and tuple of hint colors.
- `prepare_model_assets`: Resolves and downloads body and hand models before inference starts.
- `choose_video_gui`: Opens the Tk file picker used by interactive video selection.
- `sanitize_label`: Normalizes labels for filenames and source resolution.
- `resolve_output_basename`: Builds the base export stem for a single assignment or one labeled stream in a multi-input session.
- `resolve_output_path`: Adjusts an explicit `--output` path for per-camera naming when multiple labeled sources are active.
- `resolve_fused_output_basename`: Builds the fused output basename, usually from the `FRONT` source.
- `resolve_fused_output_path`: Converts an explicit output path into a fused-output name.
- `choose_camera_assignments_gui`: Interactive helper that asks for camera count and assigns files to logical view labels.
- `resolve_sources`: Converts CLI or interactive input into a list of `InputAssignment` objects.
- `build_parser`: Declares the full CLI surface, including model, tracking, rendering, output, and live UDP options.

### Validation And Configuration

- `validate_config`: Performs runtime sanity checks on the resolved `PipelineConfig`.
- `_prepare_runtime_config`: Calls `prepare_model_assets` and `validate_config`; used at the start of each runtime mode.
- `_build_pipeline_config`: Converts parsed CLI arguments into a fully populated `PipelineConfig`.
- `build_config_for_assignment`: Creates the config for a single labeled assignment.
- `build_fused_config`: Creates the config for a fused multi-camera session.

### Export Helpers

- `export_motion_bundle`: Normalizes single-person frames once and writes both JSON and FBX outputs.
- `_build_fused_metadata`: Builds export metadata for fused modes, including camera labels and calibration path.
- `_print_saved_paths`: Prints output paths after a run completes.

### Multi-Camera Identity Alignment

- `_track_sort_key`: Sort key that prioritizes labeled tracks and then left-to-right ordering.
- `_person_key`: Chooses a stable person key from a track label, tracker id, or fallback index.
- `_align_people_across_cameras`: Groups per-camera `PersonTrack` objects into fused people by label first and color similarity second.

### Payload And Rendering Helpers

- `_build_person_payload`: Creates the exported and OSC-ready person payload, including `body_points`, `hands_by_side`, and `joints`.
- `_draw_person_overlay`: Draws person bounding boxes and labels on rendered frames.
- `_box_from_body_points`: Rebuilds a bounding box from confident body points when a fused person no longer has a detector box.
- `_fuse_smoothed_hands`: Fuses hand views and runs them through the renderer's smoother for temporal consistency.

### Runtime Modes

- `run_multi_person_assignment`: Single-camera multi-person control loop; uses `MultiPersonTracker`, renders per-track overlays, sends live UDP, and writes multi-person JSON and FBX bundle outputs.
- `run_assignment`: Single-camera single-person control loop; uses `PoseHandPipeline`, builds a per-frame `JointMap`, sends live UDP, and writes rendered/JSON/FBX outputs.
- `run_fused_assignments`: Multi-camera control loop; reads synchronized sources, performs either single-person or multi-person fusion, estimates pseudo-depth, renders the fused canvas, and exports fused outputs.
- `main`: Top-level entrypoint; parses CLI args, resolves sources, selects the correct runtime mode, and starts the session.

## `config.py`

### Types

- `LiveUdpDefaults`: Central default host, port, and enabled flag for live UDP output.
- `PipelineConfig`: Main runtime configuration object shared across the entire pipeline.

### `PipelineConfig` Behavior

- `PipelineConfig.__post_init__`: Resolves paths, creates the output directory, determines the next stacked run index, and computes sibling output filenames for video, JSON, and FBX.
- `PipelineConfig._next_run_index`: Finds the first unused numeric suffix so new runs do not overwrite previous outputs.

### Constants

- `BODY_EDGES`: Body skeleton drawing edges for 17-keypoint pose data.
- `BODY_KEYPOINTS`: Body indices that should be drawn as landmark dots.
- `HAND_EDGES`: Hand skeleton drawing edges for 21-point hand data.
- `WRIST_TO_ELBOW`: Maps wrist keypoint indices to their matching elbow indices for left and right hand crop generation.

## `camera/capture.py`

### Classes

- `VideoInputSource`: Thin OpenCV `VideoCapture` wrapper that opens webcam or file input, exposes source dimensions and FPS, and provides `read` and `close`.
- `VideoOutputWriter`: Thin OpenCV `VideoWriter` wrapper that opens the output file and exposes `write` and `close`.
- `VideoCaptureSession`: Convenience wrapper that owns both `VideoInputSource` and `VideoOutputWriter` for single-input runs.

### Methods

- `VideoInputSource.__init__`: Opens the source and determines frame size and FPS, falling back to the configured FPS when needed.
- `VideoInputSource.read`: Reads one frame from the source.
- `VideoInputSource.close`: Releases the source capture.
- `VideoOutputWriter.__init__`: Opens the output writer with the requested FourCC and validated frame dimensions.
- `VideoOutputWriter.write`: Writes one rendered frame.
- `VideoOutputWriter.close`: Releases the writer.
- `VideoCaptureSession.__init__`: Builds a linked source-and-writer pair for the same session.
- `VideoCaptureSession.read`: Proxies `VideoInputSource.read`.
- `VideoCaptureSession.write`: Proxies `VideoOutputWriter.write`.
- `VideoCaptureSession.close`: Closes both capture and writer and destroys OpenCV windows.

## `inference/rtmpose.py`

### Types

- `BodyDetection`: Typed dictionary holding a detector id, confidence, box, and body landmarks for one person.

### Helpers

- `_normalize_provider_name`: Strips non-printable characters from provider names, mainly for Windows shell edge cases.
- `_resolve_provider_names`: Intersects requested ONNX providers with the providers actually available in the current runtime.

### `ONNXPoseHandRunner`

- `ONNXPoseHandRunner.__init__`: Loads the YOLO body model and ONNX hand session from the resolved config.
- `ONNXPoseHandRunner.detect_body`: Convenience wrapper that returns the top detected body or an all-zero 17-point fallback.
- `ONNXPoseHandRunner._to_numpy`: Converts PyTorch-like or NumPy-like tensor outputs into concrete NumPy arrays.
- `ONNXPoseHandRunner.detect_bodies`: Runs YOLO prediction or tracker mode, converts model outputs into `BodyDetection` records, sorts by confidence, and caps the result count.
- `ONNXPoseHandRunner.detect_hand`: Crops the hand region, normalizes it into the ONNX model input format, runs the hand model, and converts the winning detection row into 21 image-space points.

## `pipeline/pipeline.py`

### `PoseHandPipeline`

- `PoseHandPipeline.__init__`: Stores shared references to config, runner, smoother, and OSC sender.
- `PoseHandPipeline.process_frame`: Convenience wrapper that detects and renders pose for one frame and returns the mutated frame.
- `PoseHandPipeline.detect_pose`: Runs body detection, smooths body points, and then calls `detect_hands`.
- `PoseHandPipeline.detect_hands`: Builds wrist-centered crops, runs hand inference, validates results, smooths hands, anchors them to wrists, enforces constraints, and falls back to generated default hands when needed.
- `PoseHandPipeline.render_pose`: Draws body and hands on the frame and optionally sends live UDP.
- `PoseHandPipeline._draw_body`: Draws body skeleton edges and confident body keypoints.
- `PoseHandPipeline._draw_hands`: Draws hand boxes, hand skeleton edges, and confident hand keypoints.

## `network/osc_sender.py`

### `OSCSender`

- `OSCSender.__init__`: Creates a UDP socket only when live output is enabled.
- `OSCSender.send_pose`: Builds and sends a single-person live payload.
- `OSCSender.send_people`: Builds and sends a multi-person live payload.
- `OSCSender._build_person_payload`: Normalizes one person's body and hand landmarks into the live packet schema.
- `OSCSender.close`: Closes the UDP socket.

## `utils/bootstrap_dependencies.py`

### Types

- `ModuleStatus`: Import-check result for one required Python module.
- `RuntimeReport`: Captures detected CUDA/cuDNN paths, include directories, path updates, and warnings.
- `TerminalProgress`: Small progress-bar helper used by the bootstrap CLI and startup path.

### `TerminalProgress` Methods

- `TerminalProgress.__init__`: Stores the number of steps and bar width.
- `TerminalProgress._render`: Draws the current progress line.
- `TerminalProgress.note`: Writes a non-advancing status note.
- `TerminalProgress.advance`: Advances the progress counter and redraws the progress line.
- `TerminalProgress.break_line`: Ends the current progress line cleanly.
- `TerminalProgress.finish`: Marks the bootstrap as complete and ends the progress display.

### Environment And Path Helpers

- `_dedupe_paths`: Removes duplicate filesystem paths while preserving order.
- `_prepend_sys_path`: Adds a path to `sys.path` and registers it with `site`.
- `_prepend_env_path`: Adds a path to the process `PATH` if it is not already present.
- `_register_windows_dll_directory`: Registers a directory with `os.add_dll_directory` so Windows DLL loading can find CUDA and cuDNN.
- `_prepend_pythonpath`: Adds a path to the process `PYTHONPATH`.
- `_broadcast_environment_change`: Broadcasts a Windows environment-change message after persistent updates.
- `_persist_user_path`: Writes CUDA/cuDNN path additions into the current user's environment on Windows.

### Dependency Inspection Helpers

- `_distribution_installed`: Checks whether a Python distribution is installed.
- `_module_status`: Attempts to import one module and records success or failure.
- `_module_group_status`: Runs `_module_status` across a tuple of module names.
- `_find_missing_project_files`: Verifies that critical project files exist before runtime starts.
- `_find_nvidia_smi`: Locates `nvidia-smi` via `PATH` or the standard Windows install location.
- `_path_has_glob`: Checks whether a directory contains files matching a glob pattern.
- `_collect_runtime_roots`: Builds the search list for CUDA and cuDNN discovery from environment variables and standard install locations.
- `_collect_torch_runtime_roots`: Adds `torch/lib` locations because some GPU-capable installs carry runtime DLLs there.
- `_bin_candidates`: Expands a runtime root into likely DLL directories.
- `_include_candidates`: Expands a runtime root into likely include directories.
- `_inspect_runtime`: Scans for NVIDIA driver presence, CUDA DLLs, cuDNN DLLs, and matching headers, then returns a `RuntimeReport`.
- `_repair_runtime_paths`: Applies discovered DLL directories to `PATH`, `CUDA_PATH`, `CUDNN_PATH`, and Windows DLL registration.

### Installation And Verification Helpers

- `_ensure_local_environment`: Creates local runtime directories and seeds `PYTHONPATH`, `sys.path`, and `YOLO_CONFIG_DIR`.
- `_choose_onnxruntime_distribution`: Chooses `onnxruntime-gpu` or `onnxruntime` based on what is installed and whether CUDA/cuDNN appear usable.
- `_resolve_install_plan`: Turns missing imports into an install plan for the local vendor directory.
- `_ensure_pip`: Ensures `pip` exists in the current Python environment.
- `_install_packages`: Installs missing dependencies into the repo-local vendor directory.
- `_probe_runtime`: Imports required modules, inspects ONNX providers, and returns module statuses plus warnings.
- `_dedupe_warning_messages`: Removes duplicate warning strings before printing them.

### Public Bootstrap API

- `ensure_runtime_ready`: Main bootstrap entrypoint; prepares directories, inspects GPU runtime state, installs missing packages, re-verifies imports, and prints warnings.
- `_build_cli_parser`: Builds the CLI parser for standalone bootstrap usage.
- `main`: Standalone bootstrap CLI entrypoint for `utils/bootstrap_dependencies.py`.

## `utils/normalize.py`

- `build_hand_box`: Builds a wrist-centered square crop using the wrist-to-elbow direction, a configurable scale, a minimum size, and a forward shift.

## `utils/smoothing.py`

### `LandmarkSmoother`

- `LandmarkSmoother.__init__`: Initializes body and per-hand smoothing state and missing-point counters.
- `LandmarkSmoother.smooth_body`: Smooths body points using `_smooth_points` and stores body state.
- `LandmarkSmoother.smooth_hand`: Smooths one hand side using `_smooth_points` and stores per-side state.
- `LandmarkSmoother._smooth_points`: Shared smoothing routine that applies exponential averaging to confident points and short-term hold with confidence decay to temporarily missing points.

## `utils/hand_fallback.py`

### Helpers

- `_normalize`: Returns a unit vector for 2D hand template orientation.
- `_point_xy`: Converts a point into a float XY pair.

### Public Hand Utilities

- `anchor_hand_to_wrist`: Translates detected hand points so the hand wrist aligns exactly with the tracked body wrist.
- `is_hand_detection_valid`: Rejects hand outputs that are too far from the wrist, too sparse, too large relative to the forearm, or implausible in palm scale.
- `generate_default_hand`: Builds a synthetic 21-point hand from a wrist-elbow direction vector and a template scaled by forearm length.

## `utils/hand_constraints.py`

### Geometry Helpers

- `_distance`: Euclidean distance helper.
- `_normalize`: Normalizes a 2D vector.
- `_rotate`: Rotates a 2D vector by a given angle.
- `_signed_angle`: Computes the signed angle between two normalized vectors.
- `_point_xy`: Converts a point to float XY.
- `_make_point`: Converts float geometry back to the integer point contract.
- `_base_hand_scale`: Estimates a palm-scale baseline from wrist to finger-root distances.
- `_clamp_distance`: Constrains a target point to a minimum and maximum radius from an origin.

### Constraint Passes

- `_enforce_radial_limits`: Prevents any point from drifting too far from the wrist.
- `_enforce_bone_lengths`: Constrains segment lengths between parent and child landmarks.
- `_enforce_chain_bend`: Limits per-joint bending for thumb and finger chains.
- `_project_local`: Projects a point into a hand-local lateral/forward basis.
- `_unproject_local`: Converts hand-local coordinates back into image-space coordinates.
- `_enforce_finger_lanes`: Prevents fingers from crossing into each other's lateral bands.
- `enforce_hand_constraints`: Orchestrates the full hand-cleanup pass by applying radial, length, bend, and lane constraints.

## `utils/model_assets.py`

### Types

- `ModelSpec`: Describes where a model should be downloaded, stored, and how the hand model input should be configured.

### Model Asset Functions

- `_download_to_path`: Downloads a model atomically through a `.part` temporary file.
- `ensure_model_file`: Ensures a model described by `ModelSpec` exists in the project tree.
- `ensure_body_model_file`: Resolves a body model path or downloads a known YOLO pose model into `models/body`.

## `utils/multi_person.py`

### Types

- `PersonDetection`: Per-frame detector result used before association into tracks.
- `PersonTrack`: Persistent track state containing detector id, pipeline, motion state, label, color signature, and smoothing history.

### Matching And Color Helpers

- `_iou`: Intersection-over-union score for boxes.
- `_center_distance_score`: Closeness score based on box centers.
- `_size_similarity_score`: Similarity score based on relative box area.
- `_non_max_suppress`: Basic NMS helper retained for box filtering patterns.
- `_expand_box`: Expands a detector box before hand inference.
- `_torso_crop`: Crops the torso area used for clothing-color analysis.
- `_color_scores`: Computes simple HSV color occupancy scores over a torso crop.
- `_blend_color_scores`: Smooths clothing-color signatures over time.
- `color_profile_similarity`: Computes similarity between two color signatures.
- `_translate_body_points`: Utility to shift body points by a 2D offset.
- `_translate_hands`: Utility to shift hand boxes and points by a 2D offset.

### `MultiPersonTracker`

- `MultiPersonTracker.__init__`: Stores config and runner and initializes track state.
- `MultiPersonTracker.update`: High-level per-frame tracker update; detects people, associates tracks, resolves hand ownership, and returns active tracks.
- `MultiPersonTracker._detect_people`: Runs body detection in tracking mode and converts raw detections into `PersonDetection` objects with expanded boxes and color scores.
- `MultiPersonTracker._associate_tracks`: Matches current detections to previous tracks, updates matched tracks, carries forward briefly missing tracks, and creates new tracks.
- `MultiPersonTracker._create_track`: Builds a new `PersonTrack` with its own `PoseHandPipeline`, `LandmarkSmoother`, and disabled OSC sender.
- `MultiPersonTracker._match_score`: Scores one detection against one track using tracker-id continuity, motion prediction, box overlap, size, and color similarity.
- `MultiPersonTracker._best_identity_label`: Chooses the best unassigned identity label from clothing hints.
- `MultiPersonTracker._identity_score`: Computes how well a detection matches one configured identity-hint color profile.
- `MultiPersonTracker._update_track_from_detection`: Updates a matched track's velocity, box, body points, hands, label, and color signature.
- `MultiPersonTracker._enforce_unique_labels`: Ensures at most one active track holds each identity label.
- `MultiPersonTracker._predict_track_box`: Predicts the next box using simple per-track velocity.
- `MultiPersonTracker._refresh_track_label`: Revisits label assignment after the latest detection and color evidence.
- `MultiPersonTracker._hand_owner_score`: Scores how likely one detected hand belongs to one track based on wrist distance, elbow distance, box overlap, and whether the hand sits inside the body box.
- `MultiPersonTracker._resolve_cross_person_hands`: Replaces hands that appear to belong to a different active track with a generated fallback hand, reducing hand stealing during crossings.

## `utils/fusion.py`

### Types

- `FrameReference`: Reference frame with origin and scale used to normalize coordinates between camera views.

### Calibration And Reference Functions

- `load_camera_calibrations`: Loads optional per-camera depth overrides from JSON and merges them with defaults.
- `compute_body_reference`: Builds a body reference frame from confident torso points.
- `compute_hand_reference`: Builds a hand reference frame from a hand payload.
- `project_points`: Reprojects points from one camera reference frame into another.
- `_project_box`: Reprojects a box via projected corner points.
- `_choose_reference`: Chooses a preferred reference payload, falling back to the first viable candidate.
- `_prepare_body_sources`: Filters and annotates body-camera sources that are usable for fusion.
- `_prepare_hand_sources`: Filters and annotates hand-camera sources that are usable for fusion.

### Fusion Functions

- `fuse_body_views`: Projects all body views into a common reference and fuses each point with confidence-weighted averaging.
- `fuse_hand_views`: Projects all hand views into a common reference and fuses each hand point with confidence-weighted averaging.

### Depth Estimation Functions

- `_estimate_body_joint_depths`: Estimates pseudo-depth for named body joints from lateral displacement across calibrated camera views.
- `_estimate_hand_joint_depths`: Estimates pseudo-depth for named hand joints from lateral displacement across calibrated camera views.
- `estimate_joint_depths`: Merges body and hand depth estimates and ensures root defaults exist.

## `utils/exports.py`

### Types

- `JointSpec`: Named skeleton joint plus optional parent joint.

### Coordinate And Averaging Helpers

- `_to_world`: Converts image-space integer coordinates into export-space coordinates.
- `_to_world_float`: Float-preserving version of `_to_world`.
- `_average_points`: Averages a list of points and returns world-space position plus average confidence.
- `_average_screen_points`: Averages point positions in screen space.
- `_make_joint`: Normalizes raw coordinates into a `JointValue`.
- `_zero_joint`: Returns a zeroed joint record.
- `_copy_joint_value`: Returns a detached copy of a joint record.
- `_lerp`: Linear interpolation helper used in derived joint construction.

### Derived Joint Construction

- `_derive_head_joints`: Derives `Neck` and `Head` either from face points or from torso direction when face points are unavailable.
- `_derive_foot_chain`: Derives `Foot` and `ToeBase` from knee-to-ankle direction.
- `build_joint_map`: Core body-and-hand-to-skeleton conversion used by exports, live UDP, fused depth, and Blender JSON generation.

### Frame Normalization And Metadata

- `_localize_joint_map`: Converts world-space joints into parent-relative local transforms.
- `_frame_joint_map`: Extracts the typed joint map from an export frame.
- `_joint_is_valid`: Checks whether a joint is confident and non-zero.
- `_ground_joint_frames_on_axis`: Grounds all confident joints to a shared minimum on one axis.
- `_ground_joint_frames`: Grounds joint frames along the original image-up axis before coordinate conversion.
- `_z_up_joint_frames`: Swaps normalized axes so export-space becomes Z-up.
- `_ground_z_axis_frames`: Grounds the final export on the Z axis.
- `_normalize_export_frames`: Standard normalization pass for single-person frames.
- `_build_skeleton_metadata`: Serializes the Kinara skeleton hierarchy into metadata.
- `_compute_rest_joints`: Builds stable per-joint rest positions from median segment offsets across a clip.
- `_collect_multi_person_joint_tracks`: Expands multi-person frame payloads into per-person joint-track sequences.
- `_normalize_multi_person_frames`: Applies single-person normalization independently to each person track and rebuilds multi-person frames.
- `_build_export_metadata`: Enriches output metadata with skeleton, coordinate system, and rest-joint information.

### JSON Export

- `_write_motion_json`: Writes the common JSON envelope shared by single-person and multi-person exports.
- `export_motion_json`: Writes `kinara-motion-json-v1` after optional normalization.
- `export_multi_person_json`: Writes `kinara-multi-person-json-v1` after optional normalization.

### Multi-Person Export Helpers

- `_sanitize_person_label`: Converts person labels into filesystem-safe suffixes.
- `_empty_joint_map`: Builds an all-zero joint map.
- `_coerce_frame_index`: Converts frame indices into integers.
- `export_multi_person_fbx_bundle`: Splits multi-person frames into one per-person FBX export file.

### BVH And FBX Export

- `_compute_offsets`: Builds the initial local joint offsets used by BVH hierarchy generation.
- `_build_bvh_hierarchy_lines`: Recursively serializes the Kinara skeleton hierarchy into BVH text.
- `export_motion_bvh`: Writes BVH motion output using localized joint translations.
- `_fbx_template_header`: Returns the shared FBX header scaffold.
- `export_motion_fbx`: Writes a lightweight FBX that stores animation as per-joint local translation curves.

## `blender_kinematics/kinara_motion.py`

### Types

- `SkeletonJoint`: Blender-side skeleton node with a name and parent.
- `PersonMotion`: One person's frame indices, frame list, and rest-joint data.
- `MotionClip`: Parsed Kinara clip containing source path, format, FPS, skeleton, coordinate metadata, and people.

### Parsing And Validation Helpers

- `sanitize_label`: Converts labels into stable import-safe names.
- `empty_joint_map`: Creates an all-zero joint map for a supplied skeleton.
- `joint_is_valid`: Checks whether a parsed joint is confident and non-zero enough to be trusted.
- `_coerce_joint_value`: Converts arbitrary JSON-like input into a normalized joint value.
- `_coerce_joint_map`: Converts arbitrary JSON-like input into a full skeleton-shaped joint map.
- `_normalize_frame_index`: Safely converts frame indices from JSON values into integers.
- `_extract_skeleton`: Loads a skeleton from JSON metadata or falls back to Kinara's default skeleton.

### Rest-Pose Helpers

- `_default_bone_direction`: Supplies fallback bone directions when rest-joint inference has too little data.
- `_scaled_direction`: Normalizes and scales a fallback direction to the importer's minimum bone length.
- `compute_rest_joints`: Builds a stable rest skeleton from median per-segment offsets across the clip.
- `build_held_joint_frames`: Reuses the last valid joint sample when a frame is weak or missing.

### Clip Loading

- `load_motion_clip`: Reads Kinara JSON, resolves skeleton and coordinate metadata, separates single-person or multi-person tracks, and returns a `MotionClip` ready for Blender import.

## `blender_kinematics/import_kinara_motion.py`

### Import Helpers

- `_vector_from_joint`: Converts Kinara joint dictionaries into Blender `Vector` values.
- `_ensure_collection`: Creates or reuses a Blender collection for an import session.
- `_set_scene_fps`: Sets Blender scene FPS from clip metadata.
- `_clear_object_selection`: Clears current Blender object selection.
- `_create_armature_object`: Creates a new Blender armature object and links it into the import collection.
- `_root_tail`: Chooses a stable root-bone tail direction from the rest pose.
- `_fallback_tail`: Provides a minimum valid tail direction when a rest segment is degenerate.

### Rig Construction

- `_build_armature_rig`: Builds the `KinaraRoot` plus one edit bone per non-root joint, parents them, and switches pose bones to quaternion rotation.
- `_create_joint_targets`: Creates hidden empties for each joint and keys their locations across all frames.
- `_apply_constraints`: Adds root-copy and bone-track constraints so the armature follows keyed joint targets.
- `_create_action`: Creates the Blender `Action` that will receive baked animation.
- `_bake_animation`: Bakes constrained motion into pose keys and removes the live constraints.
- `_delete_targets`: Removes the temporary hidden target empties after baking.

### Public Import API

- `import_motion_clip`: Imports every requested person from a `MotionClip` into Blender, building one armature and one action per person.
- `_parse_cli_args`: Parses `--input` and `--person` for Blender background or script execution.
- `main`: Blender script entrypoint; loads a Kinara clip, imports it, and prints a success message.

## `tests/test_exports_metadata.py`

- `_CapturePath`: In-memory fake path-like object used to capture JSON output without writing files.
- `MotionExportMetadataTests.test_single_person_export_includes_blender_metadata`: Guards the single-person JSON metadata contract.
- `MotionExportMetadataTests.test_multi_person_export_includes_per_person_rest_joints`: Guards the multi-person metadata contract.

## `tests/test_kinara_motion.py`

- `KinaraMotionTests.test_compute_rest_joints_uses_stable_median_offsets`: Verifies rest-joint inference from stable median segment offsets.
- `KinaraMotionTests.test_held_frames_reuse_last_valid_joint`: Verifies that missing frames reuse the last valid joint data during Blender import preparation.
