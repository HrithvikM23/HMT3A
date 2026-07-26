from __future__ import annotations

import copy
import multiprocessing
import multiprocessing.shared_memory
import queue
import time

import cv2
import numpy as np

from camera.capture import VideoCaptureSession
from core.config import PipelineConfig
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from runners.common import export_motion_bundle, print_saved_paths
from utils.body_constraints import BodyKinematicConstraints
from utils.exports import build_joint_map
from utils.fps import FpsMeter, draw_fps_overlay
from utils.logging import log_error, log_info, log_warning
from utils.parallel_sizing import auto_parallel_workers
from utils.preview_stream import PreviewFrameSink
from utils.run_metadata import build_run_metadata, write_run_metadata
from utils.smoothing import LandmarkSmoother


class DummySmoother:
    def smooth_body(self, points):
        return points

    def smooth_hand(self, side, points):
        return points


class DummyOSCSender:
    def send_pose(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _pipeline_worker(
    config: PipelineConfig,
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    shm_names: list[str],
    frame_shape: tuple[int, ...],
    frame_dtype: np.dtype,
):
    import os

    from utils.bootstrap_paths import ensure_local_environment
    from utils.logging import install_safe_stdio

    install_safe_stdio()
    ensure_local_environment()

    config.body_constraints_enabled = False
    config.enable_preview = False
    config.osc_enabled = False

    runner = ONNXPoseHandRunner(config)
    smoother = DummySmoother()
    pipeline = PoseHandPipeline(config, runner, smoother, DummyOSCSender())

    shms = [multiprocessing.shared_memory.SharedMemory(name=n) for n in shm_names]
    arrays = [np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf) for shm in shms]

    try:
        while True:
            msg = input_queue.get()
            if msg is None:
                break
            frame_index, shm_idx = msg
            frame = arrays[shm_idx]

            body_points, hands_by_side = pipeline.detect_pose(frame)

            output_queue.put((frame_index, shm_idx, body_points, hands_by_side))
    except KeyboardInterrupt:
        pass
    finally:
        for shm in shms:
            shm.close()


def run_pipeline_parallel(config: PipelineConfig) -> bool:
    assert config.output_path is not None
    session = VideoCaptureSession(
        config.video_path,
        config.output_path,
        fallback_fps=config.fallback_fps,
        output_fourcc=config.output_fourcc,
    )
    ok, first_frame = session.read()
    if not ok or first_frame is None:
        session.close()
        return False

    frame_shape = first_frame.shape
    frame_dtype = first_frame.dtype
    frame_bytes = int(np.prod(frame_shape) * np.dtype(frame_dtype).itemsize)

    total_frames = int(session.source.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    workers = resolve_parallel_workers(config, total_frames=total_frames, fps=session.fps)
    log_info(f"Starting pipeline-parallel runner with {workers} workers")

    ring_size = max(4, workers * 2)
    shms = []
    processes = []
    osc_sender = None
    input_queue = multiprocessing.Queue(maxsize=ring_size)
    output_queue = multiprocessing.Queue()

    try:
        shm_names = []
        arrays = []
        for _ in range(ring_size):
            shm = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_bytes)
            shms.append(shm)
            shm_names.append(shm.name)
            arrays.append(np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf))

        for _ in range(workers):
            p = multiprocessing.Process(
                target=_pipeline_worker,
                args=(copy.copy(config), input_queue, output_queue, shm_names, frame_shape, frame_dtype),
            )
            p.start()
            processes.append(p)

        smoother = LandmarkSmoother(config)
        body_constraints = BodyKinematicConstraints(config)
        osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
        pipeline = PoseHandPipeline(config, None, smoother, osc_sender)

        motion_frames = []
        fps_meter = FpsMeter("pipeline-parallel", config.fps_log_interval)
        preview_sink = PreviewFrameSink()
        started_at = time.perf_counter()

        available_shms = list(range(1, ring_size))
        in_flight = 1
        eof_reached = False
        next_output_index = 0
        pending_results = {}
        capture_frame_index = 1

        # Send first frame
        arrays[0][:] = first_frame[:]
        input_queue.put((0, 0))

        while not eof_reached or in_flight > 0:
            while not eof_reached and len(available_shms) > 0 and not input_queue.full():
                if config.benchmark_frames and capture_frame_index >= config.benchmark_frames:
                    eof_reached = True
                    break

                ok, frame = session.read()
                if not ok or frame is None:
                    eof_reached = True
                    break

                shm_idx = available_shms.pop(0)
                arrays[shm_idx][:] = frame[:]
                input_queue.put((capture_frame_index, shm_idx))
                in_flight += 1
                capture_frame_index += 1

            if in_flight > 0:
                try:
                    timeout = None if len(available_shms) == 0 and not eof_reached else 0.01
                    msg = output_queue.get(timeout=timeout)
                    frame_index, shm_idx, body_points, hands_by_side = msg
                    pending_results[frame_index] = (shm_idx, body_points, hands_by_side)
                except queue.Empty:
                    pass

            while next_output_index in pending_results:
                shm_idx, body_points, hands_by_side = pending_results.pop(next_output_index)
                frame = arrays[shm_idx]

                body_points = smoother.smooth_body(body_points)
                if body_points is not None:
                    body_points = body_constraints.apply(body_points)
                for side, hand_payload in hands_by_side.items():
                    if hand_payload.get("points") is not None:
                        hand_payload["points"] = smoother.smooth_hand(side, hand_payload["points"])

                pipeline.render_pose(frame, body_points, hands_by_side, send_osc=False)

                joint_depths = pipeline.last_joint_depths if config.single_camera_depth_mode == "mediapipe" else None
                joints = build_joint_map(body_points, hands_by_side, joint_depths=joint_depths)
                osc_sender.send_pose(
                    body_points,
                    hands_by_side,
                    joints=joints,
                    metadata={
                        "frame_index": next_output_index,
                        "mode": "pipeline-parallel",
                        "profile": config.profile,
                        "body_backend": config.body_backend,
                        "hand_backend": config.hand_backend,
                        "mediapipe_pose_model": config.mediapipe_pose_model,
                        "source": str(config.video_path),
                    },
                )
                motion_frames.append({"frame_index": next_output_index, "people": [{"id": 1, "joints": joints}]})

                fps_meter.tick(next_output_index)
                draw_fps_overlay(frame, fps_meter, config.fps_overlay_enabled)
                preview_sink.write(frame, next_output_index)
                if hasattr(session, "writer") and session.writer:
                    session.writer.write(frame)

                available_shms.append(shm_idx)
                in_flight -= 1
                next_output_index += 1

        export_metadata = {
            "mediapipe_pose_model": config.mediapipe_pose_model,
            "source": str(config.video_path),
            "body_model_variant": config.body_model_variant,
            "hand_model_variant": config.hand_model_variant,
        }
        export_motion_bundle(
            config,
            fps=session.fps,
            frames=motion_frames,
            metadata=export_metadata,
        )
        write_run_metadata(
            config.metadata_output_path,
            build_run_metadata(
                config,
                mode="pipeline-parallel",
                fps=session.fps,
                frame_count=len(motion_frames),
                extra=export_metadata,
            ),
        )
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        log_info(f"Processed {len(motion_frames)} frames in {elapsed:.2f}s ({len(motion_frames) / elapsed:.2f} FPS)")
        print_saved_paths(config.output_path, config.json_output_path, config.fbx_output_path, config.metadata_output_path)

    except KeyboardInterrupt:
        log_warning("Interrupted by user, shutting down...")
    finally:
        for _ in range(workers):
            try:
                input_queue.put(None, block=False)
            except queue.Full:
                pass
        for p in processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        for shm in shms:
            shm.close()
            shm.unlink()
        session.close()
        if osc_sender is not None:
            osc_sender.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return True
