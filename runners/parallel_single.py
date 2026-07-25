from __future__ import annotations

import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import PipelineConfig
from runners.common import export_motion_bundle, print_saved_paths
from utils.exports import build_joint_map
from utils.fps import FpsMeter, draw_fps_overlay
from utils.logging import log_info, log_warning
from utils.run_metadata import build_run_metadata, write_run_metadata


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    index: int
    start_frame: int
    end_frame: int
    warmup_start_frame: int
    rendered_path: Path


@dataclass(frozen=True, slots=True)
class ChunkResult:
    index: int
    start_frame: int
    end_frame: int
    rendered_path: Path
    motion_frames: list[dict[str, object]]
    processed_frames: int


def _is_gpu_backend(config: PipelineConfig) -> bool:
    device_text = " ".join(
        str(value or "")
        for value in (
            config.yolo_device,
            config.rtmpose_device,
            *config.provider_names,
        )
    ).lower()
    if "cuda" in device_text or config.yolo_device == "0":
        return True
    return config.body_backend in {"yolo", "rtmpose", "rtmpose-wholebody"} and "cpu" not in device_text


def _auto_parallel_workers(config: PipelineConfig, total_frames: int, fps: float) -> int:
    if total_frames <= 0 or fps <= 0:
        return 1
    duration_seconds = total_frames / fps
    chunk_seconds = resolve_parallel_chunk_seconds(config, total_frames=total_frames, fps=fps)
    if duration_seconds < max(chunk_seconds * 2.0, 8.0):
        return 1

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1

    chunk_count = max(1, math.ceil(duration_seconds / chunk_seconds))
    backend_cap = 2 if _is_gpu_backend(config) else 4
    return max(1, min(cpu_count - 1, backend_cap, chunk_count))


def resolve_parallel_workers(config: PipelineConfig, *, total_frames: int = 0, fps: float = 0.0) -> int:
    if config.parallel_workers == 0:
        return _auto_parallel_workers(config, total_frames, fps)
    return max(1, config.parallel_workers)


def resolve_parallel_chunk_seconds(config: PipelineConfig, *, total_frames: int = 0, fps: float = 0.0) -> float:
    if config.parallel_chunk_seconds == 0:
        if total_frames > 0 and fps > 0 and (total_frames / fps) >= 120:
            return 10.0
        return 5.0
    return max(0.25, config.parallel_chunk_seconds)


def resolve_parallel_overlap_seconds(config: PipelineConfig) -> float:
    if config.parallel_overlap_seconds == 0:
        return 0.5
    return max(0.0, config.parallel_overlap_seconds)


def eligible_for_parallel_single(config: PipelineConfig) -> bool:
    return (
        isinstance(config.video_path, Path)
        and config.max_people == 1
        and not config.enable_preview
        and not config.osc_enabled
    )


def build_chunk_specs(
    config: PipelineConfig,
    *,
    total_frames: int,
    fps: float,
    chunk_root: Path,
) -> list[ChunkSpec]:
    target_frames = total_frames
    if config.benchmark_frames:
        target_frames = min(target_frames, config.benchmark_frames)
    if target_frames <= 0:
        return []

    chunk_seconds = resolve_parallel_chunk_seconds(config, total_frames=target_frames, fps=fps)
    overlap_seconds = resolve_parallel_overlap_seconds(config)
    chunk_frames = max(1, int(round(max(fps, 1.0) * chunk_seconds)))
    overlap_frames = max(0, int(round(max(fps, 1.0) * overlap_seconds)))

    specs: list[ChunkSpec] = []
    start_frame = 0
    while start_frame < target_frames:
        end_frame = min(target_frames, start_frame + chunk_frames)
        specs.append(
            ChunkSpec(
                index=len(specs),
                start_frame=start_frame,
                end_frame=end_frame,
                warmup_start_frame=max(0, start_frame - overlap_frames),
                rendered_path=chunk_root / f"chunk_{len(specs):04d}.mp4",
            )
        )
        start_frame = end_frame
    return specs


def _probe_video(config: PipelineConfig) -> tuple[int, float, int, int]:
    from camera.capture import VideoInputSource

    source = VideoInputSource(config.video_path, fallback_fps=config.fallback_fps)
    try:
        total_frames = int(source.cap.get(7) or 0)
        return total_frames, source.fps, source.frame_width, source.frame_height
    finally:
        source.close()


def _process_chunk(config: PipelineConfig, spec: ChunkSpec) -> ChunkResult:
    from camera.capture import VideoInputSource, VideoOutputWriter
    from inference.rtmpose import ONNXPoseHandRunner
    from network.osc_sender import OSCSender
    from pipeline.pipeline import PoseHandPipeline
    from utils.bootstrap_paths import ensure_local_environment
    from utils.logging import install_safe_stdio
    from utils.smoothing import LandmarkSmoother

    install_safe_stdio()
    ensure_local_environment()
    config.enable_preview = False
    config.osc_enabled = False

    source = VideoInputSource(config.video_path, fallback_fps=config.fallback_fps)
    writer = VideoOutputWriter(
        spec.rendered_path,
        frame_width=source.frame_width,
        frame_height=source.frame_height,
        fps=source.fps,
        output_fourcc=config.output_fourcc,
    )
    runner = ONNXPoseHandRunner(config)
    smoother = LandmarkSmoother(config)
    osc_sender = OSCSender(config.osc_host, config.osc_port, False)
    pipeline = PoseHandPipeline(config, runner, smoother, osc_sender)
    fps_meter = FpsMeter(f"chunk-{spec.index}", 0.0)
    motion_frames: list[dict[str, object]] = []
    processed_frames = 0

    try:
        source.cap.set(1, spec.warmup_start_frame)
        current_frame = spec.warmup_start_frame
        while current_frame < spec.end_frame:
            ok, frame = source.read()
            if not ok or frame is None:
                break

            body_points, hands_by_side = pipeline.detect_pose(frame)
            joint_depths = pipeline.last_joint_depths if config.single_camera_depth_mode == "mediapipe" else None
            joints = build_joint_map(body_points, hands_by_side, joint_depths=joint_depths)
            pipeline.render_pose(frame, body_points, hands_by_side, send_osc=False)

            if current_frame >= spec.start_frame:
                motion_frames.append({"frame_index": current_frame, "joints": joints})
                fps_meter.tick(current_frame)
                draw_fps_overlay(frame, fps_meter, config.fps_overlay_enabled)
                writer.write(frame)
                processed_frames += 1
            current_frame += 1
    finally:
        source.close()
        writer.close()
        osc_sender.close()

    return ChunkResult(
        index=spec.index,
        start_frame=spec.start_frame,
        end_frame=spec.end_frame,
        rendered_path=spec.rendered_path,
        motion_frames=motion_frames,
        processed_frames=processed_frames,
    )


def _concatenate_rendered_chunks(chunk_paths: list[Path], output_path: Path, output_fourcc: str, fallback_fps: float) -> None:
    import cv2
    from camera.capture import VideoOutputWriter

    first_capture = None
    for path in chunk_paths:
        first_capture = cv2.VideoCapture(str(path))
        if first_capture.isOpened():
            break
        first_capture.release()
        first_capture = None
    if first_capture is None:
        raise RuntimeError("No rendered chunk videos could be opened for stitching.")

    try:
        width = int(first_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(first_capture.get(cv2.CAP_PROP_FPS) or fallback_fps)
    finally:
        first_capture.release()

    writer = VideoOutputWriter(output_path, width, height, fps, output_fourcc)
    try:
        for path in chunk_paths:
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open rendered chunk: {path}")
                while True:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    writer.write(frame)
            finally:
                capture.release()
    finally:
        writer.close()


def run_parallel_assignment(config: PipelineConfig) -> bool:
    if not eligible_for_parallel_single(config):
        return False

    total_frames, fps, _width, _height = _probe_video(config)
    if total_frames <= 0 and config.benchmark_frames:
        total_frames = config.benchmark_frames
    workers = resolve_parallel_workers(config, total_frames=total_frames, fps=fps)
    if workers <= 1:
        return False

    chunk_root = config.project_root / ".kinara_runtime" / "chunks" / f"run_{os.getpid()}_{config.run_index}"
    if chunk_root.exists():
        shutil.rmtree(chunk_root, ignore_errors=True)
    chunk_root.mkdir(parents=True, exist_ok=True)

    specs = build_chunk_specs(config, total_frames=total_frames, fps=fps, chunk_root=chunk_root)
    if len(specs) <= 1:
        shutil.rmtree(chunk_root, ignore_errors=True)
        return False

    workers = min(workers, len(specs))
    chunk_seconds = resolve_parallel_chunk_seconds(config, total_frames=total_frames, fps=fps)
    overlap_seconds = resolve_parallel_overlap_seconds(config)
    started_at = time.perf_counter()
    log_info(
        "Parallel single-person processing: "
        f"{workers} workers, {len(specs)} chunks, {chunk_seconds:.2f}s chunks, {overlap_seconds:.2f}s overlap"
    )

    try:
        results: list[ChunkResult] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_chunk, config, spec) for spec in specs]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda result: result.index)

        rendered_paths = [result.rendered_path for result in results]
        motion_frames = [
            frame
            for result in results
            for frame in result.motion_frames
        ]
        motion_frames.sort(key=lambda frame: int(frame.get("frame_index", 0)))
        _concatenate_rendered_chunks(rendered_paths, config.rendered_output_path, config.output_fourcc, fps)

        export_metadata: dict[str, Any] = {
            "mode": "single-parallel",
            "profile": config.profile,
            "body_backend": config.body_backend,
            "hand_backend": config.hand_backend,
            "mediapipe_pose_model": config.mediapipe_pose_model,
            "source": str(config.video_path),
            "body_model_variant": config.body_model_variant,
            "hand_model_variant": config.hand_model_variant,
            "parallel_workers": workers,
            "parallel_chunk_seconds": chunk_seconds,
            "parallel_overlap_seconds": overlap_seconds,
            "parallel_chunk_count": len(results),
        }
        export_motion_bundle(config, fps=fps, frames=motion_frames, metadata=export_metadata)
        write_run_metadata(
            config.metadata_output_path,
            build_run_metadata(
                config,
                mode="single-parallel",
                fps=fps,
                frame_count=len(motion_frames),
                extra=export_metadata,
            ),
        )
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        log_info(f"Processed {len(motion_frames)} frames in {elapsed:.2f}s ({len(motion_frames) / elapsed:.2f} FPS)")
        print_saved_paths(config.output_path, config.json_output_path, config.fbx_output_path, config.metadata_output_path)
        return True
    except Exception as exc:
        log_warning(f"Parallel processing failed; falling back to serial processing: {type(exc).__name__}: {exc}")
        return False
    finally:
        shutil.rmtree(chunk_root, ignore_errors=True)
