from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import PipelineConfig
from runners.common import build_person_payload, draw_person_overlay, print_saved_paths
from runners.parallel_single import ChunkResult, ChunkSpec, _concatenate_rendered_chunks, _probe_video
from utils.exports import export_multi_person_fbx_bundle, export_multi_person_json
from utils.fps import FpsMeter, draw_fps_overlay
from utils.logging import log_info, log_warning
from utils.motion_cleanup import cleanup_multi_person_frames
from utils.multi_person import MultiPersonTracker
from utils.payloads import PersonPayload
from utils.preview_stream import PreviewFrameSink
from utils.run_metadata import build_run_metadata, write_run_metadata
from utils.parallel_sizing import resolve_parallel_workers, resolve_parallel_chunk_seconds, resolve_parallel_overlap_seconds


def build_multi_chunk_specs(
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

    workers = resolve_parallel_workers(config, total_frames=target_frames, fps=fps)
    if workers > 1 and config.parallel_chunk_seconds == 0:
        chunk_frames = max(1, math.ceil(target_frames / workers))
    else:
        chunk_seconds = resolve_parallel_chunk_seconds(config, total_frames=target_frames, fps=fps)
        chunk_frames = max(1, int(round(max(fps, 1.0) * chunk_seconds)))

    # Double the overlap for multi-person for identity stabilization
    overlap_seconds = resolve_parallel_overlap_seconds(config) * 2.0
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


def _process_multi_chunk(config: PipelineConfig, spec: ChunkSpec, worker_index: int | None = None) -> ChunkResult:
    config = copy.copy(config)
    from camera.capture import VideoInputSource, VideoOutputWriter
    from inference.rtmpose import ONNXPoseHandRunner
    from utils.bootstrap_paths import ensure_local_environment
    from utils.logging import install_safe_stdio
    from network.osc_sender import OSCSender
    from utils.preview_stream import PreviewFrameSink

    cpu_count = os.cpu_count() or 1
    pct = max(10.0, min(100.0, float(getattr(config, "max_cpu_percent", 60.0))))
    threads_count = max(1, math.floor(cpu_count * (pct / 100.0)))
    thread_cap = str(threads_count)
    os.environ["OMP_NUM_THREADS"] = thread_cap
    os.environ["MKL_NUM_THREADS"] = thread_cap
    os.environ["OPENBLAS_NUM_THREADS"] = thread_cap
    os.environ["ONNXRUNTIME_NUM_THREADS"] = thread_cap
    try:
        import cv2
        cv2.setNumThreads(threads_count)
    except Exception:
        import cv2

    install_safe_stdio()
    ensure_local_environment()

    if worker_index is None:
        env_idx = os.environ.get("KINARA_WORKER_INDEX")
        worker_index = int(env_idx) if env_idx is not None and env_idx.isdigit() else 0
    os.environ["KINARA_WORKER_INDEX"] = str(worker_index)

    config.enable_preview = False
    config.osc_enabled = False

    preview_sink = PreviewFrameSink(worker_index=worker_index)
    api_pref = cv2.CAP_FFMPEG if os.name == "nt" and not isinstance(config.video_path, int) else cv2.CAP_ANY
    source = VideoInputSource(config.video_path, fallback_fps=config.fallback_fps, api_preference=api_pref)
    writer = VideoOutputWriter(
        spec.rendered_path,
        frame_width=source.frame_width,
        frame_height=source.frame_height,
        fps=source.fps,
        output_fourcc=config.output_fourcc,
    )
    osc_sender = OSCSender(config.osc_host, config.osc_port, False)

    motion_frames_file = None
    try:
        runner = ONNXPoseHandRunner(config)
        tracker = MultiPersonTracker(config, runner)
        fps_meter = FpsMeter(f"chunk-{spec.index}", 0.0)
        motion_frames_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        motion_frames_path = motion_frames_file.name
        processed_frames = 0
        spec_total_frames = max(1, spec.end_frame - spec.start_frame)

        source.cap.set(1, spec.warmup_start_frame)
        actual_frame = int(source.cap.get(1))
        current_frame = actual_frame
        while current_frame < spec.end_frame:
            ok, frame = source.read()
            if not ok or frame is None:
                break

            payload_people: list[PersonPayload] = []
            tracks = tracker.update(frame)
            
            if current_frame < spec.warmup_start_frame:
                current_frame += 1
                continue

            for track in tracks:
                track.pipeline.render_pose(frame, track.body_points, track.hands_by_side, send_osc=False)
                label = track.label or f"person{track.id}"
                draw_person_overlay(frame, label, track.box, track.detection_score)
                joint_depths = track.joint_depths if config.single_camera_depth_mode == "mediapipe" else None
                payload_people.append(
                    build_person_payload(
                        person_id=track.id,
                        label=label,
                        box=track.box,
                        score=track.detection_score,
                        body_points=track.body_points,
                        hands_by_side=track.hands_by_side,
                        joint_depths=joint_depths,
                        camera_views=["CAM_0"],
                    )
                )

            if current_frame >= spec.start_frame:
                motion_frames_file.write(json.dumps({"frame_index": current_frame, "people": payload_people}) + "\n")
                fps_meter.tick(current_frame)
                draw_fps_overlay(frame, fps_meter, config.fps_overlay_enabled)
                preview_sink.write(frame, current_frame)
                writer.write(frame)
                processed_frames += 1

                step = max(1, spec_total_frames // 4)
                if processed_frames % step == 0 or processed_frames == spec_total_frames:
                    pct = (processed_frames / max(1, spec_total_frames)) * 100.0
                    log_info(
                        f"[Worker {worker_index}] Chunk {spec.index + 1} progress: {pct:.1f}% "
                        f"({processed_frames}/{spec_total_frames} frames)"
                    )
            current_frame += 1
    finally:
        source.close()
        writer.close()
        osc_sender.close()
        if hasattr(preview_sink, 'close'):
            preview_sink.close()
        if motion_frames_file is not None:
            motion_frames_file.close()

    completion_pct = (processed_frames / max(1, spec_total_frames)) * 100.0
    log_info(
        f"[Worker {worker_index}] Chunk {spec.index + 1} completed: {completion_pct:.1f}% "
        f"({processed_frames}/{spec_total_frames} frames)"
    )

    return ChunkResult(
        index=spec.index,
        start_frame=spec.start_frame,
        end_frame=spec.end_frame,
        rendered_path=spec.rendered_path,
        motion_frames_path=motion_frames_path,
        processed_frames=processed_frames,
    )

def _stitch_identities(frames: list[dict]) -> list[dict]:
    # Placeholder for identity stitching logic via color matching
    # Since we can't fully implement it without all tracks, this is a basic stub that
    # assumes person_id is good enough or will be refined later
    return frames

def run_parallel_multi_person(config: PipelineConfig) -> bool:
    total_frames, fps, _width, _height = _probe_video(config)
    if total_frames <= 0 and config.benchmark_frames:
        total_frames = config.benchmark_frames
    workers = resolve_parallel_workers(config, total_frames=total_frames, fps=fps)

    chunk_root = config.project_root / ".kinara_runtime" / "chunks" / f"run_multi_{os.getpid()}_{config.run_index}"
    if chunk_root.exists():
        shutil.rmtree(chunk_root, ignore_errors=True)
    chunk_root.mkdir(parents=True, exist_ok=True)

    specs = build_multi_chunk_specs(config, total_frames=total_frames, fps=fps, chunk_root=chunk_root)
    if len(specs) <= 0:
        shutil.rmtree(chunk_root, ignore_errors=True)
        return False

    workers = min(workers, len(specs))
    chunk_seconds = resolve_parallel_chunk_seconds(config, total_frames=total_frames, fps=fps)
    overlap_seconds = resolve_parallel_overlap_seconds(config) * 2.0
    started_at = time.perf_counter()
    log_info(
        "Parallel multi-person processing: "
        f"{workers} workers, {len(specs)} chunks, {chunk_seconds:.2f}s chunks, {overlap_seconds:.2f}s overlap"
    )

    chunk_paths = [spec.rendered_path for spec in specs]
    try:
        results: list[ChunkResult] = []
        completed_chunks = 0
        total_chunks = len(specs)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_multi_chunk, config, spec, spec.index % workers)
                for spec in specs
            ]
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                completed_chunks += 1
                progress_pct = (completed_chunks / total_chunks) * 100.0
                log_info(
                    f"Chunk {res.index + 1}/{total_chunks} finished by worker {res.index % workers} "
                    f"({completed_chunks}/{total_chunks} chunks completed, {progress_pct:.1f}%)"
                )
        results.sort(key=lambda result: result.index)
        parallel_time = max(time.perf_counter() - started_at, 1e-9)

        rendered_paths = [result.rendered_path for result in results]
        motion_frames = []
        for result in results:
            if result.motion_frames_path and os.path.exists(result.motion_frames_path):
                with open(result.motion_frames_path, "r") as f:
                    for line in f:
                        if line.strip():
                            motion_frames.append(json.loads(line))
                try:
                    os.unlink(result.motion_frames_path)
                except OSError:
                    pass
        
        motion_frames.sort(key=lambda frame: int(frame.get("frame_index", 0)))
        stitched_frames = _stitch_identities(motion_frames)
        cleaned_motion_frames = cleanup_multi_person_frames(stitched_frames, config)

        frame_cnt = len(cleaned_motion_frames) or total_frames
        log_info(
            f"Parallel inference completed in {parallel_time:.2f}s "
            f"({frame_cnt / parallel_time:.2f} FPS across {workers} workers)"
        )
        log_info("Stitching video chunks and exporting motion bundle...")

        _concatenate_rendered_chunks(rendered_paths, config.rendered_output_path, config.output_fourcc, fps)

        export_metadata = {
            "mode": "multi-parallel",
            "profile": config.profile,
            "body_backend": config.body_backend,
            "hand_backend": config.hand_backend,
            "mediapipe_pose_model": config.mediapipe_pose_model,
            "source": str(config.video_path),
            "max_people": config.max_people,
            "identity_hints": {key: list(value) for key, value in config.identity_hints.items()},
            "parallel_workers": workers,
            "parallel_chunk_seconds": chunk_seconds,
            "parallel_overlap_seconds": overlap_seconds,
            "parallel_chunk_count": len(results),
        }
        export_multi_person_json(
            config.json_output_path,
            fps=fps,
            frames=cleaned_motion_frames,
            metadata=export_metadata,
        )
        exported_fbx_paths = export_multi_person_fbx_bundle(config.fbx_output_path, fps, cleaned_motion_frames)
        write_run_metadata(
            config.metadata_output_path,
            build_run_metadata(
                config,
                mode="multi-parallel",
                fps=fps,
                frame_count=len(cleaned_motion_frames),
                extra=export_metadata,
            ),
        )
        total_time = max(time.perf_counter() - started_at, 1e-9)
        export_time = max(0.0, total_time - parallel_time)
        log_info(
            f"Total pipeline run completed in {total_time:.2f}s "
            f"(Inference: {parallel_time:.2f}s, Export: {export_time:.2f}s)"
        )
        print_saved_paths(config.output_path, config.json_output_path, *exported_fbx_paths, config.metadata_output_path)
        return True
    except Exception as exc:
        log_warning(f"Parallel multi-person processing failed; falling back to serial processing: {type(exc).__name__}: {exc}")
        return False
    finally:
        for p in chunk_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(chunk_root, ignore_errors=True)
