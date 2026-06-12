# Body Backend Options

Kinara currently ships two body paths:

- `rtmpose`: RTMPose through `rtmlib`. This is the preferred CUDA body path for RTX 50-series / `sm_120` systems.
- `rtmpose-wholebody`: RTMPose WholeBody through `rtmlib`; body and hand backends are selected together.
- `yolo`: Legacy Ultralytics YOLO pose models. Keep this for compatibility and comparisons.
- `mediapipe`: MediaPipe pose. This is useful for simple local single-person tracking, but it is not the recommended target for future high-speed multi-person work.

For RTX 50-series RTMPose runs, prefer:

```powershell
kinara --landmark-backend rtmpose --rtmpose-device cuda --rtmpose-mode balanced --benchmark-frames 120 --no-preview
```

To let RTMPose own both body and hands:

```powershell
kinara --landmark-backend rtmpose-wholebody --rtmpose-device cuda --rtmpose-mode balanced --benchmark-frames 120 --no-preview
```

In the launcher, choosing `rtmpose-wholebody` disables the separate body and hand backend controls because WholeBody is a paired body+hand model.

If CUDA runtime compatibility is not ready yet, fall back to CPU only for diagnosis:

```powershell
kinara --landmark-backend rtmpose --rtmpose-device cpu --rtmpose-mode lightweight --benchmark-frames 30 --no-preview
```

For legacy YOLO comparisons:

```powershell
kinara --landmark-backend yolo --yolo-fast-preset nano --benchmark-frames 120 --no-preview
```

Legacy YOLO tradeoffs:

- `--yolo-fast-preset nano`: fastest, lowest body accuracy.
- `--yolo-fast-preset small`: still light, often a better stability/speed balance.
- `--yolo-fast-preset medium`: balanced offline or GPU mode.
- `--yolo-fast-preset xlarge`: highest YOLO quality, slowest.

RTMPose/RTMO from the OpenMMLab/MMPose family is the preferred non-MediaPipe direction. RTMPose reports strong real-time CPU/GPU speed while keeping high COCO AP, and RTMO is a newer one-stage real-time multi-person pose direction. Kinara's RTMPose path uses `rtmlib` and returns the same 17 COCO body keypoints as the legacy YOLO runner.

Current implementation path:

1. `body_backend="rtmpose"` uses `rtmlib` and returns Kinara `BodyDetection` records.
2. `body_backend="rtmpose-wholebody"` maps WholeBody output into Kinara body and hand payloads.
3. Existing Kinara smoothing, identity, fusion, export, and Unreal UDP layers stay unchanged.
4. Benchmark against legacy YOLO with `--benchmark-frames`.
5. A future TensorRT engine/cache path can build on the current ONNX Runtime backend.
