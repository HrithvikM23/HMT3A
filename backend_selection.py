from __future__ import annotations

from typing import Any


BODY_BACKENDS = ("yolo", "mediapipe")
HAND_BACKENDS = ("onnx", "mediapipe")
LANDMARK_BACKENDS = ("yolo", "mediapipe", "hybrid")


def resolve_backend_selection(args: Any) -> tuple[str, str, bool]:
    body_backend = args.body_backend
    hand_backend = args.hand_backend
    fallback = bool(getattr(args, "backend_fallbacks", False))

    if args.landmark_backend == "mediapipe":
        body_backend = body_backend or "mediapipe"
        hand_backend = hand_backend or "mediapipe"
    elif args.landmark_backend == "hybrid":
        body_backend = body_backend or "mediapipe"
        hand_backend = hand_backend or "mediapipe"
        fallback = True
    else:
        body_backend = body_backend or "yolo"
        hand_backend = hand_backend or "onnx"

    return body_backend, hand_backend, fallback


def needs_yolo_body(body_backend: str, enable_fallbacks: bool) -> bool:
    return body_backend == "yolo" or (body_backend == "mediapipe" and enable_fallbacks)


def needs_onnx_hand(hand_backend: str, enable_fallbacks: bool) -> bool:
    return hand_backend == "onnx" or (hand_backend == "mediapipe" and enable_fallbacks)


def needs_mediapipe(body_backend: str, hand_backend: str, enable_fallbacks: bool) -> bool:
    return body_backend == "mediapipe" or hand_backend == "mediapipe" or enable_fallbacks
