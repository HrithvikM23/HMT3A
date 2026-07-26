import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PipelineConfig
from utils.parallel_sizing import is_gpu_backend, auto_parallel_workers

class TestParallelSizing(unittest.TestCase):
    def test_is_gpu_backend(self):
        config = PipelineConfig(video_path=Path("test.mp4"))

        # String "0" — PyTorch GPU index
        config.yolo_device = "0"
        config.rtmpose_device = "cpu"
        config.provider_names = ("CPUExecutionProvider",)
        config.body_backend = "mediapipe"
        self.assertTrue(is_gpu_backend(config))

        # Integer 0 — PyTorch GPU index
        config.yolo_device = 0
        self.assertTrue(is_gpu_backend(config))

        # Integer 1 — second GPU
        config.yolo_device = 1
        self.assertTrue(is_gpu_backend(config))

        # "cuda" device string
        config.yolo_device = "cpu"
        config.rtmpose_device = "cuda"
        config.provider_names = ("CPUExecutionProvider",)
        self.assertTrue(is_gpu_backend(config))

        # "cuda:0" device string
        config.rtmpose_device = "cuda:0"
        self.assertTrue(is_gpu_backend(config))

        # "cuda:1" device string
        config.rtmpose_device = "cuda:1"
        self.assertTrue(is_gpu_backend(config))

        # "mps" — Apple Silicon GPU
        config.yolo_device = "mps"
        config.rtmpose_device = "cpu"
        self.assertTrue(is_gpu_backend(config))

        # ONNX CUDAExecutionProvider
        config.yolo_device = None
        config.rtmpose_device = "cpu"
        config.provider_names = ("CUDAExecutionProvider",)
        self.assertTrue(is_gpu_backend(config))

        # ONNX TensorrtExecutionProvider
        config.provider_names = ("TensorrtExecutionProvider",)
        self.assertTrue(is_gpu_backend(config))

        # ONNX DmlExecutionProvider (DirectML)
        config.provider_names = ("DmlExecutionProvider",)
        self.assertTrue(is_gpu_backend(config))

        # ONNX ROCmExecutionProvider (AMD)
        config.provider_names = ("ROCMExecutionProvider",)
        self.assertTrue(is_gpu_backend(config))

        # GPU backend (yolo) with no explicit device — defaults to GPU
        config.yolo_device = None
        config.rtmpose_device = "cuda"
        config.provider_names = ("CPUExecutionProvider",)
        config.body_backend = "yolo"
        self.assertTrue(is_gpu_backend(config))

        # Fully CPU — yolo backend with explicit cpu devices
        config.yolo_device = "cpu"
        config.rtmpose_device = "cpu"
        config.provider_names = ("CPUExecutionProvider",)
        config.body_backend = "yolo"
        self.assertFalse(is_gpu_backend(config))

        # Fully CPU — mediapipe with CPU provider
        config.body_backend = "mediapipe"
        config.yolo_device = None
        config.rtmpose_device = "cpu"
        config.provider_names = ("CPUExecutionProvider",)
        self.assertFalse(is_gpu_backend(config))

    @patch('os.cpu_count')
    def test_auto_parallel_workers_gpu(self, mock_cpu_count):
        mock_cpu_count.return_value = 8
        config = PipelineConfig(video_path=Path("test.mp4"))
        config.yolo_device = "0"
        config.parallel_chunk_seconds = 5.0
        workers = auto_parallel_workers(config, total_frames=300, fps=30.0)
        self.assertLessEqual(workers, 8)

    @patch('os.cpu_count')
    def test_auto_parallel_workers_cpu(self, mock_cpu_count):
        mock_cpu_count.return_value = 16
        config = PipelineConfig(video_path=Path("test.mp4"))
        config.yolo_device = "cpu"
        config.rtmpose_device = "cpu"
        config.provider_names = ("CPUExecutionProvider",)
        config.parallel_chunk_seconds = 5.0
        workers = auto_parallel_workers(config, total_frames=900, fps=30.0)
        self.assertLessEqual(workers, 16)
        self.assertGreaterEqual(workers, 1)

    @patch('os.cpu_count')
    def test_auto_parallel_workers_bounds(self, mock_cpu_count):
        mock_cpu_count.return_value = 1
        config = PipelineConfig(video_path=Path("test.mp4"))
        config.yolo_device = "cpu"
        workers = auto_parallel_workers(config, total_frames=300, fps=30.0)
        self.assertEqual(workers, 1)

if __name__ == '__main__':
    unittest.main()
