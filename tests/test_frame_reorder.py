import unittest

class TestFrameReorder(unittest.TestCase):
    def test_frame_reordering_basic(self):
        # Given frames out of order
        frames = [{"frame_index": 3}, {"frame_index": 1}, {"frame_index": 2}]
        # When ordered
        frames.sort(key=lambda x: x["frame_index"])
        # Then they are sequential
        self.assertEqual(frames[0]["frame_index"], 1)
        self.assertEqual(frames[1]["frame_index"], 2)
        self.assertEqual(frames[2]["frame_index"], 3)

    def test_boundary_conditions(self):
        frames = []
        frames.sort(key=lambda x: x["frame_index"])
        self.assertEqual(len(frames), 0)

        frames = [{"frame_index": 1}]
        frames.sort(key=lambda x: x["frame_index"])
        self.assertEqual(len(frames), 1)

if __name__ == '__main__':
    unittest.main()
