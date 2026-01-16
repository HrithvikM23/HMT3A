Human Motion to Animation Pipeline

This repository contains a hybrid Blender + Unity + MediaPipe workflow that converts human motion (from webcam or video) into 3D character animation.

The system tracks a person using MediaPipe Holistic, streams body landmarks to Unity in real time, maps them onto a humanoid rig, and optionally exports usable keyframe animation for further use in Unity or Blender.

Project Goal

To create an end-to-end motion capture pipeline that:
Tracks full-body movement from a camera or pre-recorded video
Streams motion data to Unity in real time
Drives a humanoid character using the tracked data
Records and exports animation as keyframes
Allows further cleanup and rendering inside Blender

Camera / Video
       ↓
MediaPipe Holistic (Python)
       ↓   UDP stream (port 5052)
Unity Receiver (C#)
       ↓
Humanoid Character Mapping
       ↓
Real-time Preview
       ↓
Record Motion → .anim file
       ↓
(Optional) Export to Blender for polishing

How to Run (Basic)
Step 1 — Start Unity

Open Mediapipe_Unity_Client in Unity

Press Play

Step 2 — Run MediaPipe

In VS Code:
cd HMT3A_Unity/mediapipe_server
python main.py
Choose:

1 → Webcam

2 → Video file

If everything is correct, your Unity character will respond to your movement.

Current Status
✅ MediaPipe tracking working
✅ UDP streaming to Unity working
🔄 Character mapping in progress
⏳ Animation recording system pending
⏳ Blender cleanup pipeline pending

Future Work
Planned improvements:
Full-body bone mapping
Inverse Kinematics (IK) in Unity
One-click animation recording
Automatic export to FBX
Better smoothing and filtering
Blender automation scripts

Tech Stack
Python: MediaPipe, OpenCV, NumPy
Unity: C#, Animation Rigging
Blender: FBX import/export, animation cleanup
