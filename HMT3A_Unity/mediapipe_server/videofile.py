import cv2
import mediapipe as mp
import json
import time
import tkinter as tk
from tkinter import filedialog

from sender import LandmarkSender   # your sender class

def run_video():

    # ---------- FILE PICKER (FOREGROUND) ----------
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    video_path = filedialog.askopenfilename(
        title="Select a video for MediaPipe",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not video_path:
        print("No file selected. Exiting.")
        return

    print(f"\nSelected video:\n{video_path}\n")

    # ---------- SETUP ----------
    sender = LandmarkSender()

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    # ---- progress info ----
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video loaded:")
    print(f"Total frames : {total_frames}")
    print(f"FPS          : {fps:.1f}\n")

    frame_idx = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            landmarks_dict = {}

            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks_dict[i] = {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z)
                    }

            # send to Unity
            sender.send(landmarks_dict)

            # draw skeleton on video
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

            cv2.imshow("MediaPipe Video", frame)

            # ----- progress in terminal -----
            percent = (frame_idx / total_frames) * 100
            print(
                f"\rProcessed: {frame_idx}/{total_frames} frames ({percent:.1f}%)",
                end=""
            )

            #
