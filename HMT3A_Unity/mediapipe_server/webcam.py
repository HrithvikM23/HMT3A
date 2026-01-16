import cv2
import mediapipe as mp
import json
import socket
from sender import LandmarkSender

UDP_IP = "127.0.0.1"
UDP_PORT = 5052

def run_webcam():

    sender = LandmarkSender()

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

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

            frame = cv2.flip(frame, 1)
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

            sender.send(landmarks_dict)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

            cv2.imshow("MediaPipe Webcam", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
