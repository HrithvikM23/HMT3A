import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import json
from tkinter import Tk, filedialog

# ============================================
# VIDEO SOURCE SELECTION
# ============================================
print("\n" + "="*60)
print("   BODY + HANDS MOTION CAPTURE (NO HEAD)")
print("="*60)
print("\nSelect video source:")
print("  [1] Webcam")
print("  [2] Video file")

choice = input("\nChoice (1 or 2): ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Using webcam")
    USE_WEBCAM = True
else:
    print("\nOpening file browser...")
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not video_path:
        print("ERROR: No file selected!")
        input("Press Enter to exit...")
        exit(1)

    cap = cv2.VideoCapture(video_path)
    print(f"✓ Selected: {video_path}")
    USE_WEBCAM = False

if not cap.isOpened():
    print("ERROR: Cannot open video source!")
    input("Press Enter to exit...")
    exit(1)

# ============================================
# VIDEO SETTINGS
# ============================================
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0 or fps > 120:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "output_motion.mp4")
OUTPUT_JSON  = os.path.join(OUTPUT_DIR, "motion_data_cleaned.json")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ============================================
# MEDIAPIPE SETUP
# ============================================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# LANDMARK DEFINITIONS
# ============================================
BODY_LANDMARKS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

HAND_LANDMARKS = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_mcp': 5,
    'index_pip': 6,
    'index_dip': 7,
    'index_tip': 8,
    'middle_mcp': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    'middle_tip': 12,
    'ring_mcp': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    'ring_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20
}

# ============================================
# CAPTURE LOOP
# ============================================
motion_data = []
frame_count = 0

print("\n" + "="*60)
print("RECORDING - Body: 12 points | Hands: 21 points each")
print("Press 'Q' to stop")
print("="*60 + "\n")

if USE_WEBCAM:
    for _ in range(30):
        cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)

    rgb.flags.writeable = True
    annotated_frame = frame.copy()

    frame_data = {
        'frame': frame_count,
        'timestamp': (frame_count - 1) / fps,
        'body': {},
        'left_hand': {},
        'right_hand': {}
    }

    # Initialize fixed hand structure
    for name in HAND_LANDMARKS:
        frame_data['left_hand'][name] = None
        frame_data['right_hand'][name] = None

    # BODY
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        for name, idx in BODY_LANDMARKS.items():
            landmark = pose_results.pose_landmarks.landmark[idx]
            frame_data['body'][name] = {
                'x': float(landmark.x),
                'y': float(landmark.y),
                'z': float(landmark.z),
                'visibility': float(landmark.visibility)
            }

    # HANDS
    if hands_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            hand_key = 'left_hand' if hand_idx == 0 else 'right_hand'

            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for name, idx in HAND_LANDMARKS.items():
                lm = hand_landmarks.landmark[idx]
                frame_data[hand_key][name] = {
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z)
                }

    motion_data.append(frame_data)

    cv2.putText(
        annotated_frame,
        f"F:{frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    out.write(annotated_frame)
    cv2.imshow("Motion Capture (Q to stop)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
hands.close()

# ============================================
# SAVE DATA
# ============================================
output_data = {
    'metadata': {
        'fps': fps,
        'total_frames': frame_count,
        'duration': frame_count / fps,
        'width': width,
        'height': height,
        'body_landmarks': list(BODY_LANDMARKS.keys()),
        'hand_landmarks': list(HAND_LANDMARKS.keys()),
        'body_count': len(BODY_LANDMARKS),
        'hand_count_per_hand': len(HAND_LANDMARKS)
    },
    'frames': motion_data
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)

print("\nCAPTURE COMPLETE")
print(f"Video: {OUTPUT_VIDEO}")
print(f"Data: {OUTPUT_JSON}")
 