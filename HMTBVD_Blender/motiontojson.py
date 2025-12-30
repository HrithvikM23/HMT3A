import cv2
import mediapipe as mp
import numpy as np
from tkinter import filedialog
import tkinter as tk
import json

# ============================================
# CONFIGURATION
# ============================================
def get_video_source():
    """Choose between webcam or video file"""
    print("\n" + "="*60)
    print("   FULL BODY + FINGER MOTION CAPTURE")
    print("="*60)
    print("\nSelect video source:")
    print("  [W] - Use Webcam")
    print("  [F] - Select video File")
    print("="*60)
    
    while True:
        choice = input("\nEnter your choice (W/F): ").strip().upper()
        if choice == 'W':
            print("\n✓ Using webcam...")
            return None, True
        elif choice == 'F':
            print("\n✓ Opening file browser...")
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)  # Bring to foreground
            root.lift()
            root.focus_force()
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ],
                parent=root
            )
            root.destroy()
            
            if file_path:
                print(f"✓ Selected: {file_path}")
                return file_path, False
            else:
                print("\nNo file selected. Try again.")
        else:
            print("Invalid. Enter 'W' or 'F'.")

INPUT_VIDEO, USE_WEBCAM = get_video_source()
OUTPUT_VIDEO = "output_full_skeleton.mp4"
OUTPUT_JSON = "motion_data_full.json"

# ============================================
# MEDIAPIPE SETUP - POSE + HANDS
# ============================================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Highest accuracy
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,  # Back to complexity 1 to avoid C2__PACKET error
    min_detection_confidence=0.7,  # Higher for better finger tracking
    min_tracking_confidence=0.7
)

# ============================================
# OPEN VIDEO
# ============================================
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("Error: Could not open video source")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0 or fps > 120:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ============================================
# BODY LANDMARKS (33 points)
# ============================================
BODY_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}

# ============================================
# HAND LANDMARKS (21 points per hand)
# ============================================
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
# PROCESS VIDEO
# ============================================
motion_data = []
frame_count = 0

print(f"\n{'='*60}")
print(f"Processing video with FULL FINGER TRACKING...")
print(f"Input: {'Webcam' if USE_WEBCAM else INPUT_VIDEO}")
print(f"Output video: {OUTPUT_VIDEO}")
print(f"Motion data: {OUTPUT_JSON}")
print(f"FPS: {fps}")
print(f"Tracking: 33 body + 42 finger landmarks (21 per hand)")
print(f"Press 'Q' to stop")
print(f"{'='*60}\n")

# Wait for camera to initialize if using webcam
if USE_WEBCAM:
    print("Warming up camera...")
    for _ in range(30):
        cap.read()
    print("Camera ready!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Ensure frame is valid
    if frame is None or frame.size == 0:
        continue
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False  # Improve performance
    
    # Process both pose and hands
    try:
        pose_results = pose.process(rgb)
        hands_results = hands.process(rgb)
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        continue
    
    rgb.flags.writeable = True

    # Draw on frame
    annotated_frame = frame.copy()
    
    # Initialize frame data
    frame_data = {
        'frame': frame_count,
        'timestamp': frame_count / fps,
        'body_landmarks': {},
        'left_hand_landmarks': {},
        'right_hand_landmarks': {}
    }

    tracking_status = []

    # ============================================
    # BODY TRACKING
    # ============================================
    if pose_results.pose_landmarks:
        # Draw pose
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Extract body landmarks
        for name, idx in BODY_LANDMARKS.items():
            landmark = pose_results.pose_landmarks.landmark[idx]
            frame_data['body_landmarks'][name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        tracking_status.append("Body: ✓")
    else:
        tracking_status.append("Body: ✗")

    # ============================================
    # HAND TRACKING
    # ============================================
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            # Determine which hand (left or right)
            handedness = hands_results.multi_handedness[hand_idx].classification[0].label
            
            # Draw hand skeleton
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract finger landmarks
            hand_data = {}
            for name, idx in HAND_LANDMARKS.items():
                landmark = hand_landmarks.landmark[idx]
                hand_data[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                }

            # Store in correct hand
            if handedness == "Left":
                frame_data['left_hand_landmarks'] = hand_data
                tracking_status.append("L.Hand: ✓")
            else:
                frame_data['right_hand_landmarks'] = hand_data
                tracking_status.append("R.Hand: ✓")

    motion_data.append(frame_data)

    # ============================================
    # DISPLAY MINIMAL STATUS - TOP RIGHT
    # ============================================
    # Simple frame counter in top right corner
    frame_text = f"Frame: {frame_count}"
    
    # Get text size for positioning
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = width - text_size[0] - 10  # 10px from right edge
    text_y = 25  # 25px from top
    
    cv2.putText(annotated_frame, frame_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Write frame
    out.write(annotated_frame)

    # Display
    cv2.imshow("Motion Capture (Press Q to stop)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nStopped by user")
        break

    # Progress
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
hands.close()

# ============================================
# SAVE MOTION DATA
# ============================================
output_data = {
    'metadata': {
        'fps': fps,
        'total_frames': frame_count,
        'duration': frame_count / fps,
        'width': width,
        'height': height,
        'body_landmarks': len(BODY_LANDMARKS),
        'hand_landmarks_per_hand': len(HAND_LANDMARKS),
        'total_possible_landmarks': len(BODY_LANDMARKS) + 2 * len(HAND_LANDMARKS)
    },
    'landmark_definitions': {
        'body': BODY_LANDMARKS,
        'hand': HAND_LANDMARKS
    },
    'frames': motion_data
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n{'='*60}")
print(f"✓ PROCESSING COMPLETE!")
print(f"{'='*60}")
print(f"Total frames: {frame_count}")
print(f"Duration: {frame_count/fps:.2f} seconds")
print(f"Body landmarks: {len(BODY_LANDMARKS)}")
print(f"Finger landmarks per hand: {len(HAND_LANDMARKS)}")
print(f"Total landmarks: {len(BODY_LANDMARKS) + 2*len(HAND_LANDMARKS)}")
print(f"\n✓ Saved video: {OUTPUT_VIDEO}")
print(f"✓ Saved motion: {OUTPUT_JSON}")
