import cv2
import mediapipe as mp
import json
from tkinter import Tk, filedialog

print("\n" + "="*60)
print("   MOTION CAPTURE - STEP 1")
print("="*60)

# ============================================
# CHOOSE INPUT
# ============================================
print("\nSelect video source:")
print("  [1] Webcam")
print("  [2] Video file")

choice = input("\nChoice (1 or 2): ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Using webcam")
else:
    # Create file dialog
    print("\nOpening file browser...")
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
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
    print(f"Using video: {video_path}")

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

# Output files
OUTPUT_VIDEO = "output_motion.mp4"
OUTPUT_JSON = "motion_data_cleaned.json"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ============================================
# MEDIAPIPE SETUP
# ============================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# KEY BODY LANDMARKS
# ============================================
BODY_PARTS = {
    'nose': 0,
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

# ============================================
# CAPTURE LOOP
# ============================================
motion_data = []
frame_count = 0

print("\n" + "="*60)
print("RECORDING...")
print("Press 'Q' to stop")
print("="*60 + "\n")

# Warm up camera
if choice == "1":
    print("Warming up camera...")
    for _ in range(30):
        cap.read()
    print("✓ Ready!\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    
    # Detect pose
    results = pose.process(rgb)
    
    rgb.flags.writeable = True
    annotated_frame = frame.copy()
    
    # Prepare frame data
    frame_data = {
        'frame': frame_count,
        'timestamp': frame_count / fps,
        'body': {}
    }
    
    # Draw skeleton and extract data
    if results.pose_landmarks:
        # Draw on video
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Extract landmark positions
        for name, idx in BODY_PARTS.items():
            landmark = results.pose_landmarks.landmark[idx]
            
            # Only save if visible
            if landmark.visibility > 0.5:
                frame_data['body'][name] = {
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'v': float(landmark.visibility)
                }
        
        # Only save frames with good tracking
        if len(frame_data['body']) >= 8:  # At least 8 body parts visible
            motion_data.append(frame_data)
    
    # Show frame counter
    cv2.putText(
        annotated_frame, 
        f"Frame: {frame_count} | Captured: {len(motion_data)}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 0), 
        2
    )
    
    # Write and display
    out.write(annotated_frame)
    cv2.imshow("Motion Capture (Press Q to stop)", annotated_frame)
    
    # Stop on Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n✓ Stopped by user")
        break
    
    # Progress update
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames, captured {len(motion_data)} good frames...")

# ============================================
# CLEANUP
# ============================================
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()

# ============================================
# SAVE DATA
# ============================================
output_data = {
    'metadata': {
        'fps': fps,
        'total_frames': len(motion_data),
        'duration': len(motion_data) / fps,
        'width': width,
        'height': height,
        'body_parts': list(BODY_PARTS.keys())
    },
    'frames': motion_data
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("CAPTURE COMPLETE!")
print("="*60)
print(f"Total frames processed: {frame_count}")
print(f"Good frames captured: {len(motion_data)}")
print(f"Duration: {len(motion_data)/fps:.2f} seconds")
print(f"\nVideo saved: {OUTPUT_VIDEO}")
print(f"Data saved: {OUTPUT_JSON}")
print(f"\nNext step: Run convert_to_unreal.py")
print("="*60)

input("\nPress Enter to exit...")