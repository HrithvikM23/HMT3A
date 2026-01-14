
import json
import csv
import os

print("\n" + "="*60)
print("   CONVERT TO UNREAL")
print("="*60)

# ============================================
# CHECK INPUT FILE
# ============================================
# Output files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current script folder
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_JSON = os.path.join(OUTPUT_DIR, "motion_data_cleaned.json")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "motion_data_unreal.csv")

if not os.path.exists(INPUT_JSON):
    print(f"\nERROR: {INPUT_JSON} not found!")
    print("Run capture_motion.py first!")
    input("Press Enter to exit...")
    exit(1)

print(f"\nLoading: {INPUT_JSON}")

with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

metadata = data['metadata']
frames = data['frames']

print(f"Loaded {metadata['total_frames']} frames")
print(f"FPS: {metadata['fps']}")
print(f"Duration: {metadata['duration']:.2f}s")

# ============================================
# BODY PARTS TO EXPORT
# ============================================
BODY_PARTS = [
    'nose',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

# ============================================
# CREATE CSV FOR UNREAL DATA TABLE
# ============================================
print(f"\nCreating CSV for Unreal...")

# CSV headers
csv_headers = ['RowName', 'Frame', 'Timestamp']
for part in BODY_PARTS:
    csv_headers.extend([f'{part}_X', f'{part}_Y', f'{part}_Z'])

# Write CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()
    
    for frame in frames:
        row = {
    'RowName': str(frame['frame']),  # must be string
    'Frame': frame['frame'],
    'Timestamp': frame['timestamp']
}

        
        # Convert each body part to Unreal coordinates
        for part in BODY_PARTS:
            if part in frame['body']:
                lm = frame['body'][part]
                
                # Convert MediaPipe (0-1 normalized) to Unreal (cm)
                # Scale: 100 = 1 meter in Unreal
                x = round((lm['x'] - 0.5) * 200, 2)  # -100 to +100 cm
                y = round((lm['y'] - 0.5) * 200, 2)  # -100 to +100 cm
                z = round(-lm['z'] * 200, 2)         # Depth
                
                row[f'{part}_X'] = x
                row[f'{part}_Y'] = y
                row[f'{part}_Z'] = z
            else:
                # Missing data = origin
                row[f'{part}_X'] = 0
                row[f'{part}_Y'] = 0
                row[f'{part}_Z'] = 0
        
        writer.writerow(row)

print(f"CSV created: {OUTPUT_CSV}")

# ============================================
# PRINT INSTRUCTIONS
# ============================================
print("\n" + "="*60)
print("   UNREAL ENGINE IMPORT GUIDE")
print("="*60)

print("\n📋 STEP 1: Create Structure in Unreal")
print("   1. Open your Unreal project")
print("   2. Content Browser → Right-click")
print("   3. Blueprints → Structure")
print("   4. Name: S_MoCapFrame")
print("   5. Add these variables:")
print()
print("   Variable Name          Type        Default")
print("   " + "-"*50)
print("   Frame                  Integer     0")
print("   Timestamp              Float       0.0")

for part in BODY_PARTS:
    print(f"   {part}_X" + " "*(22-len(part)) + "Float       0.0")
    print(f"   {part}_Y" + " "*(22-len(part)) + "Float       0.0")
    print(f"   {part}_Z" + " "*(22-len(part)) + "Float       0.0")

print("\n📋 STEP 2: Import CSV as Data Table")
print("   1. Drag 'motion_data_unreal.csv' into Content Browser")
print("   2. Import Type: Data Table")
print("   3. Row Structure: S_MoCapFrame")
print("   4. Name it: DT_MotionData")

print("\n📋 STEP 3: Create Motion Player Blueprint")
print("   1. Create Blueprint Actor: BP_MotionPlayer")
print("   2. Add Static Mesh Component (Sphere)")
print("   3. Add Variables:")
print("      - CurrentFrame (Integer)")
print("      - DataTable (DataTable Reference → DT_MotionData)")
print("   4. Event Tick:")
print("      - Increment CurrentFrame")
print("      - Get Data Table Row (CurrentFrame)")
print("      - Set Actor Location (nose_X, nose_Y, nose_Z)")

print("\n📋 STEP 4: Test")
print("   1. Drag BP_MotionPlayer into level")
print("   2. Press Play")
print("   3. Sphere should follow motion!")

print("\n🎯 FILES CREATED:")
print(f"   ✓ {OUTPUT_CSV}")
print(f"   → Import this into Unreal Engine")

print("\n💡 COORDINATE SYSTEM:")
print("   - Units: Centimeters (Unreal standard)")
print("   - Range: -100 to +100 cm per axis")
print("   - X axis: Left (-) / Right (+)")
print("   - Y axis: Back (-) / Forward (+)")
print("   - Z axis: Down (-) / Up (+)")

print("\n🦴 BODY PARTS INCLUDED:")
for i, part in enumerate(BODY_PARTS, 1):
    print(f"   {i:2}. {part}")

print("\n" + "="*60)
print("✓ CONVERSION COMPLETE!")
print("="*60)
print("\nNext: Import CSV into Unreal Engine")
print("Follow the instructions above ⬆")

input("\nPress Enter to exit...")