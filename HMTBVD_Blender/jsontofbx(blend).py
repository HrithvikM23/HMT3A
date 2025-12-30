import bpy
import json
import mathutils

# ============================================
# CONFIGURATION - CHANGE THESE PATHS
# ============================================
JSON_FILE = r"D:\IDT\HMTBVD\motion_data_full.json"  # ← CHANGE THIS TO YOUR PATH
FBX_OUTPUT = r"D:\IDT\HMTBVD\motion_capture.fbx"    # ← WHERE TO SAVE FBX

# ============================================
# LOAD MOTION DATA
# ============================================
print("Loading motion data...")
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

metadata = data['metadata']
frames = data['frames']

print(f"✓ Loaded {metadata['total_frames']} frames at {metadata['fps']} FPS")
print(f"✓ Duration: {metadata['duration']:.2f} seconds")

# ============================================
# CLEAR EXISTING SCENE
# ============================================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ============================================
# CREATE ARMATURE SKELETON
# ============================================
print("Creating skeleton...")
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
armature = bpy.context.active_object
armature.name = "MoCap_Body"

# Set to edit mode to create bones
bpy.ops.object.mode_set(mode='EDIT')
bones = armature.data.edit_bones

# Remove default bone
if len(bones) > 0:
    bones.remove(bones[0])

# Scale for better visibility
SCALE = 1.5

# ============================================
# CREATE BODY BONES
# ============================================

# ROOT/HIPS
root = bones.new('Root')
root.head = (0, 0, 0)
root.tail = (0, 0, 0.1 * SCALE)

# SPINE
spine = bones.new('Spine')
spine.head = (0, 0, 0.5 * SCALE)
spine.tail = (0, 0, 0.8 * SCALE)
spine.parent = root

# CHEST
chest = bones.new('Chest')
chest.head = spine.tail
chest.tail = (0, 0, 1.0 * SCALE)
chest.parent = spine

# NECK
neck = bones.new('Neck')
neck.head = chest.tail
neck.tail = (0, 0, 1.15 * SCALE)
neck.parent = chest

# HEAD
head = bones.new('Head')
head.head = neck.tail
head.tail = (0, 0, 1.35 * SCALE)
head.parent = neck

# LEFT ARM
l_shoulder = bones.new('L_Shoulder')
l_shoulder.head = (0.15 * SCALE, 0, 0.95 * SCALE)
l_shoulder.tail = (0.35 * SCALE, 0, 0.95 * SCALE)
l_shoulder.parent = chest

l_upper_arm = bones.new('L_UpperArm')
l_upper_arm.head = l_shoulder.tail
l_upper_arm.tail = (0.6 * SCALE, 0, 0.95 * SCALE)
l_upper_arm.parent = l_shoulder

l_forearm = bones.new('L_Forearm')
l_forearm.head = l_upper_arm.tail
l_forearm.tail = (0.85 * SCALE, 0, 0.95 * SCALE)
l_forearm.parent = l_upper_arm

l_hand = bones.new('L_Hand')
l_hand.head = l_forearm.tail
l_hand.tail = (0.95 * SCALE, 0, 0.95 * SCALE)
l_hand.parent = l_forearm

# RIGHT ARM
r_shoulder = bones.new('R_Shoulder')
r_shoulder.head = (-0.15 * SCALE, 0, 0.95 * SCALE)
r_shoulder.tail = (-0.35 * SCALE, 0, 0.95 * SCALE)
r_shoulder.parent = chest

r_upper_arm = bones.new('R_UpperArm')
r_upper_arm.head = r_shoulder.tail
r_upper_arm.tail = (-0.6 * SCALE, 0, 0.95 * SCALE)
r_upper_arm.parent = r_shoulder

r_forearm = bones.new('R_Forearm')
r_forearm.head = r_upper_arm.tail
r_forearm.tail = (-0.85 * SCALE, 0, 0.95 * SCALE)
r_forearm.parent = r_upper_arm

r_hand = bones.new('R_Hand')
r_hand.head = r_forearm.tail
r_hand.tail = (-0.95 * SCALE, 0, 0.95 * SCALE)
r_hand.parent = r_forearm

# LEFT LEG
l_hip = bones.new('L_Hip')
l_hip.head = (0.1 * SCALE, 0, 0)
l_hip.tail = (0.1 * SCALE, 0, -0.45 * SCALE)
l_hip.parent = root

l_knee = bones.new('L_Knee')
l_knee.head = l_hip.tail
l_knee.tail = (0.1 * SCALE, 0, -0.9 * SCALE)
l_knee.parent = l_hip

l_foot = bones.new('L_Foot')
l_foot.head = l_knee.tail
l_foot.tail = (0.1 * SCALE, 0.15 * SCALE, -0.9 * SCALE)
l_foot.parent = l_knee

# RIGHT LEG
r_hip = bones.new('R_Hip')
r_hip.head = (-0.1 * SCALE, 0, 0)
r_hip.tail = (-0.1 * SCALE, 0, -0.45 * SCALE)
r_hip.parent = root

r_knee = bones.new('R_Knee')
r_knee.head = r_hip.tail
r_knee.tail = (-0.1 * SCALE, 0, -0.9 * SCALE)
r_knee.parent = r_hip

r_foot = bones.new('R_Foot')
r_foot.head = r_knee.tail
r_foot.tail = (-0.1 * SCALE, 0.15 * SCALE, -0.9 * SCALE)
r_foot.parent = r_knee

# ============================================
# CREATE FINGER BONES
# ============================================
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
finger_offsets_y = [0.03, 0.01, 0, -0.01, -0.02]  # Spread fingers

# LEFT HAND FINGERS
for i, finger_name in enumerate(finger_names):
    offset_y = finger_offsets_y[i] * SCALE
    for joint in range(4):  # 4 joints per finger
        bone_name = f'L_{finger_name}_{joint}'
        bone = bones.new(bone_name)
        
        x_pos = l_hand.tail[0] + (joint * 0.025 * SCALE)
        bone.head = (x_pos, offset_y, l_hand.tail[2])
        bone.tail = (x_pos + 0.025 * SCALE, offset_y, l_hand.tail[2])
        
        if joint == 0:
            bone.parent = l_hand
        else:
            bone.parent = bones[f'L_{finger_name}_{joint-1}']

# RIGHT HAND FINGERS
for i, finger_name in enumerate(finger_names):
    offset_y = finger_offsets_y[i] * SCALE
    for joint in range(4):
        bone_name = f'R_{finger_name}_{joint}'
        bone = bones.new(bone_name)
        
        x_pos = r_hand.tail[0] - (joint * 0.025 * SCALE)
        bone.head = (x_pos, offset_y, r_hand.tail[2])
        bone.tail = (x_pos - 0.025 * SCALE, offset_y, r_hand.tail[2])
        
        if joint == 0:
            bone.parent = r_hand
        else:
            bone.parent = bones[f'R_{finger_name}_{joint-1}']

print(f"✓ Created {len(bones)} bones")

# ============================================
# SWITCH TO POSE MODE FOR ANIMATION
# ============================================
bpy.ops.object.mode_set(mode='POSE')

# Set FPS
bpy.context.scene.render.fps = metadata['fps']

print("Animating skeleton...")

# ============================================
# ANIMATE FROM MOTION DATA
# ============================================
def convert_to_world(landmark, scale=SCALE):
    """Convert normalized MediaPipe coordinates to Blender world space"""
    return (
        (landmark['x'] - 0.5) * 3 * scale,  # X: center and scale
        landmark['z'] * 2 * scale,           # Y: depth (z becomes y)
        (1 - landmark['y']) * 2 * scale      # Z: invert y (screen to world)
    )

# Bone to landmark mapping
bone_landmark_map = {
    'Head': 'nose',
    'L_Shoulder': 'left_shoulder',
    'R_Shoulder': 'right_shoulder',
    'L_UpperArm': 'left_elbow',
    'R_UpperArm': 'right_elbow',
    'L_Forearm': 'left_wrist',
    'R_Forearm': 'right_wrist',
    'L_Hip': 'left_hip',
    'R_Hip': 'right_hip',
    'L_Knee': 'left_knee',
    'R_Knee': 'right_knee',
    'L_Foot': 'left_ankle',
    'R_Foot': 'right_ankle',
}

# Finger mapping
finger_landmark_map = {
    'Thumb': 'thumb_tip',
    'Index': 'index_tip',
    'Middle': 'middle_tip',
    'Ring': 'ring_tip',
    'Pinky': 'pinky_tip'
}

# Animate each frame
for frame_data in frames:
    frame_num = frame_data['frame']
    bpy.context.scene.frame_set(frame_num)
    
    body_landmarks = frame_data.get('body_landmarks', {})
    left_hand = frame_data.get('left_hand_landmarks', {})
    right_hand = frame_data.get('right_hand_landmarks', {})
    
    # Animate body bones
    for bone_name, landmark_name in bone_landmark_map.items():
        if landmark_name in body_landmarks:
            landmark = body_landmarks[landmark_name]
            if landmark['visibility'] > 0.5:  # Only use visible landmarks
                pose_bone = armature.pose.bones[bone_name]
                world_pos = convert_to_world(landmark)
                pose_bone.location = mathutils.Vector(world_pos)
                pose_bone.keyframe_insert(data_path="location", frame=frame_num)
    
    # Animate left hand fingers
    if left_hand:
        for finger_name, landmark_name in finger_landmark_map.items():
            if landmark_name in left_hand:
                landmark = left_hand[landmark_name]
                # Animate the tip bone (joint 3)
                bone_name = f'L_{finger_name}_3'
                if bone_name in armature.pose.bones:
                    pose_bone = armature.pose.bones[bone_name]
                    world_pos = convert_to_world(landmark)
                    pose_bone.location = mathutils.Vector(world_pos)
                    pose_bone.keyframe_insert(data_path="location", frame=frame_num)
    
    # Animate right hand fingers
    if right_hand:
        for finger_name, landmark_name in finger_landmark_map.items():
            if landmark_name in right_hand:
                landmark = right_hand[landmark_name]
                bone_name = f'R_{finger_name}_3'
                if bone_name in armature.pose.bones:
                    pose_bone = armature.pose.bones[bone_name]
                    world_pos = convert_to_world(landmark)
                    pose_bone.location = mathutils.Vector(world_pos)
                    pose_bone.keyframe_insert(data_path="location", frame=frame_num)
    
    # Progress
    if frame_num % 30 == 0:
        print(f"  Processed frame {frame_num}/{metadata['total_frames']}")

print(f"✓ Animation complete!")

# ============================================
# EXPORT TO FBX
# ============================================
print(f"Exporting to FBX: {FBX_OUTPUT}")

bpy.ops.export_scene.fbx(
    filepath=FBX_OUTPUT,
    use_selection=False,
    global_scale=1.0,
    apply_scale_options='FBX_SCALE_ALL',
    axis_forward='-Z',
    axis_up='Y',
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    add_leaf_bones=False
)

print(f"\n{'='*60}")
print(f"✓ SUCCESS!")
print(f"{'='*60}")
print(f"✓ Created armature with {len(armature.data.bones)} bones")
print(f"✓ Animated {metadata['total_frames']} frames")
print(f"✓ Saved FBX: {FBX_OUTPUT}")
print(f"\nYou can now import this FBX into:")
print(f"  • Blender (File → Import → FBX)")
print(f"  • Unreal Engine")
print(f"  • Unity")
print(f"  • Any 3D software that supports FBX")
print(f"{'='*60}")