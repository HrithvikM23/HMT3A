import bpy
import json
import mathutils

# ================= CONFIG =================
JSON_PATH = "D:\IDT\HMTBVD\HMTBVD_Blender\outputs\motion_data_cleaned.json"
ARMATURE_NAME = "Armature"
SCALE = 3.0
# ==========================================

def mp_to_blender(lm):
    return mathutils.Vector((
        (lm["x"] - 0.5) * SCALE,
        (0.5 - lm["y"]) * SCALE,
        -lm["z"] * SCALE
    ))

bone_map = {
    "upper_arm.L": ("left_shoulder", "left_elbow"),
    "forearm.L":   ("left_elbow", "left_wrist"),
    "upper_arm.R": ("right_shoulder", "right_elbow"),
    "forearm.R":   ("right_elbow", "right_wrist"),
    "thigh.L":     ("left_hip", "left_knee"),
    "shin.L":      ("left_knee", "left_ankle"),
    "thigh.R":     ("right_hip", "right_knee"),
    "shin.R":      ("right_knee", "right_ankle"),
}

with open(JSON_PATH) as f:
    data = json.load(f)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = data["metadata"]["total_frames"]

arm = bpy.data.objects[ARMATURE_NAME]
bpy.context.view_layer.objects.active = arm

# Store rest pose directions
rest_dirs = {}
for bone in arm.data.bones:
    rest_dirs[bone.name] = (bone.tail_local - bone.head_local).normalized()

for i, frame in enumerate(data["frames"], start=1):
    scene.frame_set(i)
    bpy.ops.object.mode_set(mode='POSE')

    for bone_name, (a, b) in bone_map.items():
        if bone_name not in arm.pose.bones:
            continue

        lm_a = frame["body"].get(a)
        lm_b = frame["body"].get(b)
        if not lm_a or not lm_b:
            continue

        p1 = mp_to_blender(lm_a)
        p2 = mp_to_blender(lm_b)

        direction = (p2 - p1)
        if direction.length == 0:
            continue

        target_dir = direction.normalized()
        rest_dir = rest_dirs[bone_name]

        rot = rest_dir.rotation_difference(target_dir)

        pb = arm.pose.bones[bone_name]
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = rot
        pb.keyframe_insert(data_path="rotation_quaternion")

bpy.ops.object.mode_set(mode='OBJECT')
print("✔ Animation created")