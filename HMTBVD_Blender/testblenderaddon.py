bl_info = {
    "name": "PosePipe (Modern)",
    "author": "HrithvikM",
    "version": (1, 0, 0),
    "blender": (4, 5, 4),
    "location": "View3D > Sidebar > PosePipe",
    "description": "Import MediaPipe pose JSON and build constraint-driven mocap rig",
    "category": "Animation",
}

import bpy
import json
from bpy.types import Panel, Operator
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
import mathutils

# -------------------------------
# CONFIG
# -------------------------------

BODY_BONES = {
    "upper_arm.L": ("left_shoulder", "left_elbow"),
    "forearm.L":   ("left_elbow", "left_wrist"),
    "upper_arm.R": ("right_shoulder", "right_elbow"),
    "forearm.R":   ("right_elbow", "right_wrist"),
    "thigh.L":     ("left_hip", "left_knee"),
    "shin.L":      ("left_knee", "left_ankle"),
    "thigh.R":     ("right_hip", "right_knee"),
    "shin.R":      ("right_knee", "right_ankle"),
}

SCALE = 3.0

def mp_to_blender(lm):
    return mathutils.Vector((
        (lm["x"] - 0.5) * SCALE,
        (0.5 - lm["y"]) * SCALE,
        -lm["z"] * SCALE
    ))

# -------------------------------
# JSON IMPORT OPERATOR
# -------------------------------

class POSEPIPE_OT_import_json(Operator, ImportHelper):
    bl_idname = "posepipe.import_json"
    bl_label = "Import MediaPipe JSON"
    filename_ext = ".json"

    def execute(self, context):
        with open(self.filepath, "r") as f:
            data = json.load(f)

        scene = context.scene
        scene.frame_start = 1
        scene.frame_end = data["metadata"]["total_frames"]

        # Create empties
        empties = {}
        for name in data["metadata"]["body_landmarks"]:
            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 0.04
            context.collection.objects.link(empty)
            empties[name] = empty

        # Animate empties
        for i, frame in enumerate(data["frames"], start=1):
            scene.frame_set(i)
            for name, lm in frame["body"].items():
                if lm:
                    empties[name].location = mp_to_blender(lm)
                    empties[name].keyframe_insert("location")

        self.report({'INFO'}, "PosePipe: Trackers created")
        return {'FINISHED'}

# -------------------------------
# ARMATURE BUILDER
# -------------------------------

class POSEPIPE_OT_build_rig(Operator):
    bl_idname = "posepipe.build_rig"
    bl_label = "Build Skeleton"

    def execute(self, context):
        bpy.ops.object.armature_add()
        arm = context.object
        arm.name = "PosePipe_Armature"

        bpy.ops.object.mode_set(mode='EDIT')
        eb = arm.data.edit_bones

        def make_bone(name, head, tail):
            b = eb.new(name)
            b.head = bpy.data.objects[head].location
            b.tail = bpy.data.objects[tail].location
            return b

        for bone, (h, t) in BODY_BONES.items():
            make_bone(bone, h, t)

        bpy.ops.object.mode_set(mode='POSE')

        for bone, (h, t) in BODY_BONES.items():
            pb = arm.pose.bones[bone]

            cl = pb.constraints.new("COPY_LOCATION")
            cl.target = bpy.data.objects[h]

            st = pb.constraints.new("STRETCH_TO")
            st.target = bpy.data.objects[t]
            st.volume = 'NO_VOLUME'

        self.report({'INFO'}, "PosePipe: Skeleton built and constrained")
        return {'FINISHED'}

# -------------------------------
# UI PANEL
# -------------------------------

class POSEPIPE_PT_panel(Panel):
    bl_label = "PosePipe (Modern)"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PosePipe'

    def draw(self, context):
        layout = self.layout
        layout.operator("posepipe.import_json", icon='FILE_FOLDER')
        layout.operator("posepipe.build_rig", icon='ARMATURE_DATA')

# -------------------------------
# REGISTER
# -------------------------------

classes = (
    POSEPIPE_OT_import_json,
    POSEPIPE_OT_build_rig,
    POSEPIPE_PT_panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
