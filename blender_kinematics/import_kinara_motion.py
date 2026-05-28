from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bpy
from mathutils import Vector

from blender_kinematics.kinara_motion import MIN_BONE_LENGTH, MotionClip, PersonMotion, build_held_joint_frames, load_motion_clip


def _vector_from_joint(joint_value: dict[str, float]) -> Vector:
    return Vector((float(joint_value["x"]), float(joint_value["y"]), float(joint_value["z"])))


def _ensure_collection(collection_name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(collection_name)
    if collection is not None:
        return collection
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def _set_scene_fps(fps: float) -> None:
    if fps <= 0.0:
        return
    rounded_fps = max(1, int(round(fps)))
    bpy.context.scene.render.fps = rounded_fps
    bpy.context.scene.render.fps_base = rounded_fps / fps if fps else 1.0


def _clear_object_selection() -> None:
    for selected_object in list(bpy.context.selected_objects):
        selected_object.select_set(False)


def _create_armature_object(collection: bpy.types.Collection, object_name: str) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new(f"{object_name}Data")
    armature_data.display_type = "STICK"
    armature_object = bpy.data.objects.new(object_name, armature_data)
    collection.objects.link(armature_object)
    return armature_object


def _root_tail(rest_joints: dict[str, dict[str, float]]) -> Vector:
    hips = _vector_from_joint(rest_joints["HipsRoot"])
    chest_joint = rest_joints.get("Chest")
    if chest_joint is not None:
        chest = _vector_from_joint(chest_joint)
        delta = chest - hips
        if delta.length >= MIN_BONE_LENGTH:
            return delta.normalized() * max(delta.length * 0.25, MIN_BONE_LENGTH)
    return Vector((0.0, 0.0, 0.25))


def _fallback_tail(head: Vector, joint_name: str) -> Vector:
    if "Shoulder" in joint_name or "Elbow" in joint_name or "Wrist" in joint_name:
        direction = Vector((-1.0, 0.0, 0.0) if joint_name.startswith("Left") else (1.0, 0.0, 0.0))
    elif "Hip" in joint_name or "Knee" in joint_name or "Ankle" in joint_name:
        direction = Vector((0.0, 0.0, -1.0))
    elif "Foot" in joint_name or "Toe" in joint_name:
        direction = Vector((0.0, 1.0, 0.0))
    else:
        direction = Vector((0.0, 0.0, 1.0))
    return head + direction.normalized() * MIN_BONE_LENGTH


def _build_armature_rig(
    armature_object: bpy.types.Object,
    motion: PersonMotion,
    skeleton,
) -> None:
    bpy.context.view_layer.objects.active = armature_object
    _clear_object_selection()
    armature_object.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_object.data.edit_bones
    root_bone = edit_bones.new("KinaraRoot")
    root_bone.head = Vector((0.0, 0.0, 0.0))
    root_bone.tail = _root_tail(motion.rest_joints)
    root_bone.use_deform = False

    for joint in skeleton:
        if joint.parent is None:
            continue
        bone = edit_bones.new(joint.name)
        head = _vector_from_joint(motion.rest_joints[joint.parent])
        tail = _vector_from_joint(motion.rest_joints[joint.name])
        if (tail - head).length < MIN_BONE_LENGTH:
            tail = _fallback_tail(head, joint.name)
        bone.head = head
        bone.tail = tail
        bone.use_connect = False

    for joint in skeleton:
        if joint.parent is None:
            continue
        bone = edit_bones[joint.name]
        bone.parent = root_bone if joint.parent == "HipsRoot" else edit_bones[joint.parent]
        bone.use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    for pose_bone in armature_object.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"


def _create_joint_targets(
    collection: bpy.types.Collection,
    prefix: str,
    motion: PersonMotion,
    skeleton,
) -> dict[str, bpy.types.Object]:
    targets: dict[str, bpy.types.Object] = {}
    held_frames = build_held_joint_frames(motion.frames, skeleton, motion.rest_joints)

    for joint in skeleton:
        target = bpy.data.objects.new(f"{prefix}_{joint.name}_target", None)
        target.empty_display_type = "PLAIN_AXES"
        target.empty_display_size = 0.02
        target.hide_render = True
        target.hide_viewport = True
        collection.objects.link(target)
        targets[joint.name] = target

    frame_numbers = [frame_index + 1 for frame_index in motion.frame_indices]
    if not frame_numbers:
        frame_numbers = list(range(1, len(held_frames) + 1))

    for frame_number, frame in zip(frame_numbers, held_frames):
        for joint in skeleton:
            target = targets[joint.name]
            target.location = _vector_from_joint(frame[joint.name])
            target.keyframe_insert(data_path="location", frame=frame_number)

    return targets


def _apply_constraints(
    armature_object: bpy.types.Object,
    targets: dict[str, bpy.types.Object],
    skeleton,
) -> None:
    root_bone = armature_object.pose.bones["KinaraRoot"]
    root_copy = root_bone.constraints.new(type="COPY_LOCATION")
    root_copy.target = targets["HipsRoot"]

    for joint in skeleton:
        if joint.parent is None:
            continue
        pose_bone = armature_object.pose.bones[joint.name]
        track = pose_bone.constraints.new(type="DAMPED_TRACK")
        track.target = targets[joint.name]
        track.track_axis = "TRACK_Y"


def _create_action(armature_object: bpy.types.Object, action_name: str) -> bpy.types.Action:
    animation_data = armature_object.animation_data_create()
    action = bpy.data.actions.new(action_name)
    animation_data.action = action
    return action


def _bake_animation(armature_object: bpy.types.Object, frame_start: int, frame_end: int) -> None:
    bpy.context.view_layer.objects.active = armature_object
    _clear_object_selection()
    armature_object.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        step=1,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=True,
        bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def _delete_targets(targets: dict[str, bpy.types.Object]) -> None:
    for target in targets.values():
        bpy.data.objects.remove(target, do_unlink=True)


def import_motion_clip(clip: MotionClip) -> list[bpy.types.Object]:
    _set_scene_fps(clip.fps)
    import_collection = _ensure_collection(f"KinaraImport_{clip.source_path.stem}")
    imported_armatures: list[bpy.types.Object] = []

    for motion in clip.people:
        object_name = f"Kinara_{motion.label}"
        armature_object = _create_armature_object(import_collection, object_name)
        _build_armature_rig(armature_object, motion, clip.skeleton)
        targets = _create_joint_targets(import_collection, object_name, motion, clip.skeleton)
        _apply_constraints(armature_object, targets, clip.skeleton)
        _create_action(armature_object, f"{object_name}_Action")

        frame_numbers = [frame_index + 1 for frame_index in motion.frame_indices]
        frame_start = min(frame_numbers) if frame_numbers else 1
        frame_end = max(frame_numbers) if frame_numbers else max(1, clip.frame_count)
        _bake_animation(armature_object, frame_start, frame_end)
        _delete_targets(targets)
        imported_armatures.append(armature_object)

    return imported_armatures


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Kinara motion JSON into Blender as an animated armature.")
    parser.add_argument("--input", required=True, help="Path to a Kinara motion JSON file.")
    parser.add_argument(
        "--person",
        default="all",
        help="Which person to import from a multi-person file. Use 'all' or a person label.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = list(sys.argv[sys.argv.index("--") + 1 :]) if "--" in sys.argv else []
    args = _parse_cli_args(argv)
    clip = load_motion_clip(args.input, person_filter=args.person)
    imported_armatures = import_motion_clip(clip)
    print(f"Imported {len(imported_armatures)} Kinara armature(s) from {clip.source_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
