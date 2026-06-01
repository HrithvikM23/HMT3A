from __future__ import annotations

from typing import Any

from utils.color_profile import color_profile_similarity

PersonTrack = Any


def align_people_across_cameras(
    camera_tracks: dict[str, list[PersonTrack]],
    reference_label: str,
) -> dict[str, dict[str, PersonTrack]]:
    grouped: dict[str, dict[str, PersonTrack]] = {}
    reference_tracks = sorted(camera_tracks.get(reference_label, []), key=_track_sort_key)
    reference_keys: list[str] = []

    for index, track in enumerate(reference_tracks):
        key = _person_key(track, index)
        reference_keys.append(key)
        grouped.setdefault(key, {})[reference_label] = track

    for camera_label, tracks in camera_tracks.items():
        if camera_label == reference_label:
            continue

        remaining_tracks = _assign_labeled_tracks(grouped, camera_label, sorted(tracks, key=_track_sort_key))
        open_reference_keys = [key for key in reference_keys if camera_label not in grouped.get(key, {})]
        remaining_tracks = _assign_by_color_similarity(grouped, camera_label, remaining_tracks, open_reference_keys, reference_label)
        _assign_leftover_tracks(grouped, camera_label, remaining_tracks, reference_keys)

    return grouped


def _track_sort_key(track: PersonTrack) -> tuple[int, float]:
    return (0 if track.label else 1, track.center[0])


def _person_key(track: PersonTrack, fallback_index: int) -> str:
    if track.label:
        return track.label
    if track.id > 0:
        return f"person{track.id}"
    return f"person{fallback_index + 1}"


def _assign_labeled_tracks(
    grouped: dict[str, dict[str, PersonTrack]],
    camera_label: str,
    tracks: list[PersonTrack],
) -> list[PersonTrack]:
    remaining_tracks: list[PersonTrack] = []
    for track in tracks:
        if track.label and track.label in grouped and camera_label not in grouped[track.label]:
            grouped[track.label][camera_label] = track
            continue
        remaining_tracks.append(track)
    return remaining_tracks


def _assign_by_color_similarity(
    grouped: dict[str, dict[str, PersonTrack]],
    camera_label: str,
    tracks: list[PersonTrack],
    open_reference_keys: list[str],
    reference_label: str,
) -> list[PersonTrack]:
    if not tracks or not open_reference_keys:
        return tracks

    scored_pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for key_index, key in enumerate(open_reference_keys):
            reference_track = grouped.get(key, {}).get(reference_label)
            if reference_track is None:
                continue
            scored_pairs.append(
                (
                    color_profile_similarity(reference_track.color_signature, track.color_signature),
                    track_index,
                    key_index,
                )
            )

    used_track_indices: set[int] = set()
    used_key_indices: set[int] = set()
    for _, track_index, key_index in sorted(scored_pairs, reverse=True):
        if track_index in used_track_indices or key_index in used_key_indices:
            continue
        key = open_reference_keys[key_index]
        grouped.setdefault(key, {})[camera_label] = tracks[track_index]
        used_track_indices.add(track_index)
        used_key_indices.add(key_index)

    return [track for track_index, track in enumerate(tracks) if track_index not in used_track_indices]


def _assign_leftover_tracks(
    grouped: dict[str, dict[str, PersonTrack]],
    camera_label: str,
    tracks: list[PersonTrack],
    reference_keys: list[str],
) -> None:
    open_reference_keys = [key for key in reference_keys if camera_label not in grouped.get(key, {})]
    for key, track in zip(open_reference_keys, tracks, strict=False):
        grouped.setdefault(key, {})[camera_label] = track

    extra_index = 0
    for track in tracks[len(open_reference_keys):]:
        while f"person{len(reference_keys) + extra_index + 1}" in grouped:
            extra_index += 1
        key = f"person{len(reference_keys) + extra_index + 1}"
        grouped.setdefault(key, {})[camera_label] = track
        extra_index += 1
