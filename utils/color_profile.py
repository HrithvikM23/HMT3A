from __future__ import annotations


ColorProfile = dict[str, float]


def color_profile_similarity(profile_a: ColorProfile, profile_b: ColorProfile) -> float:
    if not profile_a or not profile_b:
        return 0.0
    colors = set(profile_a) | set(profile_b)
    if not colors:
        return 0.0
    overlap = sum(min(profile_a.get(color, 0.0), profile_b.get(color, 0.0)) for color in colors)
    magnitude = sum(max(profile_a.get(color, 0.0), profile_b.get(color, 0.0)) for color in colors)
    if magnitude <= 0:
        return 0.0
    return overlap / magnitude
