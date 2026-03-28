"""Task profiles and threshold configuration for trajectory segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class TaskProfile:
    """Thresholds and geometric references used by segmenter and labeler."""

    name: str
    pick_zone_center: Vec3
    insertion_zone_center: Vec3
    ready_pose: Vec3
    pick_zone_radius_xy: float
    insertion_zone_radius_xy: float
    pick_zone_z_tol: float
    insertion_z_tol: float
    ready_pose_tol: float
    speed_change_threshold: float
    stop_speed_threshold: float
    move_speed_threshold: float
    precision_speed_threshold: float
    gripper_event_hysteresis_s: float
    min_segment_duration_s: float
    hysteresis_duration_s: float
    alignment_tolerance: float


TASK_PROFILES: Dict[str, TaskProfile] = {
    "charger_insertion": TaskProfile(
        name="charger_insertion",
        pick_zone_center=(0.38, 0.12, 0.07),
        insertion_zone_center=(0.61, -0.05, 0.02),
        ready_pose=(0.31, 0.00, 0.22),
        pick_zone_radius_xy=0.10,
        insertion_zone_radius_xy=0.08,
        pick_zone_z_tol=0.06,
        insertion_z_tol=0.04,
        ready_pose_tol=0.09,
        speed_change_threshold=0.15,
        stop_speed_threshold=0.015,
        move_speed_threshold=0.04,
        precision_speed_threshold=0.02,
        gripper_event_hysteresis_s=0.20,
        min_segment_duration_s=0.25,
        hysteresis_duration_s=0.15,
        alignment_tolerance=0.12,
    )
}


DEFAULT_TASK_PROFILE = "charger_insertion"


def get_task_profile(name: str = DEFAULT_TASK_PROFILE) -> TaskProfile:
    """Load a known task profile by name."""

    try:
        return TASK_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_PROFILES))
        raise ValueError(f"Unknown task profile '{name}'. Available: {known}") from exc


def merge_profile_overrides(
    profile: TaskProfile, overrides: Mapping[str, float | Vec3] | None = None
) -> TaskProfile:
    """Create a derived profile with selected threshold overrides."""

    if not overrides:
        return profile
    allowed = {field.name for field in profile.__dataclass_fields__.values()}
    invalid = sorted(set(overrides) - allowed)
    if invalid:
        raise ValueError(f"Invalid profile override keys: {', '.join(invalid)}")
    payload = {**profile.__dict__, **dict(overrides)}
    return TaskProfile(**payload)
