"""Segmentation package."""

from .config import DEFAULT_TASK_PROFILE, TASK_PROFILES, TaskProfile, get_task_profile
from .pipeline import run_pipeline
from .segmenter import (
    Segment,
    boundaries_to_segments,
    detect_boundaries,
    label_segments,
    segment_trajectory,
)

__all__ = [
    "DEFAULT_TASK_PROFILE",
    "TASK_PROFILES",
    "TaskProfile",
    "get_task_profile",
    "run_pipeline",
    "Segment",
    "detect_boundaries",
    "boundaries_to_segments",
    "label_segments",
    "segment_trajectory",
]