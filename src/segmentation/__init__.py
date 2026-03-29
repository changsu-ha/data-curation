"""Segmentation package."""

from .config import DEFAULT_TASK_PROFILE, TASK_PROFILES, TaskProfile, get_task_profile
from .data_loader import (
    EpisodeRef,
    EpisodeTimeseries,
    compute_episode_fk,
    list_episodes,
    load_episode,
    save_sampling_output,
    uniform_sample_episodes,
)
from .features import FeatureBuildConfig, build_features, normalize_modality_name
from .lerobot_adapter import DatasetSchema, inspect_dataset, list_episode_refs, load_episode_arrays
from .pipeline import run_pipeline
from .segmenter import (
    Segment,
    boundaries_to_segments,
    detect_boundaries,
    label_segments,
    segment_trajectory,
)

# kinematics is intentionally NOT imported here to avoid triggering a pinocchio
# import attempt on every ``import segmentation``.  Access it via:
#   from segmentation import kinematics
#   from segmentation.kinematics import load_robot_model, compute_fk_trajectory

__all__ = [
    # config
    "DEFAULT_TASK_PROFILE",
    "TASK_PROFILES",
    "TaskProfile",
    "get_task_profile",
    # data_loader
    "EpisodeRef",
    "EpisodeTimeseries",
    "compute_episode_fk",
    "list_episodes",
    "load_episode",
    "save_sampling_output",
    "uniform_sample_episodes",
    # features
    "FeatureBuildConfig",
    "build_features",
    "normalize_modality_name",
    # lerobot adapter
    "DatasetSchema",
    "inspect_dataset",
    "list_episode_refs",
    "load_episode_arrays",
    # pipeline
    "run_pipeline",
    # segmenter
    "Segment",
    "detect_boundaries",
    "boundaries_to_segments",
    "label_segments",
    "segment_trajectory",
]
