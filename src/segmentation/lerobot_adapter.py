"""Schema-aware LeRobot dataset inspection and loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_JOINT_STATE_CANDIDATES = (
    "observation.state",
    "joint_states",
    "state",
    "obs_state",
    "robot_state",
)
_JOINT_CMD_CANDIDATES = (
    "action",
    "joint_commands",
    "commands",
    "target_state",
    "desired_state",
)
_EE_POSE_CANDIDATES = (
    "observation.ee_pose",
    "ee_pose",
    "cartesian_pose",
    "end_effector_pose",
    "eef_pose",
)
_GRIPPER_CANDIDATES = (
    "observation.gripper",
    "gripper",
    "gripper_state",
    "gripper_opening",
    "grip",
    "gripper_ratio",
)
_TIMESTAMP_CANDIDATES = (
    "timestamp",
    "timestamps",
    "time",
    "t",
)
_EPISODE_INDEX_CANDIDATES = (
    "episode_index",
    "episode_id",
    "episode",
)
_FRAME_INDEX_CANDIDATES = (
    "frame_index",
    "step_index",
    "timestep",
    "index",
)


@dataclass
class DatasetSchema:
    """Detected layout of a LeRobot-style dataset."""

    fps: float
    n_episodes: int
    layout: str
    joint_state_key: str | None
    joint_command_key: str | None
    ee_pose_key: str | None
    gripper_key: str | None
    timestamp_key: str | None
    episode_index_key: str | None
    frame_index_key: str | None
    joint_dim: int
    joint_state_names: list[str] = field(default_factory=list)
    joint_command_names: list[str] = field(default_factory=list)
    channel_names_source: dict[str, str] = field(default_factory=dict)
    all_keys: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "DatasetSchema:",
            f"  layout         : {self.layout}",
            f"  fps            : {self.fps}",
            f"  n_episodes     : {self.n_episodes}",
            f"  joint_state    : {self.joint_state_key} (dim={self.joint_dim})",
            f"  joint_command  : {self.joint_command_key}",
            f"  ee_pose        : {self.ee_pose_key}",
            f"  gripper        : {self.gripper_key}",
            f"  timestamp      : {self.timestamp_key}",
            f"  episode_index  : {self.episode_index_key}",
            f"  frame_index    : {self.frame_index_key}",
            f"  all_keys       : {self.all_keys}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "n_episodes": self.n_episodes,
            "layout": self.layout,
            "joint_state_key": self.joint_state_key,
            "joint_command_key": self.joint_command_key,
            "ee_pose_key": self.ee_pose_key,
            "gripper_key": self.gripper_key,
            "timestamp_key": self.timestamp_key,
            "episode_index_key": self.episode_index_key,
            "frame_index_key": self.frame_index_key,
            "joint_dim": self.joint_dim,
            "joint_state_names": self.joint_state_names,
            "joint_command_names": self.joint_command_names,
            "channel_names_source": self.channel_names_source,
            "all_keys": self.all_keys,
        }


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required for LeRobot dataset inspection/loading. "
            "Install with: pip install pandas pyarrow"
        ) from exc
    return pd


def _read_info_json(dataset_path: Path) -> dict[str, Any]:
    for candidate in (dataset_path / "meta" / "info.json", dataset_path / "info.json"):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _first_match(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    col_set = set(columns)
    for candidate in candidates:
        if candidate in col_set:
            return candidate
    return None


def _discover_layout(dataset_path: Path) -> tuple[str, list[Path]]:
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        return "unknown", []

    per_episode = sorted(data_dir.glob("chunk-*/episode_*.parquet")) + sorted(
        data_dir.glob("episode_*.parquet")
    )
    if per_episode:
        return "per_episode_parquet", per_episode

    shared = sorted(data_dir.glob("chunk-*/file-*.parquet")) + sorted(
        data_dir.glob("file-*.parquet")
    )
    if shared:
        return "shared_parquet", shared

    any_parquet = sorted(data_dir.glob("**/*.parquet"))
    if any_parquet:
        return "shared_parquet", any_parquet
    return "unknown", []


def _read_first_dataframe(parquet_files: list[Path]) -> Any:
    pd = _import_pandas()
    if not parquet_files:
        return None
    return pd.read_parquet(parquet_files[0])


def _infer_joint_dim(df: Any, key: str | None) -> int:
    if not key or key not in df.columns:
        return 0
    try:
        first = df[key].iloc[0]
        arr = np.asarray(first)
        if arr.ndim == 0:
            return 1
        return int(arr.shape[-1] if arr.ndim > 1 else arr.shape[0])
    except Exception:
        return 0


def _default_names(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(dim)]


def _feature_entry_names(info: dict[str, Any], key: str | None, dim: int, prefix: str) -> tuple[list[str], str]:
    if not key:
        return [], "missing"
    features = info.get("features", {})
    entry = features.get(key) if isinstance(features, dict) else None
    if isinstance(entry, dict):
        names = entry.get("names")
        if isinstance(names, list) and len(names) == dim:
            return [str(name) for name in names], "info_json"
    return _default_names(prefix, dim), "generated"


def _count_metadata_episodes(dataset_path: Path) -> int | None:
    for ep_meta in (
        dataset_path / "meta" / "episodes.jsonl",
        dataset_path / "meta" / "episodes.json",
        dataset_path / "episodes.jsonl",
        dataset_path / "episodes.json",
    ):
        if not ep_meta.exists():
            continue
        if ep_meta.suffix == ".jsonl":
            with ep_meta.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        with ep_meta.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        for key in ("episodes", "data", "items"):
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, list):
                return len(value)
    return None


def _scan_distinct_episode_ids(parquet_files: list[Path], episode_index_key: str | None) -> list[int | str]:
    if not episode_index_key:
        return []
    pd = _import_pandas()
    episode_ids: set[int | str] = set()
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path, columns=[episode_index_key])
        if episode_index_key not in df.columns:
            continue
        for value in df[episode_index_key].dropna().tolist():
            episode_ids.add(_to_python_scalar(value))
    return sorted(episode_ids, key=_episode_sort_key)


def _count_episodes(
    dataset_path: Path,
    layout: str,
    parquet_files: list[Path],
    episode_index_key: str | None,
    info: dict[str, Any],
) -> int:
    metadata_count = _count_metadata_episodes(dataset_path)
    if metadata_count is not None:
        return metadata_count

    if layout == "shared_parquet":
        distinct_ids = _scan_distinct_episode_ids(parquet_files, episode_index_key)
        if distinct_ids:
            return len(distinct_ids)

    total_episodes = info.get("total_episodes")
    if isinstance(total_episodes, int):
        return total_episodes

    if layout == "per_episode_parquet":
        return len(parquet_files)
    return len(_scan_distinct_episode_ids(parquet_files, episode_index_key))


def inspect_dataset(dataset_path: str | Path) -> DatasetSchema:
    """Inspect a LeRobot-style dataset and return a detected schema."""
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    info = _read_info_json(root)
    fps = float(info.get("fps", 30.0))
    layout, parquet_files = _discover_layout(root)
    if not parquet_files and not info:
        raise FileNotFoundError(f"No parquet data or info.json found under {root}")

    df = _read_first_dataframe(parquet_files) if parquet_files else None
    all_keys = [str(col) for col in df.columns.tolist()] if df is not None else []

    joint_state_key = _first_match(all_keys, _JOINT_STATE_CANDIDATES)
    joint_command_key = _first_match(all_keys, _JOINT_CMD_CANDIDATES)
    ee_pose_key = _first_match(all_keys, _EE_POSE_CANDIDATES)
    gripper_key = _first_match(all_keys, _GRIPPER_CANDIDATES)
    timestamp_key = _first_match(all_keys, _TIMESTAMP_CANDIDATES)
    episode_index_key = _first_match(all_keys, _EPISODE_INDEX_CANDIDATES)
    frame_index_key = _first_match(all_keys, _FRAME_INDEX_CANDIDATES)

    if joint_state_key is None and isinstance(info.get("features"), dict):
        feature_keys = list(info["features"].keys())
        joint_state_key = _first_match(feature_keys, _JOINT_STATE_CANDIDATES)
        joint_command_key = joint_command_key or _first_match(feature_keys, _JOINT_CMD_CANDIDATES)
        ee_pose_key = ee_pose_key or _first_match(feature_keys, _EE_POSE_CANDIDATES)
        gripper_key = gripper_key or _first_match(feature_keys, _GRIPPER_CANDIDATES)
        timestamp_key = timestamp_key or _first_match(feature_keys, _TIMESTAMP_CANDIDATES)

    joint_dim = _infer_joint_dim(df, joint_state_key) if df is not None else 0
    joint_state_names, joint_state_names_source = _feature_entry_names(
        info,
        joint_state_key,
        joint_dim,
        "state",
    )
    joint_command_names, joint_command_names_source = _feature_entry_names(
        info,
        joint_command_key,
        joint_dim,
        "action",
    )

    n_episodes = _count_episodes(root, layout, parquet_files, episode_index_key, info)
    schema = DatasetSchema(
        fps=fps,
        n_episodes=n_episodes,
        layout=layout,
        joint_state_key=joint_state_key,
        joint_command_key=joint_command_key,
        ee_pose_key=ee_pose_key,
        gripper_key=gripper_key,
        timestamp_key=timestamp_key,
        episode_index_key=episode_index_key,
        frame_index_key=frame_index_key,
        joint_dim=joint_dim,
        joint_state_names=joint_state_names,
        joint_command_names=joint_command_names,
        channel_names_source={
            "joint_state": joint_state_names_source,
            "joint_command": joint_command_names_source,
        },
        all_keys=all_keys,
    )
    print(schema)
    return schema


def list_episode_refs(dataset_path: str | Path, schema: DatasetSchema | None = None) -> list[Any]:
    """Return ``EpisodeRef`` objects for either LeRobot parquet layout."""
    from .data_loader import EpisodeRef

    root = Path(dataset_path).expanduser().resolve()
    resolved_schema = schema or inspect_dataset(root)

    metadata_rows = _load_episode_metadata_rows(root)
    metadata_by_id = {
        _extract_episode_id(row, idx): row
        for idx, row in enumerate(metadata_rows)
        if isinstance(row, dict)
    }

    if resolved_schema.layout == "per_episode_parquet":
        _, parquet_files = _discover_layout(root)
        episode_ids = [_episode_id_from_path(path) for path in parquet_files]
    elif resolved_schema.layout == "shared_parquet":
        _, parquet_files = _discover_layout(root)
        episode_ids = _scan_distinct_episode_ids(parquet_files, resolved_schema.episode_index_key)
    else:
        episode_ids = list(metadata_by_id.keys())

    return [
        EpisodeRef(
            dataset_path=root,
            episode_id=episode_id,
            length=_extract_episode_length(metadata_by_id.get(episode_id, {})),
            metadata=metadata_by_id.get(episode_id),
        )
        for episode_id in sorted(set(episode_ids), key=_episode_sort_key)
    ]


def _load_episode_metadata_rows(dataset_path: Path) -> list[dict[str, Any]]:
    for candidate in (
        dataset_path / "meta" / "episodes.jsonl",
        dataset_path / "meta" / "episodes.json",
        dataset_path / "episodes.jsonl",
        dataset_path / "episodes.json",
    ):
        if not candidate.exists():
            continue
        if candidate.suffix == ".jsonl":
            rows = []
            with candidate.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        with candidate.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("episodes", "data", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
    return []


def _extract_episode_id(meta: dict[str, Any], fallback_idx: int) -> int | str:
    for key in ("episode_id", "episode_index", "id", "episode"):
        if key in meta:
            return _to_python_scalar(meta[key])
    return fallback_idx


def _extract_episode_length(meta: dict[str, Any]) -> int | None:
    if not isinstance(meta, dict):
        return None
    for key in ("length", "num_frames", "num_steps", "size"):
        value = meta.get(key)
        if isinstance(value, int):
            return value
    return None


def _episode_id_from_path(path: Path) -> int | str:
    match = re.search(r"episode_(.+)\.parquet$", path.name)
    if not match:
        return path.stem
    token = match.group(1)
    return int(token) if token.isdigit() else token


def _episode_sort_key(value: int | str) -> tuple[int, Any]:
    return (0, value) if isinstance(value, int) else (1, str(value))


def _to_python_scalar(value: Any) -> int | str:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_array_from_df(df: Any, key: str) -> np.ndarray:
    col = df[key]
    first = col.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.asarray(col.tolist(), dtype=np.float64)
    return col.to_numpy(dtype=np.float64)


def _episode_filename_candidates(episode_id: int | str) -> list[str]:
    token = str(episode_id)
    names = [f"episode_{token}.parquet"]
    if isinstance(episode_id, int):
        names.insert(0, f"episode_{episode_id:06d}.parquet")
    return names


def _load_per_episode_dataframe(dataset_path: Path, episode_id: int | str) -> Any:
    pd = _import_pandas()
    data_root = dataset_path / "data"
    candidates: list[Path] = []
    for filename in _episode_filename_candidates(episode_id):
        candidates.extend(sorted(data_root.glob(f"chunk-*/{filename}")))
        candidates.append(data_root / filename)

    for candidate in candidates:
        if candidate.exists():
            return pd.read_parquet(candidate)

    raise FileNotFoundError(f"Could not find parquet for episode {episode_id} under {data_root}")


def _load_shared_episode_dataframe(
    dataset_path: Path,
    episode_id: int | str,
    episode_index_key: str | None,
) -> Any:
    if not episode_index_key:
        raise KeyError("shared parquet layout requires an episode index column")

    pd = _import_pandas()
    _, parquet_files = _discover_layout(dataset_path)
    for parquet_path in parquet_files:
        try:
            df = pd.read_parquet(
                parquet_path,
                filters=[(episode_index_key, "==", episode_id)],
            )
        except Exception:
            df = pd.read_parquet(parquet_path)
            if episode_index_key not in df.columns:
                continue
            df = df[df[episode_index_key] == episode_id]

        if episode_index_key in df.columns:
            df = df[df[episode_index_key] == episode_id]
        if not df.empty:
            return df.reset_index(drop=True)

    raise FileNotFoundError(
        f"Could not find rows for episode_id={episode_id} in shared parquet dataset {dataset_path}"
    )


def load_episode_arrays(
    ep_ref: Any,
    schema: DatasetSchema,
    fill_policy: str = "ffill_drop",
) -> dict[str, np.ndarray]:
    """Load one episode and return a standardised ``dict[str, np.ndarray]``."""
    dataset_path = Path(ep_ref.dataset_path).expanduser().resolve()
    if schema.layout == "per_episode_parquet":
        df = _load_per_episode_dataframe(dataset_path, ep_ref.episode_id)
    elif schema.layout == "shared_parquet":
        df = _load_shared_episode_dataframe(dataset_path, ep_ref.episode_id, schema.episode_index_key)
    else:
        raise FileNotFoundError(f"Unsupported dataset layout: {schema.layout}")

    result: dict[str, np.ndarray] = {}

    if schema.joint_state_key is None or schema.joint_state_key not in df.columns:
        raise KeyError(
            f"Joint state key {schema.joint_state_key!r} not found. "
            f"Available columns: {df.columns.tolist()}"
        )
    result["joint_states"] = _extract_array_from_df(df, schema.joint_state_key)

    if schema.joint_command_key and schema.joint_command_key in df.columns:
        result["joint_commands"] = _extract_array_from_df(df, schema.joint_command_key)

    if schema.ee_pose_key and schema.ee_pose_key in df.columns:
        result["ee_pose"] = _extract_array_from_df(df, schema.ee_pose_key)

    if schema.gripper_key and schema.gripper_key in df.columns:
        gripper = _extract_array_from_df(df, schema.gripper_key)
        result["gripper"] = gripper[:, None] if gripper.ndim == 1 else gripper

    if schema.timestamp_key and schema.timestamp_key in df.columns:
        result["timestamps"] = df[schema.timestamp_key].to_numpy(dtype=np.float64)
    elif schema.frame_index_key and schema.frame_index_key in df.columns:
        if schema.fps <= 0:
            raise ValueError("fps must be > 0 to reconstruct timestamps from frame index")
        frame_index = df[schema.frame_index_key].to_numpy(dtype=np.float64)
        result["timestamps"] = frame_index / schema.fps
    else:
        raise KeyError(
            "No timestamp key or frame index key found; cannot align episode timeseries."
        )

    return _clean_episode_arrays(result, fill_policy)


def _clean_episode_arrays(
    arrays: dict[str, np.ndarray],
    fill_policy: str,
) -> dict[str, np.ndarray]:
    if fill_policy not in {"ffill_drop", "none"}:
        raise ValueError(f"unsupported fill_policy: {fill_policy!r}")

    cleaned: dict[str, np.ndarray] = {}
    if fill_policy == "none":
        cleaned = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    else:
        cleaned = {
            key: _forward_fill_preserve_leading(np.asarray(value, dtype=np.float64))
            for key, value in arrays.items()
        }

    valid_mask: np.ndarray | None = None
    for value in cleaned.values():
        value_2d = value[:, None] if value.ndim == 1 else value
        row_valid = ~np.isnan(value_2d).any(axis=1)
        valid_mask = row_valid if valid_mask is None else (valid_mask & row_valid)

    if valid_mask is None or not np.any(valid_mask):
        raise ValueError("Episode contains no valid rows after NaN handling")

    return {
        key: value[valid_mask]
        for key, value in cleaned.items()
    }


def _forward_fill_preserve_leading(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]

    out = x.copy()
    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        mask = np.isnan(col)
        if not mask.any():
            continue

        valid_idx = np.where(~mask)[0]
        if len(valid_idx) == 0:
            continue

        carry_idx = np.where(~mask, np.arange(len(col)), 0)
        np.maximum.accumulate(carry_idx, out=carry_idx)
        filled = col[carry_idx]
        filled[: valid_idx[0]] = np.nan
        out[:, col_idx] = filled

    return out[:, 0] if squeeze else out
