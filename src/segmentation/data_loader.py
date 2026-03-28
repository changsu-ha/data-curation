"""Utilities for indexing, sampling and loading LeRobot episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Iterable
import warnings


@dataclass(frozen=True)
class EpisodeRef:
    """Reference to an episode in a LeRobot-style dataset."""

    dataset_path: Path
    episode_id: int | str
    length: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeTimeseries:
    """Timeseries tensors/arrays loaded for an episode."""

    episode_ref: EpisodeRef
    joint_states: Any
    joint_commands: Any
    ee_pose: Any | None
    needs_fk: bool


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_episode_metadata(dataset_path: Path) -> list[dict[str, Any]]:
    """Resolve episode metadata from common LeRobot metadata files."""

    candidates: tuple[Path, ...] = (
        dataset_path / "meta" / "episodes.jsonl",
        dataset_path / "meta" / "episodes.json",
        dataset_path / "episodes.jsonl",
        dataset_path / "episodes.json",
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".jsonl":
            return _load_jsonl(candidate)

        loaded = _load_json(candidate)
        if isinstance(loaded, dict):
            for key in ("episodes", "data", "items"):
                value = loaded.get(key)
                if isinstance(value, list):
                    return value
            raise ValueError(
                f"{candidate} exists but does not contain a supported episode list key"
            )
        if isinstance(loaded, list):
            return loaded
        raise ValueError(f"Unsupported episode metadata format in {candidate}")

    raise FileNotFoundError(
        "Could not find episode metadata. Expected one of: "
        "meta/episodes.jsonl, meta/episodes.json, episodes.jsonl, episodes.json"
    )


def _extract_episode_id(meta: dict[str, Any], fallback_idx: int) -> int | str:
    for key in ("episode_id", "episode_index", "id", "episode"):
        if key in meta:
            return meta[key]
    return fallback_idx


def _extract_episode_length(meta: dict[str, Any]) -> int | None:
    for key in ("length", "num_frames", "num_steps", "size"):
        value = meta.get(key)
        if isinstance(value, int):
            return value
    return None


def list_episodes(dataset_path: str | Path) -> list[EpisodeRef]:
    """Index episodes/trajectories from LeRobot metadata.

    Args:
        dataset_path: Root path of the dataset.

    Returns:
        Ordered list of :class:`EpisodeRef`.
    """

    root = Path(dataset_path).expanduser().resolve()
    metadata_rows = _resolve_episode_metadata(root)

    episodes: list[EpisodeRef] = []
    for idx, row in enumerate(metadata_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Episode metadata row {idx} must be a JSON object")
        episodes.append(
            EpisodeRef(
                dataset_path=root,
                episode_id=_extract_episode_id(row, idx),
                length=_extract_episode_length(row),
                metadata=row,
            )
        )

    return episodes


def uniform_sample_episodes(
    episodes: Iterable[EpisodeRef], num_samples: int, seed: int
) -> list[EpisodeRef]:
    """Uniformly sample episodes across the full index range.

    If ``num_samples`` is larger than available episodes, all episodes are returned
    and a warning is emitted.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    episode_list = list(episodes)
    if not episode_list:
        return []

    total = len(episode_list)
    if num_samples >= total:
        if num_samples > total:
            warnings.warn(
                (
                    f"Requested num_samples={num_samples} but only {total} episodes "
                    "are available. Returning all episodes."
                ),
                stacklevel=2,
            )
        return episode_list

    # Pick a deterministic phase to avoid always starting at index 0,
    # while keeping near-uniform interval coverage.
    rng = random.Random(seed)
    stride = total / num_samples
    phase = rng.random() * stride

    selected_indices: list[int] = []
    used: set[int] = set()

    for i in range(num_samples):
        idx = int(phase + i * stride)
        if idx >= total:
            idx = total - 1

        if idx in used:
            # Resolve rare collisions due to integer rounding.
            left = idx
            right = idx
            while True:
                left -= 1
                right += 1
                if left >= 0 and left not in used:
                    idx = left
                    break
                if right < total and right not in used:
                    idx = right
                    break

        used.add(idx)
        selected_indices.append(idx)

    selected_indices.sort()
    return [episode_list[idx] for idx in selected_indices]


def _load_episode_table(path: Path) -> Any | None:
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _load_json(path) if suffix == ".json" else _load_jsonl(path)

    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path)

    if suffix in {".npy", ".npz"}:
        import numpy as np

        return np.load(path, allow_pickle=False)

    return None


def load_episode(episode_ref: EpisodeRef) -> EpisodeTimeseries:
    """Load required timeseries for a single episode.

    Expected keys/files (best-effort, first hit wins):
      - joint states: ``joint_states`` / ``observation.state``
      - joint commands: ``joint_commands`` / ``action``
      - ee pose: ``ee_pose`` / ``observation.ee_pose``

    If end-effector/cartesian pose is missing, ``needs_fk`` is set to ``True``.
    """

    episode_id = episode_ref.episode_id
    data_root = episode_ref.dataset_path / "data"

    candidate_files = (
        data_root / f"episode_{episode_id}.parquet",
        data_root / f"episode_{episode_id}.json",
        data_root / f"episode_{episode_id}.jsonl",
        data_root / f"episode_{episode_id}.npz",
        data_root / f"episode_{episode_id}.npy",
    )

    table = None
    for candidate in candidate_files:
        table = _load_episode_table(candidate)
        if table is not None:
            break

    if table is None:
        raise FileNotFoundError(
            f"Could not find episode data for episode_id={episode_id} under {data_root}"
        )

    def extract(keys: tuple[str, ...]) -> Any | None:
        if hasattr(table, "columns"):  # pandas.DataFrame
            for key in keys:
                if key in table.columns:
                    return table[key].to_list()
            return None

        if isinstance(table, dict):
            for key in keys:
                if key in table:
                    return table[key]
            return None

        if isinstance(table, list):
            # list of records
            for key in keys:
                if table and isinstance(table[0], dict) and key in table[0]:
                    return [row.get(key) for row in table]
            return None

        return None

    joint_states = extract(("joint_states", "observation.state", "state"))
    joint_commands = extract(("joint_commands", "action", "commands"))
    ee_pose = extract(("ee_pose", "observation.ee_pose", "cartesian_pose"))

    if joint_states is None:
        raise KeyError(
            f"Episode {episode_id}: failed to locate joint states in episode data"
        )
    if joint_commands is None:
        raise KeyError(
            f"Episode {episode_id}: failed to locate joint commands in episode data"
        )

    return EpisodeTimeseries(
        episode_ref=episode_ref,
        joint_states=joint_states,
        joint_commands=joint_commands,
        ee_pose=ee_pose,
        needs_fk=ee_pose is None,
    )


def save_sampling_output(output: str | Path, sampled_episodes: Iterable[EpisodeRef]) -> Path:
    """Persist sampled episode ids for reproducibility.

    Args:
        output: Output directory or JSON file path.
        sampled_episodes: Sampled episode references.

    Returns:
        Path to the JSON file that contains sampled episode ids.
    """

    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "selected_episode_ids.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    episode_ids = [episode.episode_id for episode in sampled_episodes]
    payload = {
        "selected_episode_ids": episode_ids,
        "num_selected": len(episode_ids),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path
