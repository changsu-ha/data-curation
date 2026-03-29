from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="pandas is required for adapter tests")
pytest.importorskip("pyarrow", reason="pyarrow is required for parquet adapter tests")


def _write_info_json(tmp_dir: Path, fps: float, state_dim: int, action_dim: int) -> None:
    meta_dir = tmp_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "fps": fps,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": [f"joint_{i}" for i in range(state_dim)]},
            "action": {"dtype": "float32", "shape": [action_dim], "names": [f"cmd_{i}" for i in range(action_dim)]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _make_per_episode_dataset(
    tmp_dir: Path,
    fps: float = 30.0,
    n_episodes: int = 3,
    length: int = 8,
    joint_dim: int = 4,
    include_ee_pose: bool = True,
) -> None:
    _write_info_json(tmp_dir, fps=fps, state_dim=joint_dim, action_dim=joint_dim)
    meta_path = tmp_dir / "meta" / "episodes.jsonl"
    meta_path.write_text(
        "\n".join(json.dumps({"episode_id": idx, "length": length}) for idx in range(n_episodes)),
        encoding="utf-8",
    )

    data_dir = tmp_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for episode_id in range(n_episodes):
        rows: dict[str, object] = {
            "timestamp": np.arange(length, dtype=float) / fps,
            "observation.state": [rng.normal(size=joint_dim).tolist() for _ in range(length)],
            "action": [rng.normal(size=joint_dim).tolist() for _ in range(length)],
        }
        if include_ee_pose:
            rows["ee_pose"] = [rng.normal(size=7).tolist() for _ in range(length)]
        pd.DataFrame(rows).to_parquet(data_dir / f"episode_{episode_id:06d}.parquet", index=False)


def _make_shared_dataset(
    tmp_dir: Path,
    fps: float = 20.0,
    episode_lengths: tuple[int, ...] = (5, 7),
    joint_dim: int = 3,
    include_timestamp: bool = True,
    include_action: bool = True,
) -> None:
    _write_info_json(tmp_dir, fps=fps, state_dim=joint_dim, action_dim=joint_dim)
    data_dir = tmp_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    rows: dict[str, list[object]] = {
        "episode_index": [],
        "frame_index": [],
        "observation.state": [],
    }
    if include_timestamp:
        rows["timestamp"] = []
    if include_action:
        rows["action"] = []

    for episode_id, length in enumerate(episode_lengths):
        for frame_index in range(length):
            rows["episode_index"].append(episode_id)
            rows["frame_index"].append(frame_index)
            if include_timestamp:
                rows["timestamp"].append(frame_index / fps)
            rows["observation.state"].append(rng.normal(size=joint_dim).tolist())
            if include_action:
                rows["action"].append(rng.normal(size=joint_dim).tolist())

    pd.DataFrame(rows).to_parquet(data_dir / "file-000.parquet", index=False)


def _episode_ref(tmp_path: Path, episode_id: int):
    from segmentation.data_loader import EpisodeRef

    return EpisodeRef(dataset_path=tmp_path, episode_id=episode_id)


class TestInspectDataset:
    def test_detects_per_episode_layout(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset

        _make_per_episode_dataset(tmp_path, n_episodes=4)
        schema = inspect_dataset(tmp_path)

        assert schema.layout == "per_episode_parquet"
        assert schema.n_episodes == 4
        assert schema.joint_state_key == "observation.state"
        assert schema.joint_command_key == "action"

    def test_detects_shared_layout_and_counts_distinct_episodes(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset

        _make_shared_dataset(tmp_path, episode_lengths=(3, 5, 4))
        schema = inspect_dataset(tmp_path)

        assert schema.layout == "shared_parquet"
        assert schema.n_episodes == 3
        assert schema.episode_index_key == "episode_index"
        assert schema.frame_index_key == "frame_index"

    def test_reads_channel_names_from_info_json(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset

        _make_shared_dataset(tmp_path, joint_dim=2)
        schema = inspect_dataset(tmp_path)

        assert schema.joint_state_names == ["joint_0", "joint_1"]
        assert schema.joint_command_names == ["cmd_0", "cmd_1"]
        assert schema.channel_names_source["joint_state"] == "info_json"


class TestListEpisodeRefs:
    def test_lists_shared_episode_ids(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset, list_episode_refs

        _make_shared_dataset(tmp_path, episode_lengths=(2, 4, 3))
        schema = inspect_dataset(tmp_path)
        episode_refs = list_episode_refs(tmp_path, schema)

        assert [ref.episode_id for ref in episode_refs] == [0, 1, 2]


class TestLoadEpisodeArrays:
    def test_loads_per_episode_arrays(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset, load_episode_arrays

        _make_per_episode_dataset(tmp_path, length=6, joint_dim=5)
        schema = inspect_dataset(tmp_path)
        arrays = load_episode_arrays(_episode_ref(tmp_path, 0), schema)

        assert arrays["joint_states"].shape == (6, 5)
        assert arrays["joint_commands"].shape == (6, 5)
        assert arrays["timestamps"].shape == (6,)
        assert arrays["ee_pose"].shape == (6, 7)

    def test_loads_shared_episode_by_episode_index(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset, load_episode_arrays

        _make_shared_dataset(tmp_path, episode_lengths=(4, 6), joint_dim=3)
        schema = inspect_dataset(tmp_path)
        arrays = load_episode_arrays(_episode_ref(tmp_path, 1), schema)

        assert arrays["joint_states"].shape == (6, 3)
        assert arrays["joint_commands"].shape == (6, 3)
        assert arrays["timestamps"].shape == (6,)
        assert arrays["timestamps"][0] == pytest.approx(0.0)

    def test_reconstructs_timestamps_from_frame_index(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset, load_episode_arrays

        _make_shared_dataset(tmp_path, fps=25.0, episode_lengths=(5,), include_timestamp=False)
        schema = inspect_dataset(tmp_path)
        arrays = load_episode_arrays(_episode_ref(tmp_path, 0), schema)

        assert arrays["timestamps"].tolist() == pytest.approx([0.0, 0.04, 0.08, 0.12, 0.16])

    def test_forward_fill_and_drop_leading_nan_rows(self, tmp_path: Path):
        from segmentation.lerobot_adapter import inspect_dataset, load_episode_arrays

        _make_shared_dataset(tmp_path, episode_lengths=(4,), joint_dim=2)
        parquet_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(parquet_path)
        df.at[0, "observation.state"] = [np.nan, np.nan]
        df.at[1, "observation.state"] = [1.0, 2.0]
        df.to_parquet(parquet_path, index=False)

        schema = inspect_dataset(tmp_path)
        arrays = load_episode_arrays(_episode_ref(tmp_path, 0), schema, fill_policy="ffill_drop")

        assert arrays["joint_states"].shape[0] == 3
        assert arrays["joint_states"][0].tolist() == pytest.approx([1.0, 2.0])
