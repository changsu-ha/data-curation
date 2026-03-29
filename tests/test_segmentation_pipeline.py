from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _make_joint_sample(length: int = 40, joint_dim: int = 4) -> dict[str, np.ndarray]:
    timestamps = np.linspace(0.0, 1.3, length)
    q = np.column_stack([np.sin(timestamps + offset) for offset in np.linspace(0.0, 0.5, joint_dim)])
    q_cmd = q + 0.05
    return {"timestamps": timestamps, "q": q, "q_cmd": q_cmd}


def _make_cartesian_sample(length: int = 40) -> dict[str, np.ndarray]:
    timestamps = np.linspace(0.0, 1.0, length)
    position = np.column_stack([np.sin(timestamps), np.cos(timestamps), timestamps])
    quaternion = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (length, 1))
    return {"timestamps": timestamps, "position": position, "quaternion": quaternion}


class TestBuildFeatures:
    def test_joint_modality_does_not_require_cartesian(self):
        from segmentation.features import FeatureBuildConfig, build_features

        sample = _make_joint_sample()
        out = build_features(sample, "joint", FeatureBuildConfig(smoothing="none", normalize="none"))

        assert out["matrix"].shape[0] == len(sample["timestamps"])
        assert out["feature_names"][:4] == ["q_0", "q_1", "q_2", "q_3"]
        assert out["source_channels"] == {"joint": ["q"]}

    def test_joint_command_modality_does_not_include_cartesian(self):
        from segmentation.features import FeatureBuildConfig, build_features

        sample = _make_joint_sample()
        out = build_features(sample, "joint_command", FeatureBuildConfig(smoothing="none", normalize="none"))

        assert out["matrix"].shape[1] == (4 * 3) + (4 * 2)
        assert out["source_channels"]["command"] == ["q_cmd", "q_err"]

    def test_ablation_alias_maps_to_joint_command(self):
        from segmentation.features import FeatureBuildConfig, build_features

        sample = _make_joint_sample()
        out = build_features(sample, "ablation", FeatureBuildConfig(smoothing="none", normalize="none"))

        assert out["modality"] == "joint_command"

    def test_cartesian_modality_handles_pose_without_joint_state(self):
        from segmentation.features import FeatureBuildConfig, build_features

        sample = _make_cartesian_sample()
        out = build_features(sample, "cartesian", FeatureBuildConfig(smoothing="none", normalize="none"))

        assert out["matrix"].shape[0] == len(sample["timestamps"])
        assert "cartesian" in out["source_channels"]


class TestEvaluationFormatting:
    def test_fmt_helpers_handle_missing_values(self):
        from segmentation.evaluation import fmt_float, fmt_int, fmt_percent

        assert fmt_float(None) == "N/A"
        assert fmt_float(float("nan")) == "N/A"
        assert fmt_int(None) == "N/A"
        assert fmt_percent(None) == "N/A"

    def test_format_comparison_report_handles_nan_metrics(self):
        from segmentation.evaluation import format_comparison_report

        aggregated = {
            "duration_stats": {
                "pelt": {
                    "n_segments": float("nan"),
                    "mean_s": float("nan"),
                    "std_s": float("nan"),
                    "min_s": float("nan"),
                    "max_s": float("nan"),
                }
            },
            "pairwise_agreement": {
                "pelt_vs_binseg": {
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                }
            },
        }
        report = format_comparison_report(aggregated, {"pelt": {0: float("nan")}}, modality="joint")

        assert "N/A" in report


class TestBoundaryAgreement:
    def test_perfect_agreement(self):
        from segmentation.evaluation import boundary_agreement

        result = boundary_agreement([20, 50, 80], [20, 50, 80], tolerance=5)
        assert result["f1"] == pytest.approx(1.0)

    def test_zero_agreement(self):
        from segmentation.evaluation import boundary_agreement

        result = boundary_agreement([10, 30], [60, 80], tolerance=5)
        assert result["f1"] == 0.0


class TestCompareBoundaries:
    def test_output_structure(self):
        from segmentation.evaluation import compare_boundaries

        out = compare_boundaries({"pelt": [25, 50], "binseg": [24, 49]}, total_frames=100, fps=10.0, tolerance_frames=3)
        assert "duration_stats" in out
        assert "pairwise_agreement" in out
        assert "pelt_vs_binseg" in out["pairwise_agreement"]


class TestRunRuptures:
    @staticmethod
    def _signal(T: int = 160, D: int = 4) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.vstack(
            [
                rng.normal(0, 0.1, size=(T // 2, D)),
                rng.normal(3, 0.1, size=(T // 2, D)),
            ]
        )

    def test_auto_penalty_returns_boundaries(self):
        pytest.importorskip("ruptures", reason="ruptures not installed")
        from segmentation.ruptures_segmenter import RupturesConfig, run_ruptures

        bounds, info = run_ruptures(self._signal(), RupturesConfig(penalty="auto", n_penalty_steps=10))
        assert isinstance(bounds, list)
        assert "penalty_used" in info

    def test_binseg_runs(self):
        pytest.importorskip("ruptures", reason="ruptures not installed")
        from segmentation.ruptures_segmenter import RupturesConfig, run_ruptures

        bounds, _ = run_ruptures(self._signal(), RupturesConfig(algorithm="binseg", penalty=5.0))
        assert isinstance(bounds, list)


class TestTicc:
    def test_too_few_segments_returns_none(self):
        pytest.importorskip("fast_ticc", reason="fast_ticc not installed")
        from segmentation.ticc_primitives import TiccConfig, run_ticc

        result = run_ticc([np.zeros((10, 3))], TiccConfig(n_clusters=2))
        assert result is None


def _write_shared_dataset_for_cli(tmp_dir: Path, fps: float = 15.0) -> Path:
    pandas = pytest.importorskip("pandas", reason="pandas is required for CLI smoke tests")
    pytest.importorskip("pyarrow", reason="pyarrow is required for CLI smoke tests")

    meta_dir = tmp_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "fps": fps,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["joint_0", "joint_1"]},
            "action": {"dtype": "float32", "shape": [2], "names": ["cmd_0", "cmd_1"]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    data_dir = tmp_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "episode_index": [0] * 6 + [1] * 5,
        "frame_index": list(range(6)) + list(range(5)),
        "timestamp": [i / fps for i in range(6)] + [i / fps for i in range(5)],
        "observation.state": [
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.6],
            [1.0, 1.1],
            [1.1, 1.2],
            [1.2, 1.3],
            [1.3, 1.4],
            [1.4, 1.5],
        ],
        "action": [
            [0.05, 0.15],
            [0.15, 0.25],
            [0.25, 0.35],
            [0.35, 0.45],
            [0.45, 0.55],
            [0.55, 0.65],
            [1.05, 1.15],
            [1.15, 1.25],
            [1.25, 1.35],
            [1.35, 1.45],
            [1.45, 1.55],
        ],
    }
    pandas.DataFrame(rows).to_parquet(data_dir / "file-000.parquet", index=False)
    return tmp_dir


class TestCliSmoke:
    def test_joint_and_joint_command_cli_smoke(self, tmp_path: Path):
        pytest.importorskip("pandas", reason="pandas is required for CLI smoke tests")
        pytest.importorskip("pyarrow", reason="pyarrow is required for CLI smoke tests")
        pytest.importorskip("ruptures", reason="ruptures is required for CLI smoke tests")

        repo_root = Path(__file__).resolve().parents[1]
        dataset_dir = _write_shared_dataset_for_cli(tmp_path / "dataset")
        output_dir = tmp_path / "results"

        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_lerobot_segmentation.py"),
                "--dataset",
                str(dataset_dir),
                "--local",
                "--n-episodes",
                "1",
                "--modalities",
                "joint",
                "joint_command",
                "--output",
                str(output_dir),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert (output_dir / "report.md").exists()
        assert (output_dir / "boundaries.csv").exists()
        assert (output_dir / "comparison_table.csv").exists()

        results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        assert results[0]["modalities"]["joint"]["status"] == "success"
        assert results[0]["modalities"]["joint_command"]["status"] == "success"

    def test_cartesian_skip_without_urdf(self, tmp_path: Path):
        pytest.importorskip("pandas", reason="pandas is required for CLI smoke tests")
        pytest.importorskip("pyarrow", reason="pyarrow is required for CLI smoke tests")
        pytest.importorskip("ruptures", reason="ruptures is required for CLI smoke tests")

        repo_root = Path(__file__).resolve().parents[1]
        dataset_dir = _write_shared_dataset_for_cli(tmp_path / "dataset_cartesian")
        output_dir = tmp_path / "results_cartesian"

        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_lerobot_segmentation.py"),
                "--dataset",
                str(dataset_dir),
                "--local",
                "--n-episodes",
                "1",
                "--modalities",
                "cartesian",
                "--output",
                str(output_dir),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        cartesian = results[0]["modalities"]["cartesian"]
        assert cartesian["status"] == "skipped"
        assert "URDF" in cartesian["reason"] or "ee_pose unavailable" in cartesian["reason"]
