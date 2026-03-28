#!/usr/bin/env python3
"""
Convert LeRobot 2.0 dataset joint states/commands to Cartesian 6-DOF poses
using Pinocchio forward kinematics and RB-Y1 URDF, then plot with matplotlib.

Usage
-----
    python scripts/joint_to_cartesian.py \\
        --dataset tony346/rby1_HF_Test \\
        --urdf /path/to/model.urdf

    # Local dataset
    python scripts/joint_to_cartesian.py \\
        --dataset /local/path/to/dataset \\
        --urdf /path/to/model.urdf \\
        --local

FK computation is provided by the ``segmentation.kinematics`` module.
Robot-specific constants (joint names, gripper mapping, EE frames) are in
``segmentation.robots.rby1``.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from segmentation.kinematics import (
    build_joint_index_map,
    compute_fk_trajectory,
    load_robot_model,
)
from segmentation.robots.rby1 import DATASET_JOINT_NAMES, EE_FRAMES, GRIPPER_MAPPING


# ---------------------------------------------------------------------------
# Dataset loading (LeRobot 2.0 format)
# ---------------------------------------------------------------------------

def resolve_dataset_path(dataset: str, local: bool = False) -> Path:
    """Resolve dataset to a local path.

    If *local* is ``True``, treat *dataset* as a local directory.
    Otherwise download from HuggingFace Hub.
    """
    if local or Path(dataset).is_dir():
        return Path(dataset)
    path = snapshot_download(repo_id=dataset, repo_type="dataset")
    return Path(path)


def load_info(dataset_dir: Path) -> dict:
    """Load ``meta/info.json``."""
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def load_episodes(dataset_dir: Path, episode_ids: list[int] | None = None) -> dict[int, "pd.DataFrame"]:
    """Load episode data from parquet chunk files.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping episode_id → DataFrame.
    """
    data_dir = dataset_dir / "data"
    episodes: dict[int, pd.DataFrame] = {}

    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    for pf in parquet_files:
        ep_id = int(pf.stem.split("_")[-1])
        if episode_ids is not None and ep_id not in episode_ids:
            continue
        episodes[ep_id] = pd.read_parquet(pf)

    return episodes


def extract_joint_array(df: "pd.DataFrame", column: str) -> np.ndarray:
    """Extract joint values from a DataFrame column.

    Each row's value is a list/array of 44 floats.
    Returns ``np.ndarray`` of shape ``(N, 44)``.
    """
    return np.array(df[column].tolist(), dtype=np.float64)


def extract_timestamps(df: "pd.DataFrame", fps: float) -> np.ndarray:
    """Build a timestamp array from the DataFrame or from fps."""
    if "timestamp" in df.columns:
        return df["timestamp"].to_numpy(dtype=np.float64)
    return np.arange(len(df), dtype=np.float64) / fps


# ---------------------------------------------------------------------------
# Saving FK results
# ---------------------------------------------------------------------------

def save_fk_results(
    save_dir: Path,
    episode_id: int,
    ee_name: str,
    timestamps: np.ndarray,
    state_pos: np.ndarray,
    state_rot: np.ndarray,
    state_rpy: np.ndarray,
    action_pos: np.ndarray,
    action_rot: np.ndarray,
    action_rpy: np.ndarray,
) -> None:
    """Save FK results to a ``.npz`` file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"episode_{episode_id:06d}_{ee_name}_fk.npz"
    np.savez(
        fname,
        timestamps=timestamps,
        state_pos=state_pos,
        state_rotation_matrix=state_rot,
        state_rpy=state_rpy,
        action_pos=action_pos,
        action_rotation_matrix=action_rot,
        action_rpy=action_rpy,
    )
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cartesian_poses(
    timestamps: np.ndarray,
    state_poses_pos: np.ndarray,
    state_poses_rpy: np.ndarray,
    action_poses_pos: np.ndarray,
    action_poses_rpy: np.ndarray,
    ee_name: str,
    episode_id: int,
    save_path: Path | None = None,
) -> None:
    """Plot 6-DOF Cartesian poses (x, y, z, roll, pitch, yaw) over time."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    labels = ["x [m]", "y [m]", "z [m]", "roll [rad]", "pitch [rad]", "yaw [rad]"]
    state_data = np.hstack([state_poses_pos, state_poses_rpy])    # (N, 6)
    action_data = np.hstack([action_poses_pos, action_poses_rpy]) # (N, 6)

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(timestamps, state_data[:, i], label="state", color="tab:blue", linewidth=1.0)
        ax.plot(timestamps, action_data[:, i], label="action", color="tab:red", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]", fontsize=11)
    fig.suptitle(f"{ee_name} Cartesian Pose — Episode {episode_id}", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot joint states/commands to Cartesian poses via FK"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="HuggingFace repo ID (e.g. tony346/rby1_HF_Test) or local path",
    )
    parser.add_argument(
        "--urdf", type=str, required=True,
        help="Path to the RB-Y1 URDF file",
    )
    parser.add_argument(
        "--episodes", type=int, nargs="*", default=None,
        help="Episode IDs to process (default: all)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./output",
        help="Directory for saving plots and FK results (default: ./output)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Treat --dataset as a local directory path",
    )
    args = parser.parse_args()

    # --- Load URDF ---
    print(f"Loading URDF: {args.urdf}")
    model, data = load_robot_model(args.urdf)
    name_to_q_idx = build_joint_index_map(model)

    print(f"  Model joints ({model.njoints - 1}):")
    for jname in DATASET_JOINT_NAMES:
        status = "OK" if jname in name_to_q_idx else "MISSING"
        print(f"    {jname}: {status}")

    for frame in EE_FRAMES:
        fid = model.getFrameId(frame)
        print(f"  Frame '{frame}': id={fid}")

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.dataset}")
    dataset_dir = resolve_dataset_path(args.dataset, local=args.local)
    info = load_info(dataset_dir)
    fps = info.get("fps", 30)
    print(f"  FPS: {fps}")
    print(f"  Codebase version: {info.get('codebase_version', 'unknown')}")

    episodes = load_episodes(dataset_dir, args.episodes)
    print(f"  Loaded {len(episodes)} episode(s): {sorted(episodes.keys())}")

    save_dir = Path(args.save_dir)

    # --- Process each episode ---
    for ep_id in sorted(episodes.keys()):
        print(f"\nProcessing episode {ep_id}...")
        df = episodes[ep_id]
        timestamps = extract_timestamps(df, fps)

        state_array = extract_joint_array(df, "observation.state")
        action_array = extract_joint_array(df, "action")
        print(f"  Frames: {len(timestamps)}, state shape: {state_array.shape}")

        for ee_name in EE_FRAMES:
            print(f"  Computing FK for {ee_name}...")

            s_pos, s_rot, s_rpy = compute_fk_trajectory(
                model, data, name_to_q_idx, state_array, ee_name
            )
            a_pos, a_rot, a_rpy = compute_fk_trajectory(
                model, data, name_to_q_idx, action_array, ee_name
            )

            save_fk_results(
                save_dir / "fk_results",
                ep_id, ee_name, timestamps,
                s_pos, s_rot, s_rpy,
                a_pos, a_rot, a_rpy,
            )

            plot_path = save_dir / "plots" / f"episode_{ep_id:06d}_{ee_name}.png"
            plot_cartesian_poses(
                timestamps, s_pos, s_rpy, a_pos, a_rpy,
                ee_name, ep_id, save_path=plot_path,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
