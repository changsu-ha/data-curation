#!/usr/bin/env python3
"""End-to-end LeRobot segmentation and primitive discovery pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from segmentation.data_loader import EpisodeRef, uniform_sample_episodes
from segmentation.evaluation import (
    aggregate_episode_comparisons,
    compare_boundaries,
    fmt_float,
    format_comparison_report,
)
from segmentation.features import FeatureBuildConfig, build_features, normalize_modality_name
from segmentation.lerobot_adapter import (
    DatasetSchema,
    inspect_dataset,
    list_episode_refs,
    load_episode_arrays,
)
from segmentation.ruptures_segmenter import RupturesConfig, run_pelt_and_binseg
from segmentation.sktime_benchmark import get_available_segmenters, run_sktime_benchmark
from segmentation.ticc_primitives import TiccConfig, TiccResult, run_ticc


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        warnings.warn("pyyaml not installed; using default settings.", stacklevel=2)
        return {}


def _build_ruptures_config(cfg: dict[str, Any]) -> RupturesConfig:
    section = cfg.get("ruptures", {})
    return RupturesConfig(
        model=section.get("model", "rbf"),
        algorithm="pelt",
        min_size=int(section.get("min_size", 5)),
        jump=int(section.get("jump", 1)),
        penalty=section.get("penalty", "auto"),
        auto_method=section.get("auto_method", "elbow"),
        penalty_range=tuple(section.get("penalty_range", [0.5, 200.0])),
        n_penalty_steps=int(section.get("n_penalty_steps", 40)),
        max_n_bkps=int(section.get("max_n_bkps", 30)),
    )


def _build_ticc_config(cfg: dict[str, Any]) -> TiccConfig:
    section = cfg.get("ticc", {})
    return TiccConfig(
        window_size=int(section.get("window_size", 10)),
        n_clusters=section.get("n_clusters", "auto"),
        max_k=int(section.get("max_k", 10)),
        lambda_=float(section.get("lambda_", 0.11)),
        beta=float(section.get("beta", 400.0)),
        max_iter=int(section.get("max_iter", 100)),
        random_state=int(section.get("random_state", 42)),
    )


def _build_feature_config(cfg: dict[str, Any], dt_override: float | None) -> FeatureBuildConfig:
    section = cfg.get("features", {})
    dt_cfg = section.get("dt")
    dt = dt_override if dt_override is not None else (float(dt_cfg) if dt_cfg is not None else None)
    return FeatureBuildConfig(
        dt=dt,
        smoothing=section.get("smoothing", "savgol"),
        normalize=section.get("normalize", "mad"),
        modality_weights=section.get("modality_weights"),
    )


def _resolve_modalities(requested: list[str]) -> tuple[list[str], list[str]]:
    canonical: list[str] = []
    alias_notes: list[str] = []
    for modality in requested:
        normalized = normalize_modality_name(modality)
        if modality != normalized:
            alias_notes.append(f"`{modality}` was treated as `{normalized}`.")
        if normalized not in canonical:
            canonical.append(normalized)
    return canonical, alias_notes


def _build_sample_dict(arrays: dict[str, np.ndarray], modality: str) -> dict[str, Any]:
    canonical = normalize_modality_name(modality)
    sample: dict[str, Any] = {"timestamps": arrays["timestamps"]}

    if canonical in {"joint", "joint_command"}:
        if "joint_states" not in arrays:
            raise KeyError("joint_states not available for joint-based modality")
        sample["q"] = arrays["joint_states"]

    if canonical == "joint_command":
        if "joint_commands" not in arrays:
            raise KeyError("joint_commands not available for joint_command modality")
        sample["q_cmd"] = arrays["joint_commands"]

    if canonical == "cartesian":
        if "ee_pose" not in arrays:
            raise KeyError("ee_pose not available for cartesian modality")
        sample["cartesian"] = arrays["ee_pose"]

    return sample


def _plot_segmentation(
    timestamps: np.ndarray,
    signal_2d: np.ndarray,
    boundary_results: dict[str, list[int]],
    episode_id: Any,
    modality: str,
    save_path: Path,
) -> None:
    n_dims = min(3, signal_2d.shape[1])
    n_methods = len(boundary_results)
    fig, axes = plt.subplots(n_dims + 1, 1, figsize=(14, 3 * (n_dims + 1)), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_methods, 1)))

    for idx in range(n_dims):
        ax = axes[idx]
        ax.plot(timestamps, signal_2d[:, idx], color="black", linewidth=0.8, alpha=0.85)
        for color_idx, (method, bounds) in enumerate(boundary_results.items()):
            for boundary in bounds:
                if 0 < boundary < len(timestamps):
                    ax.axvline(
                        timestamps[boundary],
                        color=colors[color_idx],
                        alpha=0.6,
                        linewidth=1.2,
                        label=method if idx == 0 else None,
                    )
        ax.set_ylabel(f"dim {idx}", fontsize=9)
        ax.grid(True, alpha=0.25)
        if idx == 0 and n_methods > 0:
            ax.legend(loc="upper right", fontsize=7, ncol=max(1, min(n_methods, 4)))

    ax_bar = axes[-1]
    ax_bar.bar(list(boundary_results.keys()), [len(bounds) + 1 for bounds in boundary_results.values()], color=colors[:n_methods])
    ax_bar.set_ylabel("# segments", fontsize=9)
    ax_bar.set_xlabel("Method", fontsize=9)

    fig.suptitle(f"Episode {episode_id} - {modality} segmentation", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {save_path}")


def _plot_primitives(
    segments_feats: list[np.ndarray],
    ticc_result: TiccResult,
    episode_id: Any,
    modality: str,
    save_path: Path,
) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, max(ticc_result.n_clusters, 1)))
    fig, ax = plt.subplots(figsize=(14, 2))

    x_offset = 0
    for idx, (segment, label) in enumerate(zip(segments_feats, ticc_result.cluster_assignments)):
        width = len(segment)
        ax.barh(
            0,
            width,
            left=x_offset,
            height=0.8,
            color=colors[label % ticc_result.n_clusters],
            label=f"cluster {label}" if idx == 0 or ticc_result.cluster_assignments[idx - 1] != label else None,
        )
        x_offset += width

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("frames", fontsize=9)
    fig.suptitle(
        f"Episode {episode_id} - {modality} TICC primitives "
        f"(k={ticc_result.n_clusters}, sil={fmt_float(ticc_result.silhouette_score, precision=3)})",
        fontsize=11,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {save_path}")


def _compute_fk_if_needed(
    arrays: dict[str, np.ndarray],
    fk_assets: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], str | None]:
    if "ee_pose" in arrays:
        return arrays, None
    if fk_assets is None:
        return arrays, "ee_pose unavailable and no URDF provided"

    try:
        from segmentation.kinematics import compute_fk_trajectory, fk_to_quaternion

        positions, rot_matrices, _ = compute_fk_trajectory(
            fk_assets["model"],
            fk_assets["data"],
            fk_assets["name_to_q_idx"],
            arrays["joint_states"],
            fk_assets["ee_frame"],
        )
        quaternions = fk_to_quaternion(rot_matrices)
        updated = dict(arrays)
        updated["ee_pose"] = np.hstack([positions, quaternions])
        return updated, None
    except Exception as exc:
        return arrays, f"FK failed: {exc}"


def _process_episode(
    ep_ref: EpisodeRef,
    schema: DatasetSchema,
    modalities: list[str],
    feat_config: FeatureBuildConfig,
    rupt_config: RupturesConfig,
    ticc_config: TiccConfig,
    sktime_available: dict[str, type],
    output_dir: Path,
    fk_assets: dict[str, Any] | None,
    tolerance_frames: int,
    write_output: dict[str, bool],
    fill_policy: str,
) -> dict[str, Any]:
    ep_id = ep_ref.episode_id
    print(f"\n--- Episode {ep_id} ---")

    try:
        arrays = load_episode_arrays(ep_ref, schema, fill_policy=fill_policy)
    except Exception as exc:
        print(f"  [error] load_episode_arrays: {exc}")
        return {"episode_id": ep_id, "status": "error", "error": str(exc), "modalities": {}}

    cartesian_reason: str | None = None
    if "cartesian" in modalities:
        arrays, cartesian_reason = _compute_fk_if_needed(arrays, fk_assets)
        if cartesian_reason:
            print(f"  [skip] cartesian: {cartesian_reason}")

    ep_results: dict[str, Any] = {
        "episode_id": ep_id,
        "status": "success",
        "modalities": {},
    }
    ep_dir = output_dir / f"episode_{ep_id}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    for modality in modalities:
        print(f"  [modality={modality}]")
        mod_results: dict[str, Any] = {"status": "success"}

        if modality == "cartesian" and "ee_pose" not in arrays:
            mod_results["status"] = "skipped"
            mod_results["reason"] = cartesian_reason or "ee_pose unavailable"
            ep_results["modalities"][modality] = mod_results
            continue

        if modality == "joint_command" and "joint_commands" not in arrays:
            mod_results["status"] = "skipped"
            mod_results["reason"] = "joint_commands not available in dataset"
            ep_results["modalities"][modality] = mod_results
            continue

        try:
            sample_dict = _build_sample_dict(arrays, modality)
            feat_out = build_features(sample_dict, modality, feat_config)
        except KeyError as exc:
            mod_results["status"] = "skipped"
            mod_results["reason"] = str(exc)
            ep_results["modalities"][modality] = mod_results
            print(f"    [skip] {exc}")
            continue
        except Exception as exc:
            mod_results["status"] = "error"
            mod_results["reason"] = str(exc)
            ep_results["modalities"][modality] = mod_results
            print(f"    [error] build_features: {exc}")
            continue

        feat_mat = np.asarray(feat_out["matrix"], dtype=float)
        timestamps_rs = np.asarray(feat_out["timestamps"], dtype=float)
        fps = 1.0 / float(np.median(np.diff(timestamps_rs))) if len(timestamps_rs) > 1 else schema.fps
        mod_results["feature_names"] = feat_out["feature_names"]
        mod_results["source_channels"] = feat_out["source_channels"]

        t0 = time.perf_counter()
        rupt_out = run_pelt_and_binseg(feat_mat, rupt_config)
        rupt_runtime = time.perf_counter() - t0

        pelt_bounds, pelt_info = rupt_out["pelt"]
        binseg_bounds, binseg_info = rupt_out["binseg"]
        print(
            "    ruptures PELT: "
            f"{len(pelt_bounds)} breakpoints, penalty={fmt_float(pelt_info.get('penalty_used'), precision=2)}"
        )
        print(f"    ruptures Binseg: {len(binseg_bounds)} breakpoints")

        all_bounds: dict[str, list[int]] = {
            "pelt": pelt_bounds,
            "binseg": binseg_bounds,
        }
        all_runtimes: dict[str, float] = {
            "pelt": rupt_runtime * 0.5,
            "binseg": rupt_runtime * 0.5,
        }

        if sktime_available:
            sk_results = run_sktime_benchmark(
                feat_mat,
                n_segments=len(pelt_bounds) + 1,
                available=sktime_available,
            )
            mod_results["sktime"] = {"status": "success", "algorithms": list(sk_results.keys())}
            for sk_name, (sk_bounds, sk_rt, sk_info) in sk_results.items():
                method_name = f"sktime_{sk_name}"
                all_bounds[method_name] = sk_bounds
                all_runtimes[method_name] = sk_rt
                print(f"    sktime {sk_name}: {len(sk_bounds)} breakpoints")
                mod_results.setdefault("sktime_details", {})[method_name] = sk_info
        else:
            mod_results["sktime"] = {
                "status": "skipped",
                "reason": "No sktime segmenters available",
            }

        comparison = compare_boundaries(
            all_bounds,
            total_frames=len(timestamps_rs),
            fps=fps,
            tolerance_frames=tolerance_frames,
        )
        mod_results["boundaries"] = all_bounds
        mod_results["comparison"] = comparison
        mod_results["runtimes"] = all_runtimes
        mod_results["ruptures_info"] = {"pelt": pelt_info, "binseg": binseg_info}

        if len(pelt_bounds) + 1 >= 2:
            edges = [0] + sorted(pelt_bounds) + [len(timestamps_rs)]
            segments_feats = [feat_mat[edges[i] : edges[i + 1]] for i in range(len(edges) - 1)]
            t0_ticc = time.perf_counter()
            ticc_result = run_ticc(segments_feats, ticc_config)
            ticc_runtime = time.perf_counter() - t0_ticc

            if ticc_result is None:
                mod_results["ticc"] = {
                    "status": "skipped",
                    "reason": "fast_ticc unavailable or insufficient usable segments",
                    "runtime_s": ticc_runtime,
                }
            else:
                print(
                    f"    TICC: k={ticc_result.n_clusters}, "
                    f"sil={fmt_float(ticc_result.silhouette_score, precision=3)}"
                )
                mod_results["ticc"] = {
                    "status": "success",
                    "cluster_assignments": ticc_result.cluster_assignments,
                    "n_clusters": ticc_result.n_clusters,
                    "cluster_sizes": ticc_result.cluster_sizes,
                    "silhouette_score": ticc_result.silhouette_score,
                    "transition_matrix": ticc_result.transition_matrix.tolist(),
                    "representative_indices": ticc_result.representative_indices,
                    "diagnostics": ticc_result.diagnostics,
                    "runtime_s": ticc_runtime,
                }
                if write_output.get("plots", True):
                    _plot_primitives(
                        segments_feats,
                        ticc_result,
                        ep_id,
                        modality,
                        ep_dir / f"primitives_ticc_{modality}.png",
                    )
        else:
            mod_results["ticc"] = {
                "status": "skipped",
                "reason": "Need at least 2 PELT segments for TICC",
            }

        if write_output.get("plots", True):
            _plot_segmentation(
                timestamps_rs,
                feat_mat,
                all_bounds,
                ep_id,
                modality,
                ep_dir / f"segmentation_{modality}.png",
            )

        ep_results["modalities"][modality] = mod_results

    return ep_results


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_boundaries_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "modality", "status", "reason", "method", "boundary_frames"])
        for result in results:
            ep_id = result.get("episode_id")
            for modality, mod_data in result.get("modalities", {}).items():
                status = mod_data.get("status", "unknown")
                reason = mod_data.get("reason", "")
                boundaries = mod_data.get("boundaries", {})
                if not boundaries:
                    writer.writerow([ep_id, modality, status, reason, "", ""])
                    continue
                for method, boundary_list in boundaries.items():
                    writer.writerow([ep_id, modality, status, reason, method, ";".join(map(str, boundary_list))])


def _collect_skip_summary(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for result in results:
        ep_id = result.get("episode_id")
        for modality, mod_data in result.get("modalities", {}).items():
            if mod_data.get("status") == "skipped":
                lines.append(
                    f"- Episode {ep_id}, modality `{modality}`: {mod_data.get('reason', 'no reason provided')}"
                )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot segmentation and primitive discovery pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF repo ID")
    parser.add_argument("--urdf", default=None, help="Path to robot URDF (optional; enables Cartesian FK)")
    parser.add_argument("--n-episodes", type=int, default=5, help="Episodes to process (0=schema only)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        choices=["joint", "cartesian", "joint_command", "ablation"],
        help="Modalities to run (overrides config). `ablation` is accepted as an alias of `joint_command`.",
    )
    parser.add_argument("--ee-frame", default="ee_right", help="End-effector frame name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local", action="store_true", help="Treat --dataset as a local path")
    args = parser.parse_args()

    if args.config:
        cfg = _load_yaml_config(args.config)
    else:
        from segmentation.configs import get_config_path

        cfg = _load_yaml_config(get_config_path("lerobot_segmentation.yaml"))

    if args.modalities:
        cfg.setdefault("features", {})["modalities"] = args.modalities
    if args.n_episodes != 5:
        cfg.setdefault("dataset", {})["n_episodes"] = args.n_episodes

    requested_modalities = list(cfg.get("features", {}).get("modalities", ["joint"]))
    modalities, alias_notes = _resolve_modalities(requested_modalities)
    n_episodes = int(cfg.get("dataset", {}).get("n_episodes", args.n_episodes))
    seed = int(cfg.get("dataset", {}).get("seed", args.seed))
    tolerance_frames = int(cfg.get("evaluation", {}).get("boundary_tolerance_frames", 5))
    fill_policy = str(cfg.get("dataset", {}).get("missing_values", "ffill_drop"))
    write_output = {
        "plots": bool(cfg.get("output", {}).get("plots", True)),
        "csv": bool(cfg.get("output", {}).get("csv", True)),
        "json": bool(cfg.get("output", {}).get("json", True)),
        "report": bool(cfg.get("output", {}).get("report", True)),
    }

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Dataset inspection")
    print("=" * 60)

    if args.local or Path(args.dataset).is_dir():
        dataset_path = Path(args.dataset).expanduser().resolve()
    else:
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            dataset_path = Path(snapshot_download(repo_id=args.dataset, repo_type="dataset"))
        except Exception as exc:
            sys.exit(f"Failed to download dataset: {exc}")

    schema = inspect_dataset(dataset_path)
    schema_path = output_dir / "schema_report.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, indent=2)
    print(f"Schema saved to {schema_path}")

    if n_episodes <= 0:
        print("n_episodes=0: schema-only run complete.")
        return

    fk_assets: dict[str, Any] | None = None
    if args.urdf:
        print("\nLoading URDF for FK...")
        try:
            from segmentation.kinematics import build_joint_index_map, load_robot_model

            model, data = load_robot_model(args.urdf)
            fk_assets = {
                "model": model,
                "data": data,
                "name_to_q_idx": build_joint_index_map(model),
                "ee_frame": args.ee_frame,
            }
            print(f"  URDF loaded. EE frame: {args.ee_frame}")
        except Exception as exc:
            warnings.warn(f"Could not load URDF: {exc}. Cartesian modality will be skipped.")

    print("\nPhase 2: Sampling episodes")
    all_episode_refs = list_episode_refs(dataset_path, schema)
    if not all_episode_refs:
        sys.exit("No episodes found in dataset.")

    sampled = uniform_sample_episodes(all_episode_refs, num_samples=n_episodes, seed=seed)
    print(f"  Selected {len(sampled)} / {len(all_episode_refs)} episodes: {[ep.episode_id for ep in sampled]}")

    feat_config = _build_feature_config(cfg, None)
    rupt_config = _build_ruptures_config(cfg)
    ticc_config = _build_ticc_config(cfg)

    print("\nProbing sktime segmenters...")
    sktime_available = get_available_segmenters()
    if sktime_available:
        print(f"  Found: {list(sktime_available.keys())}")
    else:
        print("  No sktime segmenters found.")

    print("\nPhase 3: Processing episodes")
    all_results: list[dict[str, Any]] = []
    comparisons_by_modality: dict[str, list[dict[str, Any]]] = {modality: [] for modality in modalities}

    for ep_ref in sampled:
        result = _process_episode(
            ep_ref=ep_ref,
            schema=schema,
            modalities=modalities,
            feat_config=feat_config,
            rupt_config=rupt_config,
            ticc_config=ticc_config,
            sktime_available=sktime_available,
            output_dir=output_dir,
            fk_assets=fk_assets,
            tolerance_frames=tolerance_frames,
            write_output=write_output,
            fill_policy=fill_policy,
        )
        all_results.append(result)
        for modality in modalities:
            cmp = result.get("modalities", {}).get(modality, {}).get("comparison")
            if cmp:
                comparisons_by_modality[modality].append(cmp)

    if write_output.get("csv", True):
        boundaries_path = output_dir / "boundaries.csv"
        _write_boundaries_csv(all_results, boundaries_path)
        print(f"\nBoundaries saved to {boundaries_path}")

    if write_output.get("json", True):
        results_path = output_dir / "results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=_json_default)
        print(f"Full results saved to {results_path}")

    report_sections: list[str] = [
        "# LeRobot Segmentation Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Layout: `{schema.layout}`",
        f"- Episodes processed: {len(sampled)}",
        f"- Modalities: {', '.join(modalities)}",
        "",
    ]
    if alias_notes:
        report_sections.extend(["## Modality Notes", "", *[f"- {note}" for note in alias_notes], ""])

    comparison_rows: list[dict[str, Any]] = []
    for modality in modalities:
        episode_comparisons = comparisons_by_modality.get(modality, [])
        if not episode_comparisons:
            continue

        aggregated = aggregate_episode_comparisons(episode_comparisons)
        runtimes_by_method: dict[str, dict[Any, float]] = {}
        for result in all_results:
            mod_data = result.get("modalities", {}).get(modality, {})
            for method, runtime_s in mod_data.get("runtimes", {}).items():
                runtimes_by_method.setdefault(method, {})[result["episode_id"]] = runtime_s

        report_sections.append(format_comparison_report(aggregated, runtimes_by_method, modality))
        for method, stats in aggregated.get("duration_stats", {}).items():
            comparison_rows.append({"modality": modality, "method": method, **stats})

    if write_output.get("csv", True) and comparison_rows:
        comparison_path = output_dir / "comparison_table.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"Comparison table saved to {comparison_path}")

    skip_lines = _collect_skip_summary(all_results)
    if skip_lines:
        report_sections.extend(["## Skipped Modalities", "", *skip_lines, ""])

    report_sections.extend(
        [
            "## Assumptions and Limitations",
            "",
            "- No ground-truth labels were used; comparison is unsupervised via inter-method agreement.",
            "- PELT penalty is chosen automatically unless a fixed value is configured.",
            "- TICC and sktime results are skipped when the corresponding libraries are unavailable.",
            "- Cartesian modality is skipped when neither ee_pose nor a valid FK path is available.",
            "",
            "## Next Steps",
            "",
            "- Add task labels or phase annotations for supervised validation.",
            "- Tune penalty search ranges for the target dataset FPS and noise level.",
            "- Add robot-specific joint-name mapping for datasets that need FK but do not match the default profile.",
        ]
    )

    if write_output.get("report", True):
        report_path = output_dir / "report.md"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(report_sections))
        print(f"Report saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
