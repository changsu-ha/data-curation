"""Utilities for per-sample segmentation reporting.

This module builds a per-sample report containing:
- Segment timeline table
- Duration statistics by label
- Low-confidence segment list
- Timeline visualization plots

Artifacts are written to ``output_dir/sample_{id}/``:
- ``segments.json``
- ``summary.csv``
- ``timeline.png``
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import matplotlib

# Use a non-interactive backend for batch environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Segment:
    """Simple segment record.

    Attributes:
        start: Segment start timestamp (seconds).
        end: Segment end timestamp (seconds).
        label: Segment label/class name.
        confidence: Optional confidence score in [0, 1].
    """

    start: float
    end: float
    label: str
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _normalize_segments(segments: Iterable[dict[str, Any] | Segment]) -> list[Segment]:
    normalized: list[Segment] = []
    for seg in segments:
        if isinstance(seg, Segment):
            normalized.append(seg)
            continue
        normalized.append(
            Segment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                label=str(seg["label"]),
                confidence=None if seg.get("confidence") is None else float(seg["confidence"]),
            )
        )
    return sorted(normalized, key=lambda s: (s.start, s.end))


def build_segment_timeline(segments: Sequence[Segment]) -> list[dict[str, Any]]:
    """Build a timeline table for each segment."""
    return [
        {
            "segment_id": i,
            "start": seg.start,
            "end": seg.end,
            "duration": seg.duration,
            "label": seg.label,
            "confidence": seg.confidence,
        }
        for i, seg in enumerate(segments)
    ]


def compute_label_duration_stats(segments: Sequence[Segment]) -> list[dict[str, Any]]:
    """Compute duration summary grouped by label."""
    groups: dict[str, list[float]] = defaultdict(list)
    for seg in segments:
        groups[seg.label].append(seg.duration)

    stats: list[dict[str, Any]] = []
    for label in sorted(groups):
        durations = groups[label]
        stats.append(
            {
                "label": label,
                "count": len(durations),
                "total_duration": float(sum(durations)),
                "mean_duration": float(mean(durations)),
                "min_duration": float(min(durations)),
                "max_duration": float(max(durations)),
            }
        )
    return stats


def find_low_confidence_segments(
    segments: Sequence[Segment], threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Return segment records where confidence is below ``threshold``."""
    low_conf = []
    for i, seg in enumerate(segments):
        if seg.confidence is None:
            continue
        if seg.confidence < threshold:
            low_conf.append(
                {
                    "segment_id": i,
                    "label": seg.label,
                    "start": seg.start,
                    "end": seg.end,
                    "duration": seg.duration,
                    "confidence": seg.confidence,
                }
            )
    return low_conf


def _segment_index_per_timestep(
    timestamps: np.ndarray, segments: Sequence[Segment]
) -> np.ndarray:
    idx = np.full(shape=(len(timestamps),), fill_value=-1, dtype=int)
    for seg_i, seg in enumerate(segments):
        mask = (timestamps >= seg.start) & (timestamps <= seg.end)
        idx[mask] = seg_i
    return idx


def _segment_colors(segments: Sequence[Segment]) -> tuple[dict[int, Any], dict[str, Any]]:
    labels = sorted({seg.label for seg in segments})
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    label_to_color = {label: cmap(i) for i, label in enumerate(labels)}
    seg_to_color = {i: label_to_color[seg.label] for i, seg in enumerate(segments)}
    return seg_to_color, label_to_color


def plot_sample_timeline(
    *,
    timestamps: Sequence[float],
    x: Sequence[float],
    z: Sequence[float],
    gripper_state: Sequence[float],
    joint_speed_norm: Sequence[float],
    segments: Sequence[Segment],
    output_path: Path,
) -> None:
    """Create timeline visualization with three panels.

    Panels:
      1) x/z trajectory with segment color overlay
      2) gripper state with label overlay
      3) joint speed norm with boundary markers
    """
    ts = np.asarray(timestamps, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    grip = np.asarray(gripper_state, dtype=float)
    speed = np.asarray(joint_speed_norm, dtype=float)

    if not (len(ts) == len(x_arr) == len(z_arr) == len(grip) == len(speed)):
        raise ValueError("timestamps, trajectory, gripper_state, joint_speed_norm 길이가 일치해야 합니다.")

    seg_idx = _segment_index_per_timestep(ts, segments)
    seg_to_color, label_to_color = _segment_colors(segments)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    # 1) x/z trajectory + segment color overlay
    ax0 = axes[0]
    valid = seg_idx >= 0
    if np.any(valid):
        colors = [seg_to_color[i] for i in seg_idx[valid]]
        ax0.scatter(x_arr[valid], z_arr[valid], c=colors, s=10, alpha=0.9)
    if np.any(~valid):
        ax0.scatter(x_arr[~valid], z_arr[~valid], c="lightgray", s=10, alpha=0.5)
    ax0.set_title("Trajectory (x/z) with segment overlay")
    ax0.set_xlabel("x")
    ax0.set_ylabel("z")
    ax0.grid(alpha=0.2)

    # 2) gripper state + label overlay
    ax1 = axes[1]
    ax1.plot(ts, grip, color="black", linewidth=1.2, label="gripper_state")
    for seg in segments:
        ax1.axvspan(seg.start, seg.end, color=label_to_color[seg.label], alpha=0.20)
    ax1.set_title("Gripper state with label overlay")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("gripper")
    ax1.grid(alpha=0.2)

    # 3) joint speed norm + boundary marker
    ax2 = axes[2]
    ax2.plot(ts, speed, color="#1f77b4", linewidth=1.2, label="joint_speed_norm")
    for seg in segments:
        ax2.axvline(seg.start, color="red", linestyle="--", alpha=0.5, linewidth=0.9)
        ax2.axvline(seg.end, color="red", linestyle="--", alpha=0.5, linewidth=0.9)
    ax2.set_title("Joint speed norm with boundary markers")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("speed norm")
    ax2.grid(alpha=0.2)

    # Legend by label (deduplicated)
    handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=label)
        for label, color in label_to_color.items()
    ]
    if handles:
        ax1.legend(handles=handles, title="labels", ncol=min(4, len(handles)), loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_boundary_f1(
    pred_boundaries: Sequence[float],
    gt_boundaries: Sequence[float],
    tolerance: float = 0.1,
) -> dict[str, float]:
    """Boundary F1 계산 스켈레톤.

    Future GT-ready metric helper. Current implementation provides a simple
    tolerance-based matching baseline that can be extended later.
    """
    pred = sorted(float(v) for v in pred_boundaries)
    gt = sorted(float(v) for v in gt_boundaries)

    if not pred and not gt:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not gt:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gt: set[int] = set()
    tp = 0
    for p in pred:
        best_i = None
        best_dist = None
        for i, g in enumerate(gt):
            if i in matched_gt:
                continue
            dist = abs(p - g)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_i = i
                best_dist = dist
        if best_i is not None:
            matched_gt.add(best_i)
            tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_segment_iou(
    pred_segments: Sequence[Segment], gt_segments: Sequence[Segment]
) -> dict[str, Any]:
    """Segment IoU 계산 스켈레톤.

    Returns macro IoU over all pairwise best matches with same label.
    Can be replaced by stricter protocol once GT format is finalized.
    """

    def _interval_iou(a: Segment, b: Segment) -> float:
        inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
        union = max(a.end, b.end) - min(a.start, b.start)
        return inter / union if union > 0 else 0.0

    ious: list[float] = []
    used_gt: set[int] = set()
    for pred in pred_segments:
        best_iou = 0.0
        best_idx = None
        for i, gt in enumerate(gt_segments):
            if i in used_gt or pred.label != gt.label:
                continue
            score = _interval_iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_idx is not None:
            used_gt.add(best_idx)
            ious.append(best_iou)

    return {
        "matched_segments": len(ious),
        "macro_iou": float(sum(ious) / len(ious)) if ious else 0.0,
    }


def _write_summary_csv(
    output_path: Path,
    timeline: Sequence[dict[str, Any]],
    duration_stats: Sequence[dict[str, Any]],
    low_confidence: Sequence[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["section", "segment_id", "label", "start", "end", "duration", "confidence", "count", "total_duration", "mean_duration", "min_duration", "max_duration"])
        for row in timeline:
            writer.writerow([
                "timeline",
                row["segment_id"],
                row["label"],
                row["start"],
                row["end"],
                row["duration"],
                row["confidence"],
                "",
                "",
                "",
                "",
                "",
            ])

        for row in duration_stats:
            writer.writerow([
                "label_duration_stats",
                "",
                row["label"],
                "",
                "",
                "",
                "",
                row["count"],
                row["total_duration"],
                row["mean_duration"],
                row["min_duration"],
                row["max_duration"],
            ])

        for row in low_confidence:
            writer.writerow([
                "low_confidence",
                row["segment_id"],
                row["label"],
                row["start"],
                row["end"],
                row["duration"],
                row["confidence"],
                "",
                "",
                "",
                "",
                "",
            ])


def generate_sample_report(
    *,
    sample_id: str | int,
    output_dir: str | Path,
    segments: Iterable[dict[str, Any] | Segment],
    timestamps: Sequence[float],
    trajectory_x: Sequence[float],
    trajectory_z: Sequence[float],
    gripper_state: Sequence[float],
    joint_speed_norm: Sequence[float],
    low_conf_threshold: float = 0.5,
    gt_boundaries: Sequence[float] | None = None,
    gt_segments: Iterable[dict[str, Any] | Segment] | None = None,
) -> dict[str, Any]:
    """Generate per-sample report and save report artifacts.

    Saved files under ``output_dir/sample_{id}/``:
      - ``segments.json``
      - ``summary.csv``
      - ``timeline.png``
    """
    sample_root = Path(output_dir) / f"sample_{sample_id}"
    sample_root.mkdir(parents=True, exist_ok=True)

    segs = _normalize_segments(segments)
    timeline = build_segment_timeline(segs)
    duration_stats = compute_label_duration_stats(segs)
    low_confidence = find_low_confidence_segments(segs, threshold=low_conf_threshold)

    plot_sample_timeline(
        timestamps=timestamps,
        x=trajectory_x,
        z=trajectory_z,
        gripper_state=gripper_state,
        joint_speed_norm=joint_speed_norm,
        segments=segs,
        output_path=sample_root / "timeline.png",
    )

    boundary_metrics = None
    iou_metrics = None
    if gt_boundaries is not None:
        pred_boundaries = [seg.start for seg in segs] + [seg.end for seg in segs]
        boundary_metrics = compute_boundary_f1(pred_boundaries, gt_boundaries)

    if gt_segments is not None:
        gt_segs = _normalize_segments(gt_segments)
        iou_metrics = compute_segment_iou(segs, gt_segs)

    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "timeline": timeline,
        "label_duration_stats": duration_stats,
        "low_confidence_segments": low_confidence,
        "boundary_f1": boundary_metrics,
        "segment_iou": iou_metrics,
    }

    with (sample_root / "segments.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _write_summary_csv(
        output_path=sample_root / "summary.csv",
        timeline=timeline,
        duration_stats=duration_stats,
        low_confidence=low_confidence,
    )

    return payload
