"""Unsupervised / weakly supervised evaluation metrics for segmentation.

When no ground-truth labels exist we compare methods using:
- **Boundary agreement** (tolerance-based precision/recall/F1)
- **Segment duration statistics** (count, mean/std/min/max durations)
- **Cluster silhouette** (average silhouette over mean-pooled features)
- **Runtime comparison** (Markdown table)

When ground-truth *is* available:
- Precision/recall/F1 from :func:`boundary_agreement` still applies.
- :func:`compute_boundary_f1` is preserved for compatibility with
  ``segmentation.report``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Safe formatting helpers
# ---------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def fmt_float(value: Any, precision: int = 3, default: str = "N/A") -> str:
    """Safely format floats, returning ``default`` for None/NaN."""
    if _is_missing(value):
        return default
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return default


def fmt_int(value: Any, default: str = "N/A") -> str:
    """Safely format integers, returning ``default`` for None/NaN."""
    if _is_missing(value):
        return default
    try:
        return str(int(value))
    except Exception:
        return default


def fmt_percent(value: Any, precision: int = 1, default: str = "N/A") -> str:
    """Safely format ratios as percentages."""
    if _is_missing(value):
        return default
    try:
        return f"{100.0 * float(value):.{precision}f}%"
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Boundary agreement
# ---------------------------------------------------------------------------

def boundary_agreement(
    pred: list[int],
    ref: list[int],
    tolerance: int = 5,
) -> dict[str, float]:
    """Precision, recall, and F1 for boundary detection.

    A predicted boundary is a true positive if there is a reference boundary
    within ``±tolerance`` frames.  Each reference boundary is matched at most
    once (greedy closest-first matching).

    Parameters
    ----------
    pred, ref:
        Sorted lists of integer boundary frame indices.
    tolerance:
        Match window in frames (symmetric).

    Returns
    -------
    dict with keys ``precision``, ``recall``, ``f1``, ``n_pred``, ``n_ref``,
    ``n_tp``.
    """
    if not pred or not ref:
        p = 1.0 if not pred and not ref else 0.0
        r = 1.0 if not pred and not ref else 0.0
        f1 = 1.0 if p + r == 0 else 2 * p * r / (p + r)
        return {
            "precision": p, "recall": r, "f1": f1,
            "n_pred": len(pred), "n_ref": len(ref), "n_tp": 0,
        }

    pred_s = sorted(pred)
    ref_remaining = sorted(ref)

    tp = 0
    for b in pred_s:
        best_dist = tolerance + 1
        best_idx = -1
        for i, r in enumerate(ref_remaining):
            d = abs(b - r)
            if d <= tolerance and d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            tp += 1
            ref_remaining.pop(best_idx)

    precision = tp / len(pred_s) if pred_s else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(pred_s),
        "n_ref": len(ref),
        "n_tp": tp,
    }


# ---------------------------------------------------------------------------
# Segment duration statistics
# ---------------------------------------------------------------------------

def segment_duration_stats(
    boundaries: list[int],
    total_frames: int,
    fps: float,
) -> dict[str, float]:
    """Summary statistics of segment durations.

    Parameters
    ----------
    boundaries:
        Sorted boundary frame indices (exclusive ends, not including 0 or T).
    total_frames:
        Total number of frames (T).
    fps:
        Frames per second for converting to seconds.

    Returns
    -------
    dict with ``n_segments``, ``mean_s``, ``std_s``, ``min_s``, ``max_s``,
    ``total_s``.
    """
    # Build segment frame lengths
    edges = [0] + sorted(boundaries) + [total_frames]
    lengths_frames = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    lengths_s = [l / fps for l in lengths_frames]

    if not lengths_s:
        return {
            "n_segments": 0,
            "mean_s": 0.0, "std_s": 0.0, "min_s": 0.0, "max_s": 0.0, "total_s": 0.0,
        }

    arr = np.array(lengths_s)
    return {
        "n_segments": len(arr),
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "total_s": float(arr.sum()),
    }


# ---------------------------------------------------------------------------
# Cluster silhouette
# ---------------------------------------------------------------------------

def cluster_silhouette(
    features_per_segment: list[np.ndarray],
    labels: list[int],
) -> float | None:
    """Average silhouette score using mean-pooled segment features.

    Parameters
    ----------
    features_per_segment:
        List of ``(T_i, D)`` arrays, one per segment.
    labels:
        Cluster assignment for each segment (same length as list).

    Returns
    -------
    float or None (if sklearn unavailable or fewer than 2 unique labels).
    """
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except ImportError:
        return None

    if len(set(labels)) < 2 or len(labels) < 2:
        return None

    pooled = np.array([
        np.asarray(s, dtype=float).mean(axis=0) if np.asarray(s).ndim > 1
        else np.asarray(s, dtype=float)
        for s in features_per_segment
    ])
    try:
        return float(silhouette_score(pooled, labels))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Runtime table
# ---------------------------------------------------------------------------

def runtime_table(runtimes: dict[str, float]) -> str:
    """Format a method → runtime dict as a Markdown table."""
    if not runtimes:
        return "| Method | Runtime (s) |\n|--------|-------------|\n"

    lines = ["| Method | Runtime (s) |", "|--------|-------------|"]
    def _sort_key(item: tuple[str, Any]) -> float:
        value = item[1]
        if _is_missing(value):
            return float("inf")
        try:
            return float(value)
        except Exception:
            return float("inf")

    for method, rt in sorted(runtimes.items(), key=_sort_key):
        lines.append(f"| {method} | {fmt_float(rt, precision=4)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# All-pairs comparison
# ---------------------------------------------------------------------------

def compare_boundaries(
    results: dict[str, list[int]],
    total_frames: int,
    fps: float,
    tolerance_frames: int = 5,
) -> dict[str, Any]:
    """Compare all methods pairwise and compute per-method duration statistics.

    Parameters
    ----------
    results:
        ``{method_name: boundary_list}``
    total_frames:
        Number of frames in the signal.
    fps:
        Frames per second.
    tolerance_frames:
        Agreement tolerance in frames.

    Returns
    -------
    dict with:
    - ``duration_stats``: ``{method: stats_dict}``
    - ``pairwise_agreement``: ``{(ref, pred): boundary_agreement_dict}``
    - ``n_segments``: ``{method: int}``
    """
    duration_stats: dict[str, dict] = {}
    n_segs: dict[str, int] = {}
    pairwise: dict[str, dict] = {}

    for name, bounds in results.items():
        stats = segment_duration_stats(bounds, total_frames, fps)
        duration_stats[name] = stats
        n_segs[name] = stats["n_segments"]

    method_names = list(results.keys())
    for i, ref_name in enumerate(method_names):
        for pred_name in method_names[i + 1:]:
            key = f"{ref_name}_vs_{pred_name}"
            pairwise[key] = boundary_agreement(
                results[pred_name], results[ref_name], tolerance=tolerance_frames
            )

    return {
        "duration_stats": duration_stats,
        "pairwise_agreement": pairwise,
        "n_segments": n_segs,
    }


# ---------------------------------------------------------------------------
# Aggregate across episodes
# ---------------------------------------------------------------------------

def aggregate_episode_comparisons(
    episode_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Average per-episode comparison dicts across all episodes.

    Parameters
    ----------
    episode_comparisons:
        List of dicts from :func:`compare_boundaries`, one per episode.

    Returns
    -------
    dict with same structure but values averaged across episodes.
    """
    if not episode_comparisons:
        return {}

    # Aggregate duration stats
    all_methods = list(episode_comparisons[0].get("duration_stats", {}).keys())
    duration_agg: dict[str, dict[str, list]] = {m: {} for m in all_methods}
    for ep_cmp in episode_comparisons:
        for m, stats in ep_cmp.get("duration_stats", {}).items():
            for k, v in stats.items():
                duration_agg.setdefault(m, {}).setdefault(k, []).append(v)

    avg_duration: dict[str, dict[str, float]] = {}
    for m, fields in duration_agg.items():
        avg_duration[m] = {}
        for k, values in fields.items():
            numeric = [float(v) for v in values if not _is_missing(v)]
            avg_duration[m][k] = float(np.mean(numeric)) if numeric else float("nan")

    # Aggregate pairwise
    all_pairs = list(episode_comparisons[0].get("pairwise_agreement", {}).keys())
    pair_agg: dict[str, dict[str, list]] = {p: {} for p in all_pairs}
    for ep_cmp in episode_comparisons:
        for pair, metrics in ep_cmp.get("pairwise_agreement", {}).items():
            for k, v in metrics.items():
                pair_agg.setdefault(pair, {}).setdefault(k, []).append(v)

    avg_pairwise: dict[str, dict[str, float]] = {}
    for pair, fields in pair_agg.items():
        avg_pairwise[pair] = {}
        for k, values in fields.items():
            numeric = [float(v) for v in values if not _is_missing(v)]
            avg_pairwise[pair][k] = float(np.mean(numeric)) if numeric else float("nan")

    return {
        "duration_stats": avg_duration,
        "pairwise_agreement": avg_pairwise,
        "n_episodes": len(episode_comparisons),
    }


# ---------------------------------------------------------------------------
# Markdown report helpers
# ---------------------------------------------------------------------------

def format_comparison_report(
    aggregated: dict[str, Any],
    runtimes: dict[str, dict[str, float]] | None = None,
    modality: str = "unknown",
) -> str:
    """Format aggregated comparison results as a Markdown section.

    Parameters
    ----------
    aggregated:
        Output of :func:`aggregate_episode_comparisons`.
    runtimes:
        ``{method: {episode_id: runtime_s}}`` optional runtimes.
    modality:
        Label for the modality (``"joint"`` or ``"cartesian"``).
    """
    lines = [f"## Modality: {modality}", ""]

    # Duration stats table
    dur = aggregated.get("duration_stats", {})
    if dur:
        lines += ["### Segment Duration Statistics (averaged across episodes)", ""]
        header_keys = ["n_segments", "mean_s", "std_s", "min_s", "max_s"]
        lines.append("| Method | " + " | ".join(header_keys) + " |")
        lines.append("|--------|" + "|".join(["-------"] * len(header_keys)) + "|")
        for method, stats in sorted(dur.items()):
            vals = [fmt_float(stats.get(k), precision=3) for k in header_keys]
            lines.append(f"| {method} | " + " | ".join(vals) + " |")
        lines.append("")

    # Pairwise agreement
    pw = aggregated.get("pairwise_agreement", {})
    if pw:
        lines += ["### Boundary Agreement (averaged across episodes)", ""]
        lines.append("| Pair | Precision | Recall | F1 |")
        lines.append("|------|-----------|--------|----|")
        for pair, metrics in sorted(pw.items()):
            p = fmt_float(metrics.get("precision"), precision=3)
            r = fmt_float(metrics.get("recall"), precision=3)
            f = fmt_float(metrics.get("f1"), precision=3)
            lines.append(f"| {pair} | {p} | {r} | {f} |")
        lines.append("")

    # Runtime table
    if runtimes:
        avg_rt: dict[str, float] = {}
        for method, ep_rts in runtimes.items():
            vals = [float(value) for value in ep_rts.values() if not _is_missing(value)]
            avg_rt[method] = float(np.mean(vals)) if vals else float("nan")
        lines += ["### Average Runtime", "", runtime_table(avg_rt), ""]

    return "\n".join(lines)
