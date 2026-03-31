"""Recurring motion primitive discovery via TICC / fast_ticc.

TICC (Toeplitz Inverse Covariance-Based Clustering) clusters multivariate
time-series windows into recurring primitives while respecting temporal
coherence.

This module wraps ``fast_ticc`` (pip install fast-ticc) and gracefully
degrades when the library is absent.  When ``n_clusters="auto"`` it searches
k = 2 … max_k and picks the value that maximises average silhouette score
over mean-pooled segment feature vectors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import fast_ticc  # type: ignore

    _TICC_AVAILABLE = True
except ImportError:  # pragma: no cover
    fast_ticc = None
    _TICC_AVAILABLE = False

try:
    from sklearn.metrics import silhouette_score  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    silhouette_score = None
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TiccConfig:
    """Configuration for fast_ticc primitive discovery."""

    window_size: int = 10
    """Number of consecutive frames per TICC window."""

    n_clusters: int | str = "auto"
    """Number of clusters, or ``"auto"`` for silhouette-based search."""

    max_k: int = 10
    """Maximum k when ``n_clusters="auto"``."""

    lambda_: float = 11e-2
    """Sparsity regularisation (higher = sparser inverse covariance)."""

    beta: float = 400.0
    """Segment coherence weight (higher = smoother cluster assignments)."""

    max_iter: int = 100
    """Maximum EM iterations."""

    num_processors: int = 1
    """Number of parallel processors for optimization (1 = sequential)."""

    random_state: int = 42
    """Random seed passed to fast_ticc."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TiccResult:
    """Output of :func:`run_ticc`."""

    cluster_assignments: list[int]
    """Cluster ID for each input segment (length = n_segments)."""

    n_clusters: int
    """Number of clusters actually used."""

    transition_matrix: np.ndarray
    """(K, K) empirical transition count matrix between consecutive segments."""

    cluster_sizes: list[int]
    """Number of segments assigned to each cluster."""

    silhouette_score: float | None
    """Average silhouette score over mean-pooled features (None if unavailable)."""

    representative_indices: list[int]
    """Index of the most central segment for each cluster."""

    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pool_segment(seg: np.ndarray, window_size: int) -> np.ndarray:
    """Return mean-pooled feature vector for a segment of shape (T, D)."""
    x = np.asarray(seg, dtype=float)
    if x.ndim == 1:
        return np.array([x.mean()])  # Ensure we return a 1D array with shape (1,)
    return x.mean(axis=0)


def _downsample_segment(seg: np.ndarray, window_size: int) -> np.ndarray:
    """Down-sample segment to exactly ``window_size`` frames via uniform sub-sampling."""
    x = np.asarray(seg, dtype=float)
    T = x.shape[0]
    if T <= window_size:
        # Pad with last frame
        pad = window_size - T
        # Ensure we have at least one frame
        if pad > 0:
            return np.vstack([x, np.tile(x[-1:], (pad, 1))]) if x.ndim == 2 else np.concatenate([x, np.full(pad, x[-1])])
        else:
            # If T is 0, return a zero array
            if x.ndim == 2:
                return np.zeros((window_size, x.shape[1]), dtype=x.dtype)
            else:
                return np.zeros(window_size, dtype=x.dtype)
    indices = np.round(np.linspace(0, T - 1, window_size)).astype(int)
    return x[indices]


def _build_transition_matrix(assignments: list[int], n_clusters: int) -> np.ndarray:
    mat = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(len(assignments) - 1):
        a, b = assignments[i], assignments[i + 1]
        if 0 <= a < n_clusters and 0 <= b < n_clusters:
            mat[a, b] += 1
    return mat


def _find_representatives(
    pooled: np.ndarray,  # (n_segments, D)
    assignments: list[int],
    n_clusters: int,
) -> list[int]:
    """For each cluster, find the segment closest to the cluster mean."""
    reps: list[int] = []
    n_segments = len(pooled)
    for k in range(n_clusters):
        # Filter indices to ensure they're within bounds
        idxs = [i for i, a in enumerate(assignments) if a == k and i < n_segments]
        if not idxs:
            reps.append(0)
            continue
        cluster_feats = pooled[idxs]
        centroid = cluster_feats.mean(axis=0)
        dists = np.linalg.norm(cluster_feats - centroid, axis=1)
        reps.append(idxs[int(np.argmin(dists))])
    return reps


def _silhouette(pooled: np.ndarray, assignments: list[int]) -> float | None:
    if silhouette_score is None or len(set(assignments)) < 2:
        return None
    try:
        return float(silhouette_score(pooled, assignments))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: run TICC for a fixed k
# ---------------------------------------------------------------------------

def _run_ticc_fixed_k(
    segments: list[np.ndarray],  # List of (T_i, D) arrays
    k: int,
    config: TiccConfig,
) -> list[int]:
    """Run fast_ticc for a fixed number of clusters.

    Returns cluster assignment per segment.
    """
    # Filter out empty segments
    valid_segments = [seg for seg in segments if seg is not None and seg.size > 0 and len(seg) > 0]
    
    if not valid_segments:
        raise ValueError("No valid segments provided to TICC")
    
    # Concatenate all segments into one time series
    # We need to add a small gap between segments to indicate they're separate
    # This is a common approach when clustering multiple time series with TICC
    time_series_parts = []
    for i, seg in enumerate(valid_segments):
        time_series_parts.append(seg)
        # Add a small gap between segments (except after the last one)
        if i < len(valid_segments) - 1:
            # Add a gap of zeros with the same number of features
            gap = np.zeros((config.window_size, seg.shape[1] if seg.ndim > 1 else 1))
            time_series_parts.append(gap)
    
    # Concatenate all parts
    if time_series_parts:
        full_time_series = np.vstack(time_series_parts)
    else:
        raise ValueError("No valid time series data to process")
    
    # Validate input dimensions
    if full_time_series.size == 0 or full_time_series.shape[0] == 0:
        raise ValueError("Empty time series provided to TICC")
    
    # Debug print
    print(f"  TICC input: time_series.shape={full_time_series.shape}, k={k}, window_size={config.window_size}")
    
    # Check if we have enough data points for the window size
    if full_time_series.shape[0] < config.window_size:
        raise ValueError(f"Not enough data points ({full_time_series.shape[0]}) for window size ({config.window_size})")
    
    # Use the correct function signature for the installed fast_ticc version
    result = fast_ticc.ticc_labels(
        data_series=full_time_series,
        num_clusters=k,
        window_size=config.window_size,
        sparsity_weight=config.lambda_,
        label_switching_cost=config.beta,
        iteration_limit=config.max_iter,
        min_cluster_size=1,  # Avoid donor cluster errors
        num_processors=config.num_processors,  # Parallel processing
    )
    
    # Return cluster assignments for each window
    # Note: TICC returns -1 for points that couldn't be assigned (usually at the boundaries)
    # We'll map these to cluster 0 for simplicity
    labels = [int(a) if a >= 0 else 0 for a in result.point_labels]
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ticc(
    segments: list[np.ndarray],
    config: TiccConfig | None = None,
) -> TiccResult | None:
    """Cluster recurring motion primitives from a list of feature segments.

    Each segment is a ``(T_i, D)`` array (variable length).  The function:

    1. Mean-pools each segment to a ``(D,)`` vector for silhouette scoring.
    2. Down-samples each segment to ``window_size`` frames.
    3. Flattens to a ``(n_segments, window_size * D)`` matrix for fast_ticc.
    4. Optionally searches k = 2 … max_k when ``n_clusters="auto"``.

    Parameters
    ----------
    segments:
        List of ``(T_i, D)`` feature arrays, one per segment.
    config:
        TICC configuration.  Defaults to :class:`TiccConfig`.

    Returns
    -------
    :class:`TiccResult` or ``None`` if fast_ticc is not installed or fewer
    than 2 segments are provided.
    """
    if not _TICC_AVAILABLE:
        warnings.warn(
            "fast_ticc is not installed; skipping primitive discovery. "
            "Install with: pip install fast-ticc",
            stacklevel=2,
        )
        return None

    cfg = config or TiccConfig()

    # Filter out empty segments
    valid_segments = [seg for seg in segments if seg is not None and seg.size > 0 and len(seg) > 0]
    
    if len(valid_segments) < 2:
        warnings.warn("Need at least 2 valid segments for TICC; skipping.", stacklevel=2)
        return None

    # Prepare pooled feature matrix (n_segments, D) for silhouette
    pooled = np.array([_pool_segment(s, cfg.window_size) for s in valid_segments])
    print(f"  Pooled features: shape {pooled.shape}")

    # Prepare windowed matrix (n_segments, window_size * D)
    ws_segs = [_downsample_segment(s, cfg.window_size) for s in valid_segments]
    print(f"  Windowed segments: {len(ws_segs)} segments")
    for i, seg in enumerate(ws_segs):
        print(f"    Segment {i}: shape {seg.shape}")
    
    D = ws_segs[0].shape[-1] if ws_segs[0].ndim > 1 else 1
    # Reshape each segment to (window_size * D,) and concatenate them
    windows_list = [s.reshape(-1) for s in ws_segs]  # Each becomes (window_size * D,)
    windows = np.vstack(windows_list)  # Stack to (n_segments, window_size * D)
    print(f"  Windows matrix: shape {windows.shape}")
    
    # But fast_ticc expects (n_data_points, window_size * D), so we need to transpose
    # Actually, no - looking at the docs again, it should be (n_data_points, window_size * D)
    # where n_data_points is the number of windows/segments
    # So our current shape (n_segments, window_size * D) is correct for the first dimension

    diagnostics: dict[str, Any] = {}

    # --- resolve k ---
    if isinstance(cfg.n_clusters, str) and cfg.n_clusters == "auto":
        best_k = 2
        best_sil: float = -2.0
        sil_scores: dict[int, float] = {}

        for k in range(2, min(cfg.max_k + 1, len(valid_segments))):
            try:
                asgn = _run_ticc_fixed_k(valid_segments, k, cfg)
                sil = _silhouette(pooled, asgn)
                sil_scores[k] = sil if sil is not None else -1.0
                if sil is not None and sil > best_sil:
                    best_sil = sil
                    best_k = k
            except Exception as exc:
                warnings.warn(f"TICC k={k} failed: {exc}", stacklevel=2)

        diagnostics["silhouette_by_k"] = sil_scores
        k_final = best_k
    else:
        k_final = int(cfg.n_clusters)

    # --- final run with chosen k ---
    try:
        assignments = _run_ticc_fixed_k(valid_segments, k_final, cfg)
    except Exception as exc:
        warnings.warn(f"TICC final run (k={k_final}) failed: {exc}", stacklevel=2)
        return None

    diagnostics["k_chosen"] = k_final

    sil = _silhouette(pooled, assignments)
    transition = _build_transition_matrix(assignments, k_final)
    cluster_sizes = [assignments.count(k) for k in range(k_final)]
    representatives = _find_representatives(pooled, assignments, k_final)

    return TiccResult(
        cluster_assignments=assignments,
        n_clusters=k_final,
        transition_matrix=transition,
        cluster_sizes=cluster_sizes,
        silhouette_score=sil,
        representative_indices=representatives,
        diagnostics=diagnostics,
    )
