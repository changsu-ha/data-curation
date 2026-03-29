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
        return x
    return x.mean(axis=0)


def _downsample_segment(seg: np.ndarray, window_size: int) -> np.ndarray:
    """Down-sample segment to exactly ``window_size`` frames via uniform sub-sampling."""
    x = np.asarray(seg, dtype=float)
    T = x.shape[0]
    if T <= window_size:
        # Pad with last frame
        pad = window_size - T
        return np.vstack([x, np.tile(x[-1:], (pad, 1))]) if x.ndim == 2 else np.concatenate([x, np.full(pad, x[-1])])
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
    for k in range(n_clusters):
        idxs = [i for i, a in enumerate(assignments) if a == k]
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
    windows: np.ndarray,  # (n_windows, window_size * D) flat
    k: int,
    config: TiccConfig,
) -> list[int]:
    """Run fast_ticc for a fixed number of clusters.

    Returns cluster assignment per window.
    """
    assignments, _, _ = fast_ticc.ticc(
        windows,
        num_clusters=k,
        window_size=config.window_size,
        lambda_parameter=config.lambda_,
        beta=config.beta,
        maxIters=config.max_iter,
        random_state=config.random_state,
    )
    return [int(a) for a in assignments]


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

    if len(segments) < 2:
        warnings.warn("Need at least 2 segments for TICC; skipping.", stacklevel=2)
        return None

    # Prepare pooled feature matrix (n_segments, D) for silhouette
    pooled = np.array([_pool_segment(s, cfg.window_size) for s in segments])

    # Prepare windowed matrix (n_segments, window_size * D)
    ws_segs = [_downsample_segment(s, cfg.window_size) for s in segments]
    D = ws_segs[0].shape[-1] if ws_segs[0].ndim > 1 else 1
    windows = np.array([s.reshape(-1) for s in ws_segs])  # (n_segments, window_size * D)

    diagnostics: dict[str, Any] = {}

    # --- resolve k ---
    if isinstance(cfg.n_clusters, str) and cfg.n_clusters == "auto":
        best_k = 2
        best_sil: float = -2.0
        sil_scores: dict[int, float] = {}

        for k in range(2, min(cfg.max_k + 1, len(segments))):
            try:
                asgn = _run_ticc_fixed_k(windows, k, cfg)
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
        assignments = _run_ticc_fixed_k(windows, k_final, cfg)
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
