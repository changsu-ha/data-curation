"""Algorithmic change-point detection using the ``ruptures`` library.

Two-step design
---------------
1. **Penalty auto-selection** (``select_penalty_elbow``): fit PELT at many
   log-spaced penalties, then find the "elbow" of the ``n_bkps`` vs
   ``log(penalty)`` curve via maximum second-derivative.  Falls back to the
   median candidate when the curve is monotone.

2. **Segmentation** (``run_ruptures``): run PELT or Binseg with the chosen
   penalty and return boundary indices.

No task-specific thresholds are used.  The only hyperparameters exposed are
general CPD algorithm settings (model kernel, min segment size, penalty
search range) — all configurable via :class:`RupturesConfig`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import ruptures as rpt  # type: ignore

    _RUPTURES_AVAILABLE = True
except ImportError:  # pragma: no cover
    rpt = None
    _RUPTURES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RupturesConfig:
    """Configuration for ruptures-based change-point detection."""

    # --- algorithm ---
    model: str = "rbf"
    """Cost model: ``rbf`` | ``l2`` | ``l1`` | ``cosine``."""

    algorithm: str = "pelt"
    """CPD algorithm: ``pelt`` | ``binseg`` | ``window`` | ``bottomup``."""

    min_size: int = 5
    """Minimum segment length in frames."""

    jump: int = 1
    """Subsampling step (higher = faster, less precise)."""

    # --- penalty ---
    penalty: float | str = "auto"
    """Penalty value or ``"auto"`` for elbow-based selection."""

    auto_method: str = "elbow"
    """Auto-selection method: ``elbow`` (only option currently)."""

    penalty_range: tuple[float, float] = (0.5, 200.0)
    """(min, max) search range for auto penalty selection."""

    n_penalty_steps: int = 40
    """Number of log-spaced penalty candidates."""

    # --- binseg specific ---
    max_n_bkps: int = 30
    """Maximum number of breakpoints for Binseg."""

    # --- post-processing ---
    min_gap_frames: int = 0
    """Merge boundaries closer than this many frames (0 = no merging)."""


# ---------------------------------------------------------------------------
# Penalty auto-selection
# ---------------------------------------------------------------------------

def select_penalty_elbow(
    signal: np.ndarray,
    config: RupturesConfig,
) -> tuple[float, dict[str, Any]]:
    """Select PELT penalty using the elbow of the n_bkps-vs-log(penalty) curve.

    Parameters
    ----------
    signal:
        (T, D) or (T,) feature array.
    config:
        Ruptures configuration.

    Returns
    -------
    chosen_penalty : float
    diagnostics : dict
        Keys: ``penalties``, ``n_bkps_list``, ``chosen_index``.
    """
    if not _RUPTURES_AVAILABLE:
        raise ImportError("ruptures is required. Install with: pip install ruptures")

    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    penalties = np.logspace(
        np.log10(max(config.penalty_range[0], 1e-6)),
        np.log10(config.penalty_range[1]),
        num=config.n_penalty_steps,
    )

    algo = rpt.Pelt(model=config.model, min_size=config.min_size, jump=config.jump)
    algo.fit(x)

    n_bkps_list: list[int] = []
    for pen in penalties:
        try:
            result = algo.predict(pen=pen)
            # ruptures returns exclusive end points including n; subtract 1 for n
            n_bkps_list.append(max(0, len(result) - 1))
        except Exception:
            n_bkps_list.append(0)

    n_arr = np.array(n_bkps_list, dtype=float)

    # Find elbow: point of maximum second derivative (most dramatic change)
    chosen_idx: int
    if len(n_arr) < 3 or np.all(n_arr == n_arr[0]):
        # Curve is constant or too short — use median
        chosen_idx = len(penalties) // 2
    else:
        d2 = np.gradient(np.gradient(n_arr))
        # We want the elbow going from many segments (low penalty) to fewer —
        # maximum second derivative marks the inflection point.
        chosen_idx = int(np.argmax(np.abs(d2)))

    chosen_penalty = float(penalties[chosen_idx])
    diagnostics: dict[str, Any] = {
        "penalties": penalties.tolist(),
        "n_bkps_list": n_bkps_list,
        "chosen_index": chosen_idx,
        "chosen_penalty": chosen_penalty,
    }
    return chosen_penalty, diagnostics


# ---------------------------------------------------------------------------
# Main segmentation function
# ---------------------------------------------------------------------------

def run_ruptures(
    signal: np.ndarray,
    config: RupturesConfig | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Detect change-point boundaries in ``signal``.

    Parameters
    ----------
    signal:
        (T, D) or (T,) feature array.
    config:
        Ruptures settings.  Defaults to :class:`RupturesConfig`.

    Returns
    -------
    boundaries : list[int]
        0-indexed boundary indices (exclusive end, **without** the terminal n).
    info : dict
        Diagnostics: chosen penalty, algorithm, n_bkps, etc.
    """
    if not _RUPTURES_AVAILABLE:
        warnings.warn(
            "ruptures is not installed; returning empty boundary list. "
            "Install with: pip install ruptures",
            stacklevel=2,
        )
        return [], {"error": "ruptures not available"}

    cfg = config or RupturesConfig()

    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    T = x.shape[0]

    # --- resolve penalty ---
    info: dict[str, Any] = {"algorithm": cfg.algorithm, "model": cfg.model}
    if isinstance(cfg.penalty, str) and cfg.penalty == "auto":
        try:
            pen, pen_diag = select_penalty_elbow(x, cfg)
            info["penalty_selection"] = pen_diag
        except Exception as exc:
            warnings.warn(f"Penalty auto-selection failed ({exc}); using penalty=10", stacklevel=2)
            pen = 10.0
    else:
        pen = float(cfg.penalty)

    info["penalty_used"] = pen

    # --- run algorithm ---
    try:
        if cfg.algorithm.lower() in ("pelt",):
            algo = rpt.Pelt(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
            algo.fit(x)
            result = algo.predict(pen=pen)

        elif cfg.algorithm.lower() == "binseg":
            algo = rpt.Binseg(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
            algo.fit(x)
            # Estimate n_bkps from penalty via PELT first
            pelt_algo = rpt.Pelt(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
            pelt_algo.fit(x)
            pelt_result = pelt_algo.predict(pen=pen)
            n_bkps = max(1, min(len(pelt_result) - 1, cfg.max_n_bkps))
            result = algo.predict(n_bkps=n_bkps)

        elif cfg.algorithm.lower() == "window":
            algo = rpt.Window(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
            algo.fit(x)
            result = algo.predict(pen=pen)

        elif cfg.algorithm.lower() == "bottomup":
            algo = rpt.BottomUp(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
            algo.fit(x)
            result = algo.predict(pen=pen)

        else:
            raise ValueError(f"Unknown ruptures algorithm: {cfg.algorithm!r}")

    except Exception as exc:
        warnings.warn(f"ruptures segmentation failed: {exc}", stacklevel=2)
        return [], {"error": str(exc), **info}

    # ruptures returns exclusive end indices including T at the end — remove it
    boundaries = sorted(set(r for r in result if 0 < r < T))
    info["n_bkps"] = len(boundaries)

    # Optional: merge too-close boundaries
    if cfg.min_gap_frames > 0:
        merged: list[int] = []
        for b in boundaries:
            if not merged or (b - merged[-1]) >= cfg.min_gap_frames:
                merged.append(b)
        boundaries = merged
        info["n_bkps_after_merge"] = len(boundaries)

    return boundaries, info


# ---------------------------------------------------------------------------
# Convenience: run PELT + Binseg in one call
# ---------------------------------------------------------------------------

def run_pelt_and_binseg(
    signal: np.ndarray,
    config: RupturesConfig | None = None,
) -> dict[str, tuple[list[int], dict]]:
    """Run both PELT and Binseg and return their results in a dict.

    Returns
    -------
    dict with keys ``"pelt"`` and ``"binseg"``, each mapping to
    ``(boundaries, info)``.
    """
    cfg_pelt = config or RupturesConfig()
    cfg_binseg = RupturesConfig(
        model=cfg_pelt.model,
        algorithm="binseg",
        min_size=cfg_pelt.min_size,
        jump=cfg_pelt.jump,
        penalty=cfg_pelt.penalty,
        auto_method=cfg_pelt.auto_method,
        penalty_range=cfg_pelt.penalty_range,
        n_penalty_steps=cfg_pelt.n_penalty_steps,
        max_n_bkps=cfg_pelt.max_n_bkps,
        min_gap_frames=cfg_pelt.min_gap_frames,
    )
    pelt_bounds, pelt_info = run_ruptures(signal, RupturesConfig(
        model=cfg_pelt.model, algorithm="pelt", min_size=cfg_pelt.min_size,
        jump=cfg_pelt.jump, penalty=cfg_pelt.penalty,
        auto_method=cfg_pelt.auto_method, penalty_range=cfg_pelt.penalty_range,
        n_penalty_steps=cfg_pelt.n_penalty_steps, max_n_bkps=cfg_pelt.max_n_bkps,
        min_gap_frames=cfg_pelt.min_gap_frames,
    ))
    binseg_bounds, binseg_info = run_ruptures(signal, cfg_binseg)
    return {
        "pelt": (pelt_bounds, pelt_info),
        "binseg": (binseg_bounds, binseg_info),
    }
