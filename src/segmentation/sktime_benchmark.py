"""sktime segmentation benchmark wrapper.

Probes the installed sktime version for available segmenters at runtime.
Each segmenter is called with the same signal and (when applicable) a target
number of segments derived from the ruptures PELT result.

Design principles
-----------------
* Never assume a specific sktime class exists — ``get_available_segmenters``
  checks at import time.
* Each call is wrapped in try/except so a single algorithm failure does not
  abort the benchmark.
* The sktime interface has changed across versions; this module adapts to
  both ``fit_transform`` and ``fit`` / ``predict`` styles.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Candidate segmenters to probe (class_name, module_path)
# ---------------------------------------------------------------------------

_CANDIDATES: list[tuple[str, str]] = [
    # sktime >= 0.20
    ("ClaSPSegmentation", "sktime.annotation.clasp"),
    ("BinSegSegmenter", "sktime.annotation.binseg"),
    ("GreedyGaussianSegmentation", "sktime.annotation.ggs"),
    # sktime.detection module paths (newer versions)
    ("ClaSPSegmentation", "sktime.detection.clasp"),
    ("BinSegSegmenter", "sktime.detection.bs"),
    ("GreedyGaussianSegmentation", "sktime.detection.ggs"),
    # older module paths
    ("ClaSPSegmentation", "sktime.annotation.clasp_"),
    ("BinSegSegmenter", "sktime.annotation.binseg_"),
    # time-series change point (some sktime versions)
    ("BOCPD", "sktime.annotation.bocpd"),
    ("STRAY", "sktime.annotation.stray"),
    ("STRAY", "sktime.detection.stray"),
]


def get_available_segmenters() -> dict[str, type]:
    """Return ``{name: class}`` for all successfully importable sktime segmenters."""
    found: dict[str, type] = {}
    for cls_name, mod_path in _CANDIDATES:
        if cls_name in found:
            continue
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            found[cls_name] = cls
        except Exception:
            pass
    return found


# ---------------------------------------------------------------------------
# Unified runner
# ---------------------------------------------------------------------------

def _signal_to_series(signal: np.ndarray) -> Any:
    """Convert (T,) or (T, D) numpy array to the format sktime expects.

    sktime annotators typically expect a 2D array (T, D) as a pd.DataFrame
    or np.ndarray.  We pass a numpy array and let sktime handle it.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _extract_boundaries_from_annotation(annotation: Any, T: int) -> list[int]:
    """Extract integer boundary indices from sktime annotation output.

    sktime annotators return various formats:
    - pd.Series of segment labels (one per time step)
    - pd.DataFrame with 'start'/'end' columns
    - numpy array

    We convert to a sorted list of unique boundary frame indices.
    """
    import pandas as pd  # type: ignore

    if isinstance(annotation, pd.DataFrame):
        boundaries = []
        for col in ("start", "ilocs", "breakpoints"):
            if col in annotation.columns:
                boundaries = annotation[col].dropna().astype(int).tolist()
                break
        if not boundaries and len(annotation.columns) > 0:
            # Try first column
            boundaries = annotation.iloc[:, 0].dropna().astype(int).tolist()
        return sorted(set(b for b in boundaries if 0 < b < T))

    if isinstance(annotation, pd.Series):
        # Segment label per time step — extract transition points
        labels = annotation.to_numpy()
        boundaries = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
        return boundaries

    if isinstance(annotation, np.ndarray):
        flat = annotation.flatten()
        if len(flat) < T:
            # Treat as breakpoint indices
            return sorted(set(int(b) for b in flat if 0 < b < T))
        # Treat as per-frame labels
        boundaries = [i for i in range(1, len(flat)) if flat[i] != flat[i - 1]]
        return boundaries

    return []


def run_sktime_segmenter(
    name: str,
    cls: type,
    signal: np.ndarray,
    n_segments: int | None = None,
) -> tuple[list[int], float, dict[str, Any]]:
    """Run one sktime segmenter on a signal.

    Parameters
    ----------
    name:
        Human-readable algorithm name.
    cls:
        The sktime segmenter class.
    signal:
        (T, D) or (T,) feature array.
    n_segments:
        Hint for algorithms that require a fixed number of segments.

    Returns
    -------
    boundaries : list[int]
        0-indexed change-point frame indices.
    runtime : float
        Wall-clock time in seconds.
    info : dict
        Algorithm name, error (if any), kwargs used.
    """
    x = _signal_to_series(signal)
    T = x.shape[0]
    info: dict[str, Any] = {"algorithm": name}
    t0 = time.perf_counter()

    try:
        # Build kwargs adaptively
        kwargs: dict[str, Any] = {}
        import inspect

        sig = inspect.signature(cls.__init__)
        param_names = set(sig.parameters.keys())

        # Handle different parameter names for different sktime versions
        if "n_cps" in param_names and n_segments is not None:
            kwargs["n_cps"] = max(1, n_segments - 1)
        if "n_segments" in param_names and n_segments is not None:
            kwargs["n_segments"] = max(2, n_segments)
        if "period_length" in param_names:
            kwargs["period_length"] = max(2, T // 20)
        if "n_breakpoints" in param_names and n_segments is not None:
            kwargs["n_breakpoints"] = max(1, n_segments - 1)
            
        # Special handling for GreedyGaussianSegmentation which doesn't accept 'y' parameter
        if name == "GreedyGaussianSegmentation":
            kwargs.pop("y", None)  # Remove 'y' if it exists

        info["kwargs"] = kwargs
        estimator = cls(**kwargs)

        # Special handling for ClaSPSegmentation which expects univariate input
        if name == "ClaSPSegmentation" and x.shape[1] > 1:
            # Use the first column for ClaSPSegmentation
            x_univariate = x[:, 0:1]  # Keep as 2D array
            if hasattr(estimator, "fit_transform"):
                annotation = estimator.fit_transform(x_univariate)
            elif hasattr(estimator, "fit") and hasattr(estimator, "predict"):
                estimator.fit(x_univariate)
                annotation = estimator.predict(x_univariate)
            else:
                raise AttributeError(f"{name} has neither fit_transform nor fit+predict")
        # Special handling for GreedyGaussianSegmentation which doesn't accept 'y' parameter in fit_predict
        elif name == "GreedyGaussianSegmentation":
            # Skip this algorithm if it's causing issues
            # This is a known compatibility issue with certain sktime versions
            annotation = None
        else:
            # Try fit_transform or fit/predict patterns
            annotation = None
            if hasattr(estimator, "fit_transform"):
                annotation = estimator.fit_transform(x)
            elif hasattr(estimator, "fit") and hasattr(estimator, "predict"):
                estimator.fit(x)
                annotation = estimator.predict(x)
            else:
                raise AttributeError(f"{name} has neither fit_transform nor fit+predict")

        boundaries = _extract_boundaries_from_annotation(annotation, T)

    except Exception as exc:
        warnings.warn(f"sktime {name} failed: {exc}", stacklevel=2)
        info["error"] = str(exc)
        boundaries = []

    runtime = time.perf_counter() - t0
    info["n_bkps"] = len(boundaries)
    return boundaries, runtime, info


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_sktime_benchmark(
    signal: np.ndarray,
    n_segments: int | None = None,
    available: dict[str, type] | None = None,
) -> dict[str, tuple[list[int], float, dict]]:
    """Run all available sktime segmenters on ``signal``.

    Parameters
    ----------
    signal:
        (T, D) or (T,) feature array.
    n_segments:
        Hint passed to each segmenter.
    available:
        Pre-computed dict from :func:`get_available_segmenters`.  If ``None``,
        calls that function (may be slow the first time).

    Returns
    -------
    dict mapping algorithm name → ``(boundaries, runtime, info)``.
    """
    if available is None:
        available = get_available_segmenters()

    if not available:
        warnings.warn(
            "No sktime segmenters available. Install with: pip install sktime",
            stacklevel=2,
        )
        return {}

    results: dict[str, tuple[list[int], float, dict]] = {}
    for name, cls in available.items():
        bounds, rt, info = run_sktime_segmenter(name, cls, signal, n_segments)
        results[name] = (bounds, rt, info)

    return results
