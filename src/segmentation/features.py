from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, filtfilt, savgol_filter
except Exception:  # pragma: no cover
    butter = filtfilt = savgol_filter = None

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover
    Rotation = None


Array = np.ndarray


@dataclass
class FeatureBuildConfig:
    """Configuration for feature construction."""

    dt: Optional[float] = None
    smoothing: str = "savgol"  # savgol | lowpass | none
    savgol_window: int = 11
    savgol_polyorder: int = 3
    lowpass_cutoff_hz: float = 8.0
    lowpass_order: int = 3
    normalize: str = "mad"  # mad | iqr | none
    modality_weights: Optional[Mapping[str, float]] = None


# -----------------------------
# preprocessing utilities
# -----------------------------
def resample_fixed_dt(
    timestamps: Array,
    signals: Mapping[str, Array],
    dt: float,
) -> Tuple[Array, Dict[str, Array]]:
    """Resample all signals to a fixed step using linear interpolation."""
    t = np.asarray(timestamps, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("timestamps must be a 1D array with at least 2 elements")

    t_new = np.arange(t[0], t[-1] + 0.5 * dt, dt, dtype=float)
    out: Dict[str, Array] = {}

    for name, value in signals.items():
        x = np.asarray(value, dtype=float)
        if x.shape[0] != len(t):
            raise ValueError(f"signal '{name}' length mismatch: {x.shape[0]} != {len(t)}")

        if x.ndim == 1:
            out[name] = np.interp(t_new, t, x)
        else:
            cols = [np.interp(t_new, t, x[:, c]) for c in range(x.shape[1])]
            out[name] = np.stack(cols, axis=1)

    return t_new, out


def smooth_signal(
    x: Array,
    dt: float,
    method: str = "savgol",
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
    lowpass_cutoff_hz: float = 8.0,
    lowpass_order: int = 3,
) -> Array:
    """Apply Savitzky-Golay or low-pass filter, with safe fallbacks."""
    x = np.asarray(x, dtype=float)
    if method == "none":
        return x

    if method == "savgol":
        if savgol_filter is None or x.shape[0] < 5:
            return _moving_average(x, k=5)
        window = max(5, savgol_window)
        if window % 2 == 0:
            window += 1
        window = min(window, x.shape[0] - (1 - x.shape[0] % 2))
        if window <= savgol_polyorder:
            return _moving_average(x, k=5)
        return savgol_filter(x, window_length=window, polyorder=savgol_polyorder, axis=0, mode="interp")

    if method == "lowpass":
        if butter is None or filtfilt is None or x.shape[0] < 8:
            return _moving_average(x, k=5)
        fs = 1.0 / dt
        nyquist = 0.5 * fs
        wn = min(0.99, max(1e-5, lowpass_cutoff_hz / nyquist))
        b, a = butter(lowpass_order, wn, btype="low", analog=False)
        return filtfilt(b, a, x, axis=0)

    raise ValueError(f"unknown smoothing method: {method}")


def unwrap_orientation(rotvec: Array) -> Array:
    """Resolve discontinuities in rotation-vector sequence."""
    r = np.asarray(rotvec, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("rotvec must have shape [T, 3]")
    out = r.copy()
    for i in range(1, len(out)):
        if np.linalg.norm(out[i] - out[i - 1]) > np.pi:
            out[i] *= -1.0
    return out


# -----------------------------
# feature builder
# -----------------------------
def build_features(sample: Mapping[str, Any], config: Optional[FeatureBuildConfig] = None) -> Dict[str, Any]:
    """
    Build segmentation features from one sample.

    Returns a dictionary with:
      - feature_matrix: [T, D]
      - timestamps: [T]
      - aux_signals: original + intermediate signals
    """
    cfg = config or FeatureBuildConfig()

    t = np.asarray(_pick(sample, "timestamps", "t", "time"), dtype=float)
    q = _to_2d(np.asarray(_pick(sample, "q", "joint_pos"), dtype=float))

    q_cmd_raw = sample.get("q_cmd")
    q_cmd = _to_2d(np.asarray(q_cmd_raw, dtype=float)) if q_cmd_raw is not None else np.zeros_like(q)

    pos = _extract_position(sample)
    quat = _extract_quaternion(sample)
    rotvec = quat_to_rotvec(quat)
    rotvec = unwrap_orientation(rotvec)

    grip = _extract_gripper(sample)

    dt = cfg.dt if cfg.dt is not None else float(np.median(np.diff(t)))
    signals = {"q": q, "q_cmd": q_cmd, "pos": pos, "rotvec": rotvec, "grip": grip}
    t_rs, rs = resample_fixed_dt(t, signals, dt)

    q_f = smooth_signal(rs["q"], dt, cfg.smoothing, cfg.savgol_window, cfg.savgol_polyorder, cfg.lowpass_cutoff_hz, cfg.lowpass_order)
    q_cmd_f = smooth_signal(rs["q_cmd"], dt, cfg.smoothing, cfg.savgol_window, cfg.savgol_polyorder, cfg.lowpass_cutoff_hz, cfg.lowpass_order)
    pos_f = smooth_signal(rs["pos"], dt, cfg.smoothing, cfg.savgol_window, cfg.savgol_polyorder, cfg.lowpass_cutoff_hz, cfg.lowpass_order)
    rot_f = smooth_signal(rs["rotvec"], dt, cfg.smoothing, cfg.savgol_window, cfg.savgol_polyorder, cfg.lowpass_cutoff_hz, cfg.lowpass_order)
    grip_f = smooth_signal(rs["grip"], dt, cfg.smoothing, cfg.savgol_window, cfg.savgol_polyorder, cfg.lowpass_cutoff_hz, cfg.lowpass_order)

    dq = np.gradient(q_f, dt, axis=0)
    ddq = np.gradient(dq, dt, axis=0)
    q_err = q_cmd_f - q_f

    dpos = np.gradient(pos_f, dt, axis=0)
    ddpos = np.gradient(dpos, dt, axis=0)
    drot = np.gradient(rot_f, dt, axis=0)
    ddrot = np.gradient(drot, dt, axis=0)

    grip_ratio = np.clip(grip_f, 0.0, 1.0)
    grip_binary = (grip_ratio > 0.5).astype(float)
    grip_event = np.abs(np.gradient(grip_binary.squeeze(-1) if grip_binary.ndim == 2 else grip_binary, axis=0, edge_order=1))
    grip_event = grip_event.reshape(-1, 1)
    grip_binary = _to_2d(grip_binary)
    grip_ratio = _to_2d(grip_ratio)

    modalities = {
        "joint": np.concatenate([q_f, dq, ddq, q_cmd_f, q_err], axis=1),
        "cartesian": np.concatenate([pos_f, rot_f, dpos, ddpos, drot, ddrot], axis=1),
        "gripper": np.concatenate([grip_binary, grip_ratio, grip_event], axis=1),
    }

    normalized = _normalize_and_weight(modalities, cfg.normalize, cfg.modality_weights)
    feature_matrix = np.concatenate([normalized["joint"], normalized["cartesian"], normalized["gripper"]], axis=1)

    aux = {
        "original": {
            "timestamps": t,
            "q": q,
            "q_cmd": q_cmd,
            "position": pos,
            "quaternion": quat,
            "gripper": grip,
        },
        "resampled": {
            "timestamps": t_rs,
            "q": q_f,
            "q_cmd": q_cmd_f,
            "position": pos_f,
            "rotvec": rot_f,
            "gripper": grip_f,
        },
    }

    return {
        "feature_matrix": feature_matrix,
        "timestamps": t_rs,
        "aux_signals": aux,
    }


# -----------------------------
# helpers
# -----------------------------
def _pick(sample: Mapping[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in sample:
            return sample[k]
    raise KeyError(f"missing keys; expected one of: {keys}")


def _to_2d(x: Array) -> Array:
    return x[:, None] if x.ndim == 1 else x


def _extract_position(sample: Mapping[str, Any]) -> Array:
    if "position" in sample:
        return _to_2d(np.asarray(sample["position"], dtype=float))
    if "cartesian" in sample:
        c = np.asarray(sample["cartesian"], dtype=float)
        if c.shape[1] >= 3:
            return c[:, :3]
    if "x" in sample and "y" in sample and "z" in sample:
        return np.column_stack([sample["x"], sample["y"], sample["z"]]).astype(float)
    raise KeyError("position signal not found")


def _extract_quaternion(sample: Mapping[str, Any]) -> Array:
    if "quaternion" in sample:
        q = np.asarray(sample["quaternion"], dtype=float)
        return _to_2d(q)
    if "cartesian" in sample:
        c = np.asarray(sample["cartesian"], dtype=float)
        if c.shape[1] >= 7:
            return c[:, 3:7]
    raise KeyError("quaternion signal not found")


def _extract_gripper(sample: Mapping[str, Any]) -> Array:
    for k in ("gripper", "gripper_opening", "grip", "gripper_ratio"):
        if k in sample:
            return _to_2d(np.asarray(sample[k], dtype=float))
    return np.zeros((len(np.asarray(_pick(sample, "timestamps", "t", "time"))), 1), dtype=float)


def quat_to_rotvec(quat_xyzw: Array) -> Array:
    q = np.asarray(quat_xyzw, dtype=float)
    if q.shape[1] != 4:
        raise ValueError("quaternion must have shape [T, 4] in xyzw")
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)

    if Rotation is not None:
        return Rotation.from_quat(q).as_rotvec()

    xyz = q[:, :3]
    w = q[:, 3]
    angle = 2.0 * np.arctan2(np.linalg.norm(xyz, axis=1), np.clip(w, -1.0, 1.0))
    axis = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12, None)
    return axis * angle[:, None]


def _moving_average(x: Array, k: int = 5) -> Array:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    if x.ndim == 1:
        x = x[:, None]
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    y = np.stack([np.convolve(xp[:, i], kernel, mode="valid") for i in range(x.shape[1])], axis=1)
    return y if y.shape[1] > 1 else y[:, 0]


def _robust_scale(x: Array, mode: str) -> Array:
    if mode == "none":
        return x
    med = np.median(x, axis=0, keepdims=True)
    xc = x - med
    if mode == "mad":
        s = np.median(np.abs(xc), axis=0, keepdims=True) * 1.4826
    elif mode == "iqr":
        q1 = np.percentile(x, 25, axis=0, keepdims=True)
        q3 = np.percentile(x, 75, axis=0, keepdims=True)
        s = q3 - q1
    else:
        raise ValueError(f"unknown normalize mode: {mode}")
    s = np.where(np.abs(s) < 1e-8, 1.0, s)
    return xc / s


def _normalize_and_weight(
    modalities: Mapping[str, Array],
    mode: str,
    modality_weights: Optional[Mapping[str, float]],
) -> Dict[str, Array]:
    weights = {"joint": 1.0, "cartesian": 1.0, "gripper": 1.0}
    if modality_weights:
        weights.update({k: float(v) for k, v in modality_weights.items()})

    out: Dict[str, Array] = {}
    for name, x in modalities.items():
        out[name] = _robust_scale(x, mode) * weights.get(name, 1.0)
    return out
