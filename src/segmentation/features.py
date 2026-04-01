from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

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

_MODALITY_ALIASES = {"ablation": "joint_command"}
_VALID_MODALITIES = {"joint", "cartesian", "joint_command"}


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
    joint_groups: Optional[Mapping[str, list[str]]] = None  # Group name -> list of joint names
    joint_weights: Optional[Mapping[str, float]] = None  # Joint name -> weight


def normalize_modality_name(modality: str) -> str:
    """Normalize modality aliases and validate the final name."""
    normalized = _MODALITY_ALIASES.get(modality.strip().lower(), modality.strip().lower())
    if normalized not in _VALID_MODALITIES:
        raise ValueError(
            f"unknown modality: {modality!r}; expected one of "
            f"{sorted(_VALID_MODALITIES | set(_MODALITY_ALIASES))}"
        )
    return normalized


# -----------------------------
# preprocessing utilities
# -----------------------------
def resample_fixed_dt(
    timestamps: Array,
    signals: Mapping[str, Array],
    dt: float,
) -> tuple[Array, Dict[str, Array]]:
    """Resample all signals to a fixed step using linear interpolation."""
    t = np.asarray(timestamps, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("timestamps must be a 1D array with at least 2 elements")
    if not np.all(np.diff(t) > 0):
        raise ValueError("timestamps must be strictly increasing")

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
        return savgol_filter(
            x,
            window_length=window,
            polyorder=savgol_polyorder,
            axis=0,
            mode="interp",
        )

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
def build_features(
    sample: Mapping[str, Any],
    modality: str | FeatureBuildConfig | None = None,
    config: Optional[FeatureBuildConfig] = None,
) -> Dict[str, Any]:
    """
    Build segmentation features from one sample for a single modality.

    Returns a dictionary with:
      - matrix: [T, D]
      - feature_names: ordered feature channel names
      - timestamps: [T]
      - source_channels: original signals used for the modality

    Compatibility:
      - ``build_features(sample, config)`` is still accepted.
      - ``feature_matrix`` is returned as an alias of ``matrix``.
    """
    if isinstance(modality, FeatureBuildConfig):
        config = modality
        modality = None

    cfg = config or FeatureBuildConfig()
    resolved_modality = (
        normalize_modality_name(modality)
        if isinstance(modality, str)
        else _infer_modality(sample)
    )

    timestamps = np.asarray(_pick(sample, "timestamps", "t", "time"), dtype=float)
    dt = _resolve_dt(timestamps, cfg.dt)

    if resolved_modality == "joint":
        built = _build_joint_features(sample, timestamps, dt, cfg)
    elif resolved_modality == "joint_command":
        built = _build_joint_command_features(sample, timestamps, dt, cfg)
    elif resolved_modality == "cartesian":
        built = _build_cartesian_features(sample, timestamps, dt, cfg)
    else:  # pragma: no cover
        raise ValueError(f"unsupported modality: {resolved_modality}")

    return {
        "modality": resolved_modality,
        "matrix": built["matrix"],
        "feature_matrix": built["matrix"],
        "feature_names": built["feature_names"],
        "timestamps": built["timestamps"],
        "source_channels": built["source_channels"],
        "aux_signals": built["aux_signals"],
        "raw_matrices": built.get("raw_matrices", {}),  # Include raw matrices if available
    }


# -----------------------------
# modality builders
# -----------------------------
def _build_joint_features(
    sample: Mapping[str, Any],
    timestamps: Array,
    dt: float,
    cfg: FeatureBuildConfig,
) -> Dict[str, Any]:
    q = _to_2d(np.asarray(_pick(sample, "q", "joint_pos"), dtype=float))
    
    # Handle joint grouping if specified
    if cfg.joint_groups and len(cfg.joint_groups) > 0:
        return _build_grouped_joint_features(sample, timestamps, dt, cfg, q)
    
    t_rs, filtered = _resample_and_smooth({"q": q}, timestamps, dt, cfg)

    q_f = filtered["q"]
    dq = _gradient(q_f, dt)
    ddq = _gradient(dq, dt)

    blocks = {"joint": np.concatenate([q_f, dq, ddq], axis=1)}
    block_feature_names = {
        "joint": (
            _channel_names("q", q_f.shape[1])
            + _channel_names("dq", dq.shape[1])
            + _channel_names("ddq", ddq.shape[1])
        )
    }

    matrix, feature_names = _finalize_blocks(blocks, block_feature_names, cfg, ["joint"])
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "timestamps": t_rs,
        "source_channels": {"joint": ["q"]},
        "aux_signals": {
            "original": {"timestamps": timestamps, "q": q},
            "resampled": {"timestamps": t_rs, "q": q_f, "dq": dq, "ddq": ddq},
        },
    }


def _build_grouped_joint_features(
    sample: Mapping[str, Any],
    timestamps: Array,
    dt: float,
    cfg: FeatureBuildConfig,
    q: Array,
) -> Dict[str, Any]:
    """Build joint features grouped by joint names."""
    # For grouped processing, we need to know the joint names
    # This would typically come from the dataset schema or configuration
    joint_names = sample.get("joint_names", [f"joint_{i}" for i in range(q.shape[1])])
    
    # Create mapping from joint name to index
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    
    # Build features for each group
    group_blocks = {}
    group_feature_names = {}
    
    # Initialize t_rs with the original timestamps (will be updated if we process groups)
    t_rs = timestamps
    
    for group_name, group_joint_names in cfg.joint_groups.items():
        # Get indices for this group
        indices = [name_to_idx[name] for name in group_joint_names if name in name_to_idx]
        if not indices:
            continue
            
        # Extract group data
        q_group = q[:, indices]
        
        # Process group data
        t_rs, filtered = _resample_and_smooth({f"q_{group_name}": q_group}, timestamps, dt, cfg)
        q_f = filtered[f"q_{group_name}"]
        dq = _gradient(q_f, dt)
        ddq = _gradient(dq, dt)
        
        # Store group features
        group_blocks[group_name] = np.concatenate([q_f, dq, ddq], axis=1)
        group_feature_names[group_name] = (
            _channel_names(f"q_{group_name}", q_f.shape[1])
            + _channel_names(f"dq_{group_name}", dq.shape[1])
            + _channel_names(f"ddq_{group_name}", ddq.shape[1])
        )
    
    # Handle case when no groups were processed
    if not group_blocks:
        # Return empty feature matrix
        return {
            "matrix": np.empty((len(timestamps), 0)),
            "feature_names": [],
            "timestamps": timestamps,
            "source_channels": {"joint_groups": []},
            "aux_signals": {
                "original": {"timestamps": timestamps, "q": q},
                "resampled": {"timestamps": timestamps, "q_groups": {}},
            },
        }
    
    # Combine all groups
    matrix, feature_names = _finalize_blocks(group_blocks, group_feature_names, cfg, list(group_blocks.keys()))
    
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "timestamps": t_rs,
        "source_channels": {"joint_groups": list(cfg.joint_groups.keys())},
        "aux_signals": {
            "original": {"timestamps": timestamps, "q": q},
            "resampled": {"timestamps": t_rs, "q_groups": group_blocks},
        },
    }


def _build_joint_command_features(
    sample: Mapping[str, Any],
    timestamps: Array,
    dt: float,
    cfg: FeatureBuildConfig,
) -> Dict[str, Any]:
    q = _to_2d(np.asarray(_pick(sample, "q", "joint_pos"), dtype=float))
    q_cmd = _to_2d(np.asarray(_pick(sample, "q_cmd", "joint_cmd"), dtype=float))
    if q.shape != q_cmd.shape:
        raise ValueError(f"q and q_cmd shape mismatch: {q.shape} != {q_cmd.shape}")

    # Handle joint grouping if specified
    if cfg.joint_groups and len(cfg.joint_groups) > 0:
        return _build_grouped_joint_command_features(sample, timestamps, dt, cfg, q, q_cmd)
    
    t_rs, filtered = _resample_and_smooth({"q": q, "q_cmd": q_cmd}, timestamps, dt, cfg)
    q_f = filtered["q"]
    q_cmd_f = filtered["q_cmd"]
    dq = _gradient(q_f, dt)
    ddq = _gradient(dq, dt)
    q_err = q_cmd_f - q_f

    blocks = {
        "joint": np.concatenate([q_f, dq, ddq], axis=1),
        "command": np.concatenate([q_cmd_f, q_err], axis=1),
    }
    block_feature_names = {
        "joint": (
            _channel_names("q", q_f.shape[1])
            + _channel_names("dq", dq.shape[1])
            + _channel_names("ddq", ddq.shape[1])
        ),
        "command": (
            _channel_names("q_cmd", q_cmd_f.shape[1])
            + _channel_names("q_err", q_err.shape[1])
        ),
    }

    matrix, feature_names = _finalize_blocks(
        blocks,
        block_feature_names,
        cfg,
        ["joint", "command"],
    )
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "timestamps": t_rs,
        "source_channels": {"joint": ["q"], "command": ["q_cmd", "q_err"]},
        "aux_signals": {
            "original": {"timestamps": timestamps, "q": q, "q_cmd": q_cmd},
            "resampled": {
                "timestamps": t_rs,
                "q": q_f,
                "q_cmd": q_cmd_f,
                "dq": dq,
                "ddq": ddq,
                "q_err": q_err,
            },
        },
    }


def _build_grouped_joint_command_features(
    sample: Mapping[str, Any],
    timestamps: Array,
    dt: float,
    cfg: FeatureBuildConfig,
    q: Array,
    q_cmd: Array,
) -> Dict[str, Any]:
    """Build joint command features grouped by joint names."""
    # For grouped processing, we need to know the joint names
    # This would typically come from the dataset schema or configuration
    joint_names = sample.get("joint_names", [f"joint_{i}" for i in range(q.shape[1])])
    
    # Create mapping from joint name to index
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    
    # Build features for each group
    group_blocks = {}
    group_feature_names = {}
    raw_group_blocks = {}  # Store raw (unnormalized) values
    
    # Initialize t_rs with the original timestamps (will be updated if we process groups)
    t_rs = timestamps
    
    for group_name, group_joint_names in cfg.joint_groups.items():
        # Get indices for this group
        indices = [name_to_idx[name] for name in group_joint_names if name in name_to_idx]
        if not indices:
            continue
            
        # Extract group data
        q_group = q[:, indices]
        q_cmd_group = q_cmd[:, indices]
        
        # Process group data
        t_rs, filtered = _resample_and_smooth({
            f"q_{group_name}": q_group, 
            f"q_cmd_{group_name}": q_cmd_group
        }, timestamps, dt, cfg)
        q_f = filtered[f"q_{group_name}"]
        q_cmd_f = filtered[f"q_cmd_{group_name}"]
        dq = _gradient(q_f, dt)
        ddq = _gradient(dq, dt)
        q_err = q_cmd_f - q_f
        
        # Store group features (normalized)
        group_blocks[group_name] = np.concatenate([q_f, dq, ddq], axis=1)
        group_blocks[f"command_{group_name}"] = np.concatenate([q_cmd_f, q_err], axis=1)
        group_feature_names[group_name] = (
            _channel_names(f"q_{group_name}", q_f.shape[1])
            + _channel_names(f"dq_{group_name}", dq.shape[1])
            + _channel_names(f"ddq_{group_name}", ddq.shape[1])
        )
        group_feature_names[f"command_{group_name}"] = (
            _channel_names(f"q_cmd_{group_name}", q_cmd_f.shape[1])
            + _channel_names(f"q_err_{group_name}", q_err.shape[1])
        )
        
        # Store raw (unnormalized) values for separate plotting
        # For state features (q, dq, ddq)
        raw_group_blocks[f"{group_name}_state"] = np.concatenate([q_group, dq, ddq], axis=1)
        # For command features (q_cmd, q_err)
        raw_group_blocks[f"{group_name}_command"] = np.concatenate([q_cmd_group, q_err], axis=1)
    
    # Handle case when no groups were processed
    if not group_blocks:
        # Return empty feature matrix
        return {
            "matrix": np.empty((len(timestamps), 0)),
            "feature_names": [],
            "timestamps": timestamps,
            "source_channels": {"joint_groups": []},
            "aux_signals": {
                "original": {"timestamps": timestamps, "q": q, "q_cmd": q_cmd},
                "resampled": {"timestamps": timestamps, "q_groups": {}},
            },
            "raw_matrices": {},
        }
    
    # Combine all groups
    order = []
    for group_name in cfg.joint_groups.keys():
        if group_name in group_blocks:
            order.append(group_name)
        if f"command_{group_name}" in group_blocks:
            order.append(f"command_{group_name}")
    
    matrix, feature_names = _finalize_blocks(group_blocks, group_feature_names, cfg, order)
    
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "timestamps": t_rs,
        "source_channels": {"joint_groups": list(cfg.joint_groups.keys())},
        "aux_signals": {
            "original": {"timestamps": timestamps, "q": q, "q_cmd": q_cmd},
            "resampled": {"timestamps": t_rs, "q_groups": group_blocks},
        },
        "raw_matrices": raw_group_blocks,  # Include raw matrices for separate plotting
    }


def _build_cartesian_features(
    sample: Mapping[str, Any],
    timestamps: Array,
    dt: float,
    cfg: FeatureBuildConfig,
) -> Dict[str, Any]:
    pos = _extract_position(sample)
    quat = _extract_quaternion(sample, required=False)

    signals: Dict[str, Array] = {"position": pos}
    source_channels = ["position"]
    original: Dict[str, Array] = {"timestamps": timestamps, "position": pos}

    if quat is not None:
        rotvec = unwrap_orientation(quat_to_rotvec(quat))
        signals["rotvec"] = rotvec
        source_channels.append("quaternion")
        original["quaternion"] = quat

    t_rs, filtered = _resample_and_smooth(signals, timestamps, dt, cfg)
    pos_f = filtered["position"]
    dpos = _gradient(pos_f, dt)
    ddpos = _gradient(dpos, dt)

    cartesian_parts = [pos_f, dpos, ddpos]
    cartesian_feature_names = (
        _channel_names("pos", pos_f.shape[1])
        + _channel_names("dpos", dpos.shape[1])
        + _channel_names("ddpos", ddpos.shape[1])
    )
    resampled: Dict[str, Array] = {
        "timestamps": t_rs,
        "position": pos_f,
        "dpos": dpos,
        "ddpos": ddpos,
    }

    if "rotvec" in filtered:
        rot_f = filtered["rotvec"]
        drot = _gradient(rot_f, dt)
        ddrot = _gradient(drot, dt)
        cartesian_parts.extend([rot_f, drot, ddrot])
        cartesian_feature_names += (
            _channel_names("rotvec", rot_f.shape[1])
            + _channel_names("drot", drot.shape[1])
            + _channel_names("ddrot", ddrot.shape[1])
        )
        resampled["rotvec"] = rot_f
        resampled["drot"] = drot
        resampled["ddrot"] = ddrot

    blocks = {"cartesian": np.concatenate(cartesian_parts, axis=1)}
    block_feature_names = {"cartesian": cartesian_feature_names}

    matrix, feature_names = _finalize_blocks(blocks, block_feature_names, cfg, ["cartesian"])
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "timestamps": t_rs,
        "source_channels": {"cartesian": source_channels},
        "aux_signals": {"original": original, "resampled": resampled},
    }


# -----------------------------
# helpers
# -----------------------------
def _pick(sample: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    raise KeyError(f"missing keys; expected one of: {keys}")


def _to_2d(x: Array) -> Array:
    return x[:, None] if x.ndim == 1 else x


def _infer_modality(sample: Mapping[str, Any]) -> str:
    if any(key in sample for key in ("position", "cartesian", "quaternion")) and not any(
        key in sample for key in ("q", "joint_pos")
    ):
        return "cartesian"
    if "q_cmd" in sample or "joint_cmd" in sample:
        return "joint_command"
    return "joint"


def _resolve_dt(timestamps: Array, dt_override: float | None) -> float:
    if dt_override is not None:
        if dt_override <= 0:
            raise ValueError("dt must be > 0")
        return float(dt_override)

    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("timestamps must be a 1D array with at least 2 elements")
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("timestamps must contain at least one positive interval")
    return float(np.median(diffs))


def _resample_and_smooth(
    signals: Mapping[str, Array],
    timestamps: Array,
    dt: float,
    cfg: FeatureBuildConfig,
) -> tuple[Array, Dict[str, Array]]:
    t_rs, resampled = resample_fixed_dt(timestamps, signals, dt)
    filtered: Dict[str, Array] = {}
    for name, value in resampled.items():
        filtered[name] = _to_2d(
            np.asarray(
                smooth_signal(
                    value,
                    dt,
                    cfg.smoothing,
                    cfg.savgol_window,
                    cfg.savgol_polyorder,
                    cfg.lowpass_cutoff_hz,
                    cfg.lowpass_order,
                ),
                dtype=float,
            )
        )
    return t_rs, filtered


def _gradient(x: Array, dt: float) -> Array:
    return _to_2d(np.gradient(np.asarray(x, dtype=float), dt, axis=0))


def _channel_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(count)]


def _extract_position(sample: Mapping[str, Any]) -> Array:
    if "position" in sample:
        return _to_2d(np.asarray(sample["position"], dtype=float))
    if "cartesian" in sample:
        cartesian = np.asarray(sample["cartesian"], dtype=float)
        if cartesian.ndim == 2 and cartesian.shape[1] >= 3:
            return cartesian[:, :3]
    if "x" in sample and "y" in sample and "z" in sample:
        return np.column_stack([sample["x"], sample["y"], sample["z"]]).astype(float)
    raise KeyError("position signal not found")


def _extract_quaternion(sample: Mapping[str, Any], required: bool = True) -> Array | None:
    if "quaternion" in sample:
        quaternion = np.asarray(sample["quaternion"], dtype=float)
        return _to_2d(quaternion)
    if "cartesian" in sample:
        cartesian = np.asarray(sample["cartesian"], dtype=float)
        if cartesian.ndim == 2 and cartesian.shape[1] >= 7:
            return cartesian[:, 3:7]
    if required:
        raise KeyError("quaternion signal not found")
    return None


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

    input_was_1d = np.asarray(x).ndim == 1
    x_arr = np.asarray(x, dtype=float)
    if input_was_1d:
        x_arr = x_arr[:, None]

    pad = k // 2
    xp = np.pad(x_arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    y = np.stack(
        [np.convolve(xp[:, i], kernel, mode="valid") for i in range(x_arr.shape[1])],
        axis=1,
    )
    return y[:, 0] if input_was_1d else y


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
    weights = {"joint": 1.0, "cartesian": 1.0, "command": 1.0}
    if modality_weights:
        weights.update({k: float(v) for k, v in modality_weights.items()})

    out: Dict[str, Array] = {}
    for name, x in modalities.items():
        out[name] = _robust_scale(np.asarray(x, dtype=float), mode) * weights.get(name, 1.0)
    return out


def _normalize_and_weight_raw(
    modalities: Mapping[str, Array],
    mode: str,
    modality_weights: Optional[Mapping[str, float]],
) -> Dict[str, Array]:
    """Apply normalization and weighting but also return raw (unnormalized) values."""
    weights = {"joint": 1.0, "cartesian": 1.0, "command": 1.0}
    if modality_weights:
        weights.update({k: float(v) for k, v in modality_weights.items()})

    out: Dict[str, Array] = {}
    raw: Dict[str, Array] = {}
    for name, x in modalities.items():
        x_array = np.asarray(x, dtype=float)
        out[name] = _robust_scale(x_array, mode) * weights.get(name, 1.0)
        # Store raw values (unnormalized but with weights applied)
        raw[name] = x_array * weights.get(name, 1.0)
    return out, raw


def _apply_joint_weights(
    matrix: Array,
    feature_names: list[str],
    joint_weights: Optional[Mapping[str, float]],
) -> Array:
    """Apply joint-specific weights to the feature matrix."""
    if not joint_weights:
        return matrix
    
    # Create a copy of the matrix to avoid modifying the original
    weighted_matrix = matrix.copy()
    
    # Apply weights to each feature based on joint name in feature name
    for i, feature_name in enumerate(feature_names):
        # Extract joint name from feature name (e.g., "q_right_arm_3" -> "right_arm_3")
        # Feature names are in format: "q_{joint_name}_{index}" or "dq_{joint_name}_{index}" or "ddq_{joint_name}_{index}"
        if "_" in feature_name:
            parts = feature_name.split("_")
            if len(parts) >= 3:
                # Try to match joint names with different levels of specificity
                # Start with the most specific (full joint name) and work backwards
                for j in range(len(parts) - 1, 1, -1):
                    joint_candidate = "_".join(parts[1:j+1])
                    if joint_candidate in joint_weights:
                        weighted_matrix[:, i] *= joint_weights[joint_candidate]
                        break
                # Also try individual joint names
                for part in parts[1:]:
                    if part in joint_weights:
                        weighted_matrix[:, i] *= joint_weights[part]
                        break
    
    return weighted_matrix


def _finalize_blocks(
    blocks: Mapping[str, Array],
    block_feature_names: Mapping[str, list[str]],
    cfg: FeatureBuildConfig,
    order: list[str],
) -> tuple[Array, list[str]]:
    # Get both normalized and raw values
    normalized, raw = _normalize_and_weight_raw(blocks, cfg.normalize, cfg.modality_weights)
    
    # Use normalized values for the main matrix (for segmentation algorithms)
    matrices = [normalized[name] for name in order if name in normalized]
    feature_names: list[str] = []
    for name in order:
        feature_names.extend(block_feature_names.get(name, []))
    
    # Handle case when no matrices are available
    if not matrices:
        # Return empty array with appropriate shape
        return np.empty((0, 0)), []
    
    # Concatenate matrices
    matrix = np.concatenate(matrices, axis=1)
    
    # Apply joint weights if specified
    if cfg.joint_weights:
        matrix = _apply_joint_weights(matrix, feature_names, cfg.joint_weights)
    
    return matrix, feature_names
