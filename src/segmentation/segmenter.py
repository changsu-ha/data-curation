"""Two-stage trajectory segmentation utilities.

Stage 1: Temporal boundary detection
- Method A: ruptures-based PELT/Binseg change point detection
- Method B: rule-based events (speed jumps, stop zones, gripper transitions)

Stage 2: semantic labelling with confidence scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .config import TaskProfile, get_task_profile

try:
    import ruptures as rpt
except Exception:  # pragma: no cover - optional dependency
    rpt = None


@dataclass(frozen=True)
class Segment:
    start_t: float
    end_t: float


def _to_numpy(signal: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1D signal.")
    return arr


def _distance_xy(points_xyz: np.ndarray, center_xyz: Sequence[float]) -> np.ndarray:
    cx, cy, _ = center_xyz
    return np.sqrt((points_xyz[:, 0] - cx) ** 2 + (points_xyz[:, 1] - cy) ** 2)


def _duration_s(start_idx: int, end_idx: int, t: np.ndarray) -> float:
    return float(max(0.0, t[end_idx] - t[start_idx]))


def _apply_hysteresis(mask: np.ndarray, t: np.ndarray, hold_s: float) -> np.ndarray:
    if hold_s <= 0:
        return mask.copy()
    out = mask.copy()
    hold_until = -np.inf
    for i, ti in enumerate(t):
        if mask[i]:
            hold_until = ti + hold_s
        elif ti <= hold_until:
            out[i] = True
    return out


def _merge_boundaries(
    boundaries: Iterable[int], t: np.ndarray, min_gap_s: float
) -> List[int]:
    keep: List[int] = []
    for idx in sorted(set(int(v) for v in boundaries if 0 < v < len(t) - 1)):
        if not keep or (t[idx] - t[keep[-1]]) >= min_gap_s:
            keep.append(idx)
    return keep


def detect_boundaries(
    t: Sequence[float],
    ee_pos_xyz: np.ndarray,
    ee_speed: Sequence[float],
    gripper_state: Sequence[float],
    profile: TaskProfile | None = None,
    method: str = "hybrid",
    ruptures_method: str = "pelt",
    penalty: float = 5.0,
) -> List[int]:
    """Stage-1 change-point detection using method A/B or hybrid."""

    profile = profile or get_task_profile()
    t_arr = _to_numpy(t)
    speed = _to_numpy(ee_speed)
    gripper = _to_numpy(gripper_state)
    xyz = np.asarray(ee_pos_xyz, dtype=float)
    if xyz.shape != (len(t_arr), 3):
        raise ValueError("ee_pos_xyz must have shape [T, 3].")

    boundaries: List[int] = []

    use_ruptures = method in {"a", "ruptures", "hybrid"}
    if use_ruptures and rpt is not None:
        features = np.column_stack([xyz, speed, gripper])
        min_size = max(2, int(np.ceil(profile.min_segment_duration_s / np.mean(np.diff(t_arr)))))
        algo = (
            rpt.Pelt(model="rbf", min_size=min_size).fit(features)
            if ruptures_method.lower() == "pelt"
            else rpt.Binseg(model="rbf", min_size=min_size).fit(features)
        )
        cps = algo.predict(pen=penalty)
        boundaries.extend([cp for cp in cps[:-1]])

    use_rule = method in {"b", "rules", "hybrid"} or (use_ruptures and rpt is None)
    if use_rule:
        speed_jump = np.where(np.abs(np.diff(speed, prepend=speed[0])) >= profile.speed_change_threshold)[0]
        stationary_mask = _apply_hysteresis(speed <= profile.stop_speed_threshold, t_arr, profile.hysteresis_duration_s)
        stationary_edges = np.where(np.diff(stationary_mask.astype(int), prepend=0) != 0)[0]
        g_bin = gripper >= 0.5
        g_edges = np.where(np.diff(g_bin.astype(int), prepend=g_bin[0]) != 0)[0]
        boundaries.extend(speed_jump.tolist())
        boundaries.extend(stationary_edges.tolist())
        boundaries.extend(g_edges.tolist())

    merged = _merge_boundaries(boundaries, t_arr, min_gap_s=profile.hysteresis_duration_s)
    return merged


def boundaries_to_segments(
    t: Sequence[float], boundaries: Sequence[int], min_duration_s: float
) -> List[Segment]:
    t_arr = _to_numpy(t)
    cuts = [0, *sorted(set(int(i) for i in boundaries if 0 < i < len(t_arr) - 1)), len(t_arr) - 1]
    segments: List[Segment] = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        if _duration_s(s, e, t_arr) >= min_duration_s:
            segments.append(Segment(start_t=float(t_arr[s]), end_t=float(t_arr[e])))
    return segments


def _indices_for_segment(t_arr: np.ndarray, seg: Segment) -> np.ndarray:
    return np.where((t_arr >= seg.start_t) & (t_arr <= seg.end_t))[0]


def _score_bool(cond: bool, weight: float, evidence: List[str], text: str) -> float:
    if cond:
        evidence.append(text)
        return weight
    return 0.0


def label_segments(
    segments: Sequence[Segment | Mapping[str, float]],
    aux_signals: Mapping[str, Any],
    profile: TaskProfile | None = None,
) -> List[Dict[str, Any]]:
    """Stage-2 semantic labeling with confidence and evidence output."""

    profile = profile or get_task_profile()
    t_arr = _to_numpy(aux_signals["t"])
    xyz = np.asarray(aux_signals["ee_pos_xyz"], dtype=float)
    speed = _to_numpy(aux_signals["ee_speed"])
    gripper = _to_numpy(aux_signals["gripper_state"])
    alignment = _to_numpy(aux_signals.get("pose_alignment", np.zeros_like(speed)))

    pick_dist = _distance_xy(xyz, profile.pick_zone_center)
    insertion_dist = _distance_xy(xyz, profile.insertion_zone_center)
    ready_dist = _distance_xy(xyz, profile.ready_pose)
    z = xyz[:, 2]

    outputs: List[Dict[str, Any]] = []
    released_seen = False

    for raw_seg in segments:
        seg = raw_seg if isinstance(raw_seg, Segment) else Segment(**raw_seg)
        idx = _indices_for_segment(t_arr, seg)
        if len(idx) == 0:
            continue
        evidence: List[str] = []
        scores: Dict[str, float] = {
            "approach": 0.0,
            "grasp": 0.0,
            "move": 0.0,
            "insertion": 0.0,
            "place": 0.0,
            "move_to_ready": 0.0,
        }

        g_open = np.mean(gripper[idx] < 0.5) > 0.6
        g_closed = np.mean(gripper[idx] >= 0.5) > 0.6
        g_transitions = int(np.sum(np.abs(np.diff((gripper[idx] >= 0.5).astype(int))) > 0))
        near_pick = np.mean(pick_dist[idx] <= profile.pick_zone_radius_xy) > 0.5
        near_pick_z = np.mean(np.abs(z[idx] - profile.pick_zone_center[2]) <= profile.pick_zone_z_tol) > 0.5
        near_insert = np.mean(insertion_dist[idx] <= profile.insertion_zone_radius_xy) > 0.5
        near_insert_z = np.mean(np.abs(z[idx] - profile.insertion_zone_center[2]) <= profile.insertion_z_tol) > 0.5
        near_ready = np.mean(ready_dist[idx] <= profile.ready_pose_tol) > 0.5
        moving = float(np.mean(speed[idx])) > profile.move_speed_threshold
        precision = float(np.mean(speed[idx])) <= profile.precision_speed_threshold

        scores["approach"] += _score_bool(near_pick, 0.35, evidence, "near pick-zone (xy)")
        scores["approach"] += _score_bool(near_pick_z, 0.20, evidence, "pick-zone z alignment")
        scores["approach"] += _score_bool(g_open, 0.35, evidence, "gripper mostly open")

        scores["grasp"] += _score_bool(g_transitions > 0, 0.45, evidence, "gripper transition detected")
        scores["grasp"] += _score_bool(g_closed, 0.30, evidence, "gripper mostly closed")
        scores["grasp"] += _score_bool(near_pick, 0.20, evidence, "grasp near pick-zone")

        scores["move"] += _score_bool(g_closed, 0.35, evidence, "closed gripper while moving")
        scores["move"] += _score_bool(moving, 0.40, evidence, "ee speed above move threshold")
        scores["move"] += _score_bool(not near_insert, 0.15, evidence, "not yet in insertion zone")

        scores["insertion"] += _score_bool(near_insert, 0.35, evidence, "near insertion zone (xy)")
        scores["insertion"] += _score_bool(near_insert_z, 0.20, evidence, "insertion z alignment")
        scores["insertion"] += _score_bool(
            float(np.mean(alignment[idx])) >= (1.0 - profile.alignment_tolerance),
            0.25,
            evidence,
            "pose alignment high",
        )
        scores["insertion"] += _score_bool(precision, 0.15, evidence, "low-speed precision motion")

        release_now = g_transitions > 0 and g_open
        scores["place"] += _score_bool(release_now, 0.45, evidence, "gripper release event")
        scores["place"] += _score_bool(near_insert, 0.25, evidence, "place at goal zone")
        scores["place"] += _score_bool(not moving, 0.20, evidence, "settled after release")

        scores["move_to_ready"] += _score_bool(released_seen or release_now, 0.30, evidence, "after release phase")
        scores["move_to_ready"] += _score_bool(near_ready, 0.40, evidence, "near ready pose")
        scores["move_to_ready"] += _score_bool(moving or near_ready, 0.20, evidence, "return-to-ready trajectory")

        if release_now:
            released_seen = True

        label, raw_score = max(scores.items(), key=lambda kv: kv[1])
        consistency_bonus = min(0.15, g_transitions * 0.05)
        confidence = float(np.clip(raw_score + consistency_bonus, 0.0, 1.0))

        outputs.append(
            {
                "start_t": seg.start_t,
                "end_t": seg.end_t,
                "label": label,
                "confidence": confidence,
                "evidence": sorted(set(evidence)),
            }
        )

    return outputs


def segment_trajectory(
    aux_signals: Mapping[str, Any],
    profile_name: str = "charger_insertion",
    method: str = "hybrid",
    ruptures_method: str = "pelt",
    penalty: float = 5.0,
) -> List[Dict[str, Any]]:
    """End-to-end trajectory segmentation and semantic labeling pipeline."""

    profile = get_task_profile(profile_name)
    t_arr = _to_numpy(aux_signals["t"])
    boundaries = detect_boundaries(
        t=t_arr,
        ee_pos_xyz=np.asarray(aux_signals["ee_pos_xyz"], dtype=float),
        ee_speed=_to_numpy(aux_signals["ee_speed"]),
        gripper_state=_to_numpy(aux_signals["gripper_state"]),
        profile=profile,
        method=method,
        ruptures_method=ruptures_method,
        penalty=penalty,
    )
    segments = boundaries_to_segments(t_arr, boundaries, min_duration_s=profile.min_segment_duration_s)
    return label_segments(segments, aux_signals=aux_signals, profile=profile)
