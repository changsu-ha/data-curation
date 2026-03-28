from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineConfig:
    dataset_path: str
    num_samples: int
    seed: int = 42


def _segment_text(text: str, rng: random.Random) -> list[dict[str, Any]]:
    """Create simple pseudo-segments for demonstration purposes."""
    tokens = text.split()
    if not tokens:
        return [{"label": "empty", "start": 0, "end": 0}]

    labels = ["intro", "body", "conclusion"]
    segments: list[dict[str, Any]] = []
    cursor = 0
    remaining = len(tokens)

    while remaining > 0:
        max_chunk = max(1, remaining // 2)
        chunk = rng.randint(1, max_chunk)
        end = cursor + chunk
        segments.append(
            {
                "label": labels[len(segments) % len(labels)],
                "start": cursor,
                "end": end,
            }
        )
        cursor = end
        remaining = len(tokens) - cursor

    return segments


def run_pipeline(config: PipelineConfig | dict[str, Any]) -> list[dict[str, Any]]:
    """Run segmentation pipeline and return per-sample segments.

    The dataset is expected to be a text file with one sample per line.
    """
    if isinstance(config, dict):
        cfg = PipelineConfig(**config)
    else:
        cfg = config

    lines = [line.strip() for line in Path(cfg.dataset_path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]

    rng = random.Random(cfg.seed)
    samples = lines[: cfg.num_samples]

    results: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        segments = _segment_text(sample, rng)
        label_length = Counter()
        for segment in segments:
            label_length[segment["label"]] += segment["end"] - segment["start"]
        results.append(
            {
                "sample_id": idx,
                "text": sample,
                "num_segments": len(segments),
                "label_lengths": dict(label_length),
                "segments": segments,
            }
        )

    return results
