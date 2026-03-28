from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run segmentation pipeline")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output path for JSON/CSV results")
    return parser


def _save_results(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if output_path.suffix.lower() == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["sample_id", "num_segments", "label_lengths"])
            writer.writeheader()
            for row in results:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "num_segments": row["num_segments"],
                        "label_lengths": json.dumps(row["label_lengths"], ensure_ascii=False),
                    }
                )
        return

    raise ValueError("output must have .json or .csv extension")


def _print_summary(results: list[dict[str, Any]]) -> None:
    for row in results:
        label_summary = ", ".join(f"{label}:{length}" for label, length in row["label_lengths"].items())
        print(f"sample={row['sample_id']} segments={row['num_segments']} label_lengths=[{label_summary}]")


def main() -> int:
    args = build_parser().parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: dataset path not found: {dataset_path}", file=sys.stderr)
        return 1

    if args.num_samples < 1:
        print("Error: --num-samples must be >= 1", file=sys.stderr)
        return 1

    config = PipelineConfig(
        dataset_path=str(dataset_path),
        num_samples=args.num_samples,
        seed=args.seed,
    )
    results = run_pipeline(config)

    try:
        _save_results(Path(args.output), results)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
