#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit(
            "tensorboard is required to export event files. Install tensorboard or run this on the training machine."
        ) from exc
    return EventAccumulator


def export_run(run_dir: Path, output_dir: Path) -> list[dict]:
    EventAccumulator = load_event_accumulator()
    rows = []
    event_files = sorted((run_dir / "logs").glob("**/events.out.tfevents.*"))
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                rows.append(
                    {
                        "run": run_dir.name,
                        "event_file": str(event_file),
                        "tag": tag,
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": event.value,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_path = output_dir / f"{run_dir.name}_tensorboard_scalars.csv"
    with scalar_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "event_file", "tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_combined(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "event_file", "tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict]) -> None:
    grouped = {}
    for row in rows:
        key = (row["run"], row["tag"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (run, tag), values in sorted(grouped.items()):
        values = sorted(values, key=lambda item: int(item["step"]))
        summary_rows.append(
            {
                "run": run,
                "tag": tag,
                "count": len(values),
                "first_step": values[0]["step"],
                "first_value": values[0]["value"],
                "last_step": values[-1]["step"],
                "last_value": values[-1]["value"],
                "best_value": max(item["value"] for item in values),
                "min_value": min(item["value"] for item in values),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "tag",
                "count",
                "first_step",
                "first_value",
                "last_step",
                "last_value",
                "best_value",
                "min_value",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TensorBoard scalar event logs to CSV.")
    parser.add_argument("runs", nargs="+", type=Path, help="checkpoint run directories containing logs/")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/tensorboard_scalars"))
    args = parser.parse_args()

    all_rows = []
    for run_dir in args.runs:
        all_rows.extend(export_run(run_dir, args.output_dir))

    write_combined(args.output_dir / "combined_tensorboard_scalars.csv", all_rows)
    write_summary(args.output_dir / "tensorboard_scalar_summary.csv", all_rows)
    print(f"exported {len(all_rows)} scalar points to {args.output_dir}")


if __name__ == "__main__":
    main()
