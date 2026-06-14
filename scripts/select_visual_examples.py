#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EVAL_DATASETS = ["ceilnet_table2", "real20", "objects", "postcard", "wild"]
DISPLAY_NAMES = {
    "ceilnet_table2": "CEILNet Table2",
    "real20": "Real20",
    "objects": "SIR2 Objects",
    "postcard": "SIR2 Postcard",
    "wild": "SIR2 Wild",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(repo_root()))

from util.index import quality_assess


def run_inference(args, model_name: str, checkpoint: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        save_subdir = f"{model_name}/{dataset}"
        cmd = [
            sys.executable,
            "test_errnet.py",
            "--dataset", dataset,
            "--result_dir", str(output_root),
            "--save_subdir", save_subdir,
            "--name", model_name,
            "--hyper",
            "-r",
            "--icnn_path", str(checkpoint),
            "--gpu_ids", args.gpu_ids,
            "--nThreads", str(args.nthreads),
            "--display_id", "0",
            "--no-log",
        ]
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo_root(), check=True)


def as_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def crop_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred, target = crop_pair(pred, target)
    return quality_assess(pred.astype(np.uint8), target.astype(np.uint8))


def candidate_score(row: dict) -> float:
    score = 0.0
    score += row["dPSNR"] / 3.0
    score += row["dSSIM"] / 0.04
    score += row["dNCC"] / 0.04
    score += min(max(-row["dLMSE"], 0.0), 0.02) / 0.004
    penalties = 0.0
    for key in ("dPSNR", "dSSIM", "dNCC"):
        if row[key] <= 0:
            penalties += abs(row[key]) * 10.0
    if row["dLMSE"] >= 0:
        penalties += row["dLMSE"] * 1000.0
    return score - penalties


def collect_candidates(output_root: Path, baseline_name: str, ours_name: str, datasets: list[str]) -> list[dict]:
    rows = []
    for dataset in datasets:
        baseline_dir = output_root / baseline_name / dataset
        ours_dir = output_root / ours_name / dataset
        if not baseline_dir.exists() or not ours_dir.exists():
            continue

        for sample_dir in sorted(ours_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample = sample_dir.name
            b_sample_dir = baseline_dir / sample
            paths = {
                "input": sample_dir / "m_input.png",
                "target": sample_dir / "t_label.png",
                "ours": sample_dir / f"{ours_name}.png",
                "baseline": b_sample_dir / f"{baseline_name}.png",
            }
            if not all(path.exists() for path in paths.values()):
                continue

            target = as_rgb(paths["target"])
            baseline = as_rgb(paths["baseline"])
            ours = as_rgb(paths["ours"])
            baseline_metrics = compute_metrics(baseline, target)
            ours_metrics = compute_metrics(ours, target)
            row = {
                "dataset": dataset,
                "sample": sample,
                **{f"baseline_{k}": float(v) for k, v in baseline_metrics.items()},
                **{f"ours_{k}": float(v) for k, v in ours_metrics.items()},
                "input_path": str(paths["input"]),
                "target_path": str(paths["target"]),
                "baseline_path": str(paths["baseline"]),
                "ours_path": str(paths["ours"]),
            }
            row["dPSNR"] = row["ours_PSNR"] - row["baseline_PSNR"]
            row["dSSIM"] = row["ours_SSIM"] - row["baseline_SSIM"]
            row["dNCC"] = row["ours_NCC"] - row["baseline_NCC"]
            row["dLMSE"] = row["ours_LMSE"] - row["baseline_LMSE"]
            row["score"] = candidate_score(row)
            rows.append(row)
    return rows


def select_diverse(candidates: list[dict], count: int, max_per_dataset: int) -> list[dict]:
    strong = [
        row for row in candidates
        if row["dPSNR"] > 0.15 and row["dSSIM"] > 0 and row["dNCC"] > 0 and row["dLMSE"] < 0
    ]
    pool = sorted(strong or candidates, key=lambda row: row["score"], reverse=True)
    selected = []
    dataset_counts = {}

    for row in pool:
        if dataset_counts.get(row["dataset"], 0) >= max_per_dataset:
            continue
        selected.append(row)
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
        if len(selected) == count:
            return selected

    for row in sorted(candidates, key=lambda row: row["score"], reverse=True):
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == count:
            break
    return selected


def resize_keep_aspect(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(round(w * target_h / float(h))))
    return img.resize((new_w, target_h), Image.BICUBIC)


def center_crop_width(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    if w == target_w:
        return img
    if w < target_w:
        canvas = Image.new("RGB", (target_w, h), "white")
        canvas.paste(img, ((target_w - w) // 2, 0))
        return canvas
    left = (w - target_w) // 2
    return img.crop((left, 0, left + target_w, h))


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_comparison(row: dict, out_path: Path, tile_h: int = 260) -> None:
    images = [
        ("Input", Image.open(row["input_path"]).convert("RGB")),
        ("Baseline", Image.open(row["baseline_path"]).convert("RGB")),
        ("Ours", Image.open(row["ours_path"]).convert("RGB")),
        ("Ground Truth", Image.open(row["target_path"]).convert("RGB")),
    ]
    resized = [(label, resize_keep_aspect(img, tile_h)) for label, img in images]
    tile_w = max(img.size[0] for _, img in resized)
    resized = [(label, center_crop_width(img, tile_w)) for label, img in resized]

    title_font = load_font(22)
    label_font = load_font(18)
    small_font = load_font(15)
    margin = 18
    gap = 12
    label_h = 34
    footer_h = 88
    width = margin * 2 + tile_w * 4 + gap * 3
    height = margin * 2 + label_h + tile_h + footer_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"{DISPLAY_NAMES.get(row['dataset'], row['dataset'])} / {row['sample']}"
    draw.text((margin, margin - 2), title, fill=(20, 20, 20), font=title_font)

    y0 = margin + label_h
    for i, (label, img) in enumerate(resized):
        x = margin + i * (tile_w + gap)
        draw.text((x, margin + 24), label, fill=(40, 40, 40), font=label_font)
        canvas.paste(img, (x, y0))

    metrics_1 = (
        f"PSNR: {row['baseline_PSNR']:.2f} -> {row['ours_PSNR']:.2f}  "
        f"({row['dPSNR']:+.2f})    "
        f"SSIM: {row['baseline_SSIM']:.4f} -> {row['ours_SSIM']:.4f}  "
        f"({row['dSSIM']:+.4f})"
    )
    metrics_2 = (
        f"NCC: {row['baseline_NCC']:.4f} -> {row['ours_NCC']:.4f}  "
        f"({row['dNCC']:+.4f})    "
        f"LMSE: {row['baseline_LMSE']:.3g} -> {row['ours_LMSE']:.3g}  "
        f"({row['dLMSE']:+.3g}, lower is better)"
    )
    draw.text((margin, y0 + tile_h + 16), metrics_1, fill=(20, 20, 20), font=small_font)
    draw.text((margin, y0 + tile_h + 42), metrics_2, fill=(20, 20, 20), font=small_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and visualize test images where our model beats baseline.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("checkpoints/errnet/errnet_060_00463920.pt"))
    parser.add_argument("--ours-checkpoint", type=Path, default=Path("checkpoints/errnet_baseline_openrr_identity_ft/errnet_best_eval.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("analysis_outputs/visual_example_search"))
    parser.add_argument("--baseline-name", default="baseline_model")
    parser.add_argument("--ours-name", default="best_model")
    parser.add_argument("--datasets", nargs="+", default=EVAL_DATASETS)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--max-per-dataset", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    if not args.skip_inference:
        run_inference(args, args.baseline_name, args.baseline_checkpoint, args.output_root / "raw_results")
        run_inference(args, args.ours_name, args.ours_checkpoint, args.output_root / "raw_results")

    candidates = collect_candidates(
        args.output_root / "raw_results",
        args.baseline_name,
        args.ours_name,
        args.datasets,
    )
    if not candidates:
        raise RuntimeError("No candidates found. Check inference outputs and dataset paths.")

    candidates = sorted(candidates, key=lambda row: row["score"], reverse=True)
    selected = select_diverse(candidates, args.num_examples, args.max_per_dataset)

    write_csv(args.output_root / "all_candidates.csv", candidates)
    write_csv(args.output_root / "selected_examples.csv", selected)
    for idx, row in enumerate(selected, start=1):
        out = args.output_root / "comparisons" / f"{idx:02d}_{row['dataset']}_{row['sample']}.png"
        safe_out = Path(re.sub(r"[^\w./\\\\:-]+", "_", str(out)))
        make_comparison(row, safe_out)
        print(
            f"[selected {idx}] {row['dataset']}/{row['sample']} "
            f"dPSNR={row['dPSNR']:+.3f} dSSIM={row['dSSIM']:+.4f} "
            f"dNCC={row['dNCC']:+.4f} dLMSE={row['dLMSE']:+.4f} -> {safe_out}"
        )

    print("saved candidates:", args.output_root / "all_candidates.csv")
    print("saved selected:", args.output_root / "selected_examples.csv")
    print("saved comparisons:", args.output_root / "comparisons")


if __name__ == "__main__":
    main()
