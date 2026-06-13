#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOSS_LINE = re.compile(r"\(epoch:\s*(\d+),\s*iters:\s*(\d+),\s*time:\s*([0-9.]+)\)\s*(.*)")
LOSS_ITEM = re.compile(r"([A-Za-z][A-Za-z0-9_]*):\s*([-+]?\d+(?:\.\d+)?)")
FULL_EVAL = re.compile(r"\[i\]\s+full evaluation at\s+(.+?)\s+\(epoch\s+(\d+),\s+iter\s+(\d+)\)")
DATASET_METRIC = re.compile(
    r"\[eval\]\s+([\w_]+):\s+PSNR=([-+]?\d+(?:\.\d+)?),\s+"
    r"SSIM=([-+]?\d+(?:\.\d+)?),\s+NCC=([-+]?\d+(?:\.\d+)?),\s+"
    r"LMSE=([-+]?\d+(?:\.\d+)?)\s+\|\s+vs baseline:\s+"
    r"dPSNR=([-+]?\d+(?:\.\d+)?),\s+dSSIM=([-+]?\d+(?:\.\d+)?),\s+"
    r"dNCC=([-+]?\d+(?:\.\d+)?),\s+dLMSE=([-+]?\d+(?:\.\d+)?)"
)
MEAN_DELTA = re.compile(
    r"\[eval\]\s+mean metric deltas:\s+PSNR=([-+]?\d+(?:\.\d+)?),\s+"
    r"SSIM=([-+]?\d+(?:\.\d+)?),\s+NCC=([-+]?\d+(?:\.\d+)?),\s+"
    r"LMSE=([-+]?\d+(?:\.\d+)?)"
)
SELECTION_SCORE = re.compile(r"\[eval\]\s+selection score .*:\s+([-+]?\d+(?:\.\d+)?)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_opt_file(run_dir: Path) -> dict:
    opt_path = run_dir / "opt.txt"
    opts = {}
    if not opt_path.exists():
        return opts
    for line in opt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            opts[key.strip()] = value.strip()
    return opts


def as_float(opts: dict, key: str, default: float) -> float:
    try:
        return float(opts.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_loss_log(run_dir: Path) -> list[dict]:
    path = run_dir / "loss_log.txt"
    if not path.exists():
        return []
    opts = read_opt_file(run_dir)
    lambda_vgg = as_float(opts, "lambda_vgg", 0.0)
    lambda_adapter_consistency = as_float(opts, "lambda_adapter_consistency", 0.0)
    lambda_adapter_sparsity = as_float(opts, "lambda_adapter_sparsity", 0.0)

    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOSS_LINE.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        values = {k: float(v) for k, v in LOSS_ITEM.findall(match.group(4))}
        objective_proxy = values.get("IPixel", 0.0)
        objective_proxy += lambda_vgg * values.get("VGG", 0.0)
        objective_proxy += lambda_adapter_consistency * values.get("ACons", 0.0)
        objective_proxy += lambda_adapter_sparsity * values.get("ASparse", 0.0)
        row = {
            "run": run_dir.name,
            "epoch": epoch,
            "iteration": iteration,
            "x": float(epoch),
            "objective_proxy": objective_proxy,
        }
        row.update(values)
        rows.append(row)
    return rows


def parse_eval_log(run_dir: Path) -> tuple[list[dict], list[dict]]:
    path = run_dir / "tmux_eval_log.txt"
    if not path.exists():
        return [], []

    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"[\r\n]+", text)
    eval_rows = []
    dataset_rows = []
    current = None

    for raw in parts:
        line = raw.strip()
        match = FULL_EVAL.search(line)
        if match:
            current = {
                "run": run_dir.name,
                "tag": match.group(1),
                "epoch": int(match.group(2)),
                "iteration": int(match.group(3)),
            }
            continue

        if current is None:
            continue

        match = DATASET_METRIC.search(line)
        if match:
            dataset, psnr, ssim, ncc, lmse, dpsnr, dssim, dncc, dlmse = match.groups()
            dataset_rows.append({
                **current,
                "dataset": dataset,
                "PSNR": float(psnr),
                "SSIM": float(ssim),
                "NCC": float(ncc),
                "LMSE": float(lmse),
                "dPSNR": float(dpsnr),
                "dSSIM": float(dssim),
                "dNCC": float(dncc),
                "dLMSE": float(dlmse),
            })
            continue

        match = MEAN_DELTA.search(line)
        if match:
            current.update({
                "mean_dPSNR": float(match.group(1)),
                "mean_dSSIM": float(match.group(2)),
                "mean_dNCC": float(match.group(3)),
                "mean_dLMSE": float(match.group(4)),
            })
            continue

        match = SELECTION_SCORE.search(line)
        if match:
            current["selection_score"] = float(match.group(1))
            matching = [
                row for row in dataset_rows
                if row["run"] == current["run"]
                and row["epoch"] == current["epoch"]
                and row["iteration"] == current["iteration"]
            ]
            if matching:
                current["mean_LMSE"] = sum(row["LMSE"] for row in matching) / len(matching)
                current["mean_PSNR"] = sum(row["PSNR"] for row in matching) / len(matching)
                current["mean_SSIM"] = sum(row["SSIM"] for row in matching) / len(matching)
                current["mean_NCC"] = sum(row["NCC"] for row in matching) / len(matching)
            eval_rows.append(current.copy())
            current = None

    return eval_rows, dataset_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def smooth(series: pd.Series, window: int = 5) -> pd.Series:
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_curves(loss_df: pd.DataFrame, eval_df: pd.DataFrame, output_png: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    axes = axes.ravel()

    if not loss_df.empty:
        for run, group in loss_df.groupby("run"):
            group = group.sort_values("iteration")
            x = group["epoch"]
            if "IPixel" in group:
                axes[0].plot(x, smooth(group["IPixel"]), label=run, linewidth=1.8)
            axes[1].plot(x, smooth(group["objective_proxy"]), label=run, linewidth=1.8)
        axes[0].set_title("Training Pixel/Structure Loss (IPixel)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].set_title("Weighted Training Objective Proxy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("IPixel + weighted auxiliary losses")
    else:
        axes[0].axis("off")
        axes[0].text(
            0.5, 0.55,
            "No per-iteration training loss records were found.\n"
            "The synced loss_log.txt files contain headers only,\n"
            "and tmux logs do not include IPixel/VGG progress lines.",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[1].axis("off")
        axes[1].text(
            0.5, 0.55,
            "Fallback plots use full benchmark evaluation trends.\n"
            "LMSE is shown as a loss-like metric; lower is better.",
            ha="center",
            va="center",
            fontsize=10,
        )

    if not eval_df.empty:
        for run, group in eval_df.groupby("run"):
            group = group.sort_values("epoch")
            axes[2].plot(group["epoch"], group["mean_LMSE"], marker="o", label=run, linewidth=1.7)
            axes[3].plot(group["epoch"], group["selection_score"], marker="o", label=run, linewidth=1.7)
        axes[2].set_title("Benchmark Mean LMSE During Training (Lower Is Better)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Mean LMSE")
        axes[3].set_title("Selection Score During Training (Higher Is Better)")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Mean relative PSNR/SSIM score")

    for ax in axes:
        if ax.has_data():
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.35)

    fig.suptitle("Training Trend Summary", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss/evaluation trends from checkpoint logs.")
    parser.add_argument("--checkpoints-dir", type=Path, default=repo_root() / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "analysis_outputs")
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.checkpoints_dir.iterdir() if path.is_dir())
    loss_rows = []
    eval_rows = []
    dataset_rows = []

    for run_dir in run_dirs:
        loss_rows.extend(parse_loss_log(run_dir))
        run_eval_rows, run_dataset_rows = parse_eval_log(run_dir)
        eval_rows.extend(run_eval_rows)
        dataset_rows.extend(run_dataset_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_loss_points.csv", loss_rows)
    write_csv(args.output_dir / "eval_metric_trends.csv", eval_rows)
    write_csv(args.output_dir / "eval_dataset_metrics.csv", dataset_rows)

    loss_df = pd.DataFrame(loss_rows)
    eval_df = pd.DataFrame(eval_rows)
    plot_curves(loss_df, eval_df, args.output_dir / "training_loss_trends.png")

    print("runs:", ", ".join(path.name for path in run_dirs))
    print("training_loss_points:", len(loss_rows))
    print("eval_points:", len(eval_rows))
    print("saved:", args.output_dir / "training_loss_trends.png")
    print("saved:", args.output_dir / "training_loss_points.csv")
    print("saved:", args.output_dir / "eval_metric_trends.csv")
    print("saved:", args.output_dir / "eval_dataset_metrics.csv")


if __name__ == "__main__":
    main()
