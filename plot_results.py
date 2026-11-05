#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot the FP32/FP64 GPU Cholesky benchmark CSV."
    )
    parser.add_argument("csv_files", nargs="+",
                        help="One or more per-job benchmark CSV files")
    parser.add_argument("--fp32-peak", type=float, required=True,
                        help="Theoretical FP32 device peak in Gflop/s")
    parser.add_argument("--fp64-peak", type=float, required=True,
                        help="Theoretical FP64 device peak in Gflop/s")
    parser.add_argument("--gpu", default="GPU model not specified")
    parser.add_argument("--output", default="performance.png")
    args = parser.parse_args()

    best_rates = defaultdict(dict)
    for csv_file in args.csv_files:
        with open(csv_file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                precision = row["precision"]
                n = int(row["N"])
                rate = float(row["gflops"])
                best_rates[precision][n] = max(
                    rate, best_rates[precision].get(n, float("-inf"))
                )

    expected_sizes = set(range(1000, 7001, 500))
    for precision in ("Float32", "Float64"):
        missing = expected_sizes - best_rates[precision].keys()
        if missing:
            raise SystemExit(
                f"Missing {precision} results for N={sorted(missing)}"
            )

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    colors = {"Float32": "tab:blue", "Float64": "tab:orange"}
    labels = {"Float32": "FP32", "Float64": "FP64"}

    for precision in ("Float32", "Float64"):
        points = sorted(best_rates[precision].items())
        sizes, rates = zip(*points)
        color = colors[precision]
        ax.plot(sizes, rates, marker="o", linewidth=2,
                color=color, label=labels[precision])
        for x, y in points:
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=color)
        sustained = max(rates)
        ax.axhline(sustained, color=color, linestyle="--", alpha=0.8,
                   label=f"{labels[precision]} sustained peak: {sustained:.1f} Gflop/s")

    ax.axhline(args.fp32_peak, color="tab:blue", linestyle=":",
               label=f"FP32 theoretical peak: {args.fp32_peak:.1f} Gflop/s")
    ax.axhline(args.fp64_peak, color="tab:orange", linestyle=":",
               label=f"FP64 theoretical peak: {args.fp64_peak:.1f} Gflop/s")
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("Performance (Gflop/s)")
    ax.set_title(f"Blocked Cholesky performance — {args.gpu}")
    ax.set_xticks(sorted(expected_sizes))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
