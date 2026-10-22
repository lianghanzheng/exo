#!/usr/bin/env python3
"""Build and run the SGEMM comparison, then plot GFLOPS vs matrix size.

Backends:
  * exo       - the 6x8 NEON microkernel from this example + a C tiling driver
  * tvm       - a hand-scheduled TVM TE matmul (blocked + NEON vectorized)
  * blis      - BLIS CBLAS
  * openblas  - OpenBLAS CBLAS

All backends run single-threaded. Run this script inside the exo-lang conda
environment (it needs matplotlib); TVM is invoked through its own env.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SIZES = [64, 128, 256, 512, 768, 1024, 1536, 2048]
BACKENDS = ["exo", "tvm", "blis", "openblas"]
LABELS = {
    "exo": "Exo (6x8 NEON microkernel + driver)",
    "tvm": "TVM (blocked + NEON vectorized)",
    "blis": "BLIS (CBLAS)",
    "openblas": "OpenBLAS (CBLAS)",
}
MARKERS = {"exo": "o", "tvm": "s", "blis": "^", "openblas": "D"}
COLORS = {
    "exo": "#d62728",
    "tvm": "#1f77b4",
    "blis": "#2ca02c",
    "openblas": "#9467bd",
}


def find_tvm_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("TVM_PYTHON"):
        return os.environ["TVM_PYTHON"]
    candidates = [
        Path("/opt/anaconda3/envs/tvm/bin/python"),
        Path.home() / "miniconda3/envs/tvm/bin/python",
        Path.home() / "anaconda3/envs/tvm/bin/python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_tsv(
    command: list[str], env: dict[str, str]
) -> list[tuple[str, int, float, float]]:
    proc = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        backend, size, seconds, gflops = parts
        rows.append((backend, int(size), float(seconds), float(gflops)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=0.2,
        help="approximate time to spend timing each backend/size",
    )
    parser.add_argument("--tvm-python", default=None)
    parser.add_argument("--skip-tvm", action="store_true")
    parser.add_argument("--output", default=str(HERE / "matmul_benchmark.png"))
    parser.add_argument("--csv", default=str(HERE / "matmul_benchmark.csv"))
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="re-plot an existing CSV without re-running the benchmarks",
    )
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    env = os.environ.copy()
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "TVM_NUM_THREADS": "1",
        }
    )

    results: dict[str, dict[int, float]] = {b: {} for b in BACKENDS}
    timings: dict[str, dict[int, float]] = {b: {} for b in BACKENDS}

    if args.plot_only:
        with open(args.csv, newline="") as f:
            for row in csv.DictReader(f):
                backend = row["backend"]
                n = int(row["size"])
                results.setdefault(backend, {})[n] = float(row["gflops"])
                timings.setdefault(backend, {})[n] = float(row["seconds_per_call"])
        sizes = sorted({n for backend in results for n in results[backend]})
    else:
        # Make sure the Exo C code exists, then build all C benchmark binaries.
        subprocess.run(["make", "-C", str(HERE.parent)], check=True)
        subprocess.run(["make", "-C", str(HERE)], check=True)

        size_args = [str(n) for n in sizes]

        def record(rows):
            for backend, size, seconds, gflops in rows:
                results.setdefault(backend, {})[size] = gflops
                timings.setdefault(backend, {})[size] = seconds

        for backend in ("exo", "blis", "openblas"):
            binary = HERE / f"bench_{backend}"
            print(f"[run] {backend}", flush=True)
            record(
                run_tsv(
                    [str(binary), "--target", str(args.target_seconds)] + size_args,
                    env,
                )
            )

        if not args.skip_tvm:
            tvm_python = find_tvm_python(args.tvm_python)
            print(f"[run] tvm ({tvm_python})", flush=True)
            record(
                run_tsv(
                    [
                        tvm_python,
                        str(HERE / "tvm_bench.py"),
                        *size_args,
                        "--target",
                        str(args.target_seconds),
                    ],
                    env,
                )
            )

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["backend", "size", "seconds_per_call", "gflops"])
            for backend in BACKENDS:
                for n in sizes:
                    if n in results.get(backend, {}):
                        writer.writerow(
                            [
                                backend,
                                n,
                                f"{timings[backend][n]:.9g}",
                                f"{results[backend][n]:.6f}",
                            ]
                        )

    print("\nGFLOP/s")
    header = "size".rjust(6) + "".join(f"{b:>12}" for b in BACKENDS)
    print(header)
    for n in sizes:
        row = f"{n:>6}"
        for backend in BACKENDS:
            value = results.get(backend, {}).get(n)
            row += f"{value:>12.2f}" if value is not None else f"{'-':>12}"
        print(row)

    present = [b for b in BACKENDS if results.get(b)]
    if not present:
        print("no benchmark results", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for backend in present:
        xs = sorted(results[backend])
        ys = [results[backend][n] for n in xs]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[backend],
            color=COLORS[backend],
            linewidth=2,
            markersize=6,
            label=LABELS[backend],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Matrix size N  (M = N = K = N, log scale)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Square SGEMM performance (Apple M4, single thread)")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"\nwrote {args.output}")
    if not args.plot_only:
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
