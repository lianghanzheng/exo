#!/usr/bin/env python3
"""Benchmark a hand-scheduled TVM SGEMM on square matrices.

The schedule follows the classic TVM matmul blocking pattern:
  * block the output by BLOCK x BLOCK
  * split K inside each output tile and unroll by K_UNROLL
  * vectorize the innermost N dimension by VEC (4 lanes for NEON f32)
  * loop order: io, jo, ko, ii, jo_inner, ki, ji

Prints one TSV line per size:  tvm  N  seconds_per_call  GFLOPS
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import tvm
from tvm import s_tir, te
from tvm.runtime._tensor import empty, tensor

BLOCK = 32
K_UNROLL = 8
VEC = 4


def build_matmul(n: int):
    A = te.placeholder((n, n), name="A", dtype="float32")
    B = te.placeholder((n, n), name="B", dtype="float32")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    func = te.create_prim_func([A, B, C])
    sched = s_tir.Schedule(func)
    block = sched.get_sblock("C")
    i, j, kk = sched.get_loops(block)

    ko, ki = sched.split(kk, factors=[None, K_UNROLL])
    io, ii = sched.split(i, factors=[None, BLOCK])
    jo, ji = sched.split(j, factors=[None, BLOCK])
    jio, jii = sched.split(ji, factors=[None, VEC])

    sched.reorder(io, jo, ko, ii, jio, ki, jii)
    sched.vectorize(jii)
    sched.unroll(ki)

    target = tvm.target.Target("llvm")
    return tvm.build(sched.mod, target=target)


def benchmark(n: int, target_seconds: float, verify: bool) -> tuple[float, float]:
    mod = build_matmul(n)
    dev = tvm.cpu()

    rng = np.random.default_rng(0)
    a_np = rng.random((n, n), dtype=np.float32)
    b_np = rng.random((n, n), dtype=np.float32)

    a = tensor(a_np, dev)
    b = tensor(b_np, dev)
    c = empty((n, n), "float32", dev)

    mod(a, b, c)
    if verify:
        np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-3, atol=0.1)

    # Estimate the per-call cost, then run long enough for a stable mean.
    est_timer = mod.time_evaluator(mod.entry_name, dev, number=1, repeat=1)
    estimate = float(est_timer(a, b, c).mean)
    if estimate <= 0:
        estimate = 1e-9
    iters = max(1, min(200000, int(target_seconds / estimate)))

    timer = mod.time_evaluator(mod.entry_name, dev, number=iters, repeat=3)
    seconds = float(timer(a, b, c).mean)
    gflops = 2.0 * n * n * n / (seconds * 1e9)
    return seconds, gflops


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="+", type=int)
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TVM_NUM_THREADS", "1")

    # Ramp the CPU before the first measurement.
    start = time.perf_counter()
    x = 1.0
    while time.perf_counter() - start < 0.1:
        x = x * 1.0000001 + 1e-7

    for n in args.sizes:
        seconds, gflops = benchmark(n, args.target, not args.no_verify)
        print(f"tvm\t{n}\t{seconds:.9g}\t{gflops:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
