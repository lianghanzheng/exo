# NEON Matmul Example

This example is the AArch64 counterpart of [`examples/avx2_matmul`](../avx2_matmul).
It schedules the same rank-k microkernel for NEON instead of AVX2, then verifies
the result and benchmarks the generated code.

## Files

- `arm_matmul.py` — defines the unscheduled `rank_k_reduce_6x8` kernel and the
  scheduled `rank_k_reduce_6x8_scheduled` kernel.
- `main.c` — harness: initializes the inputs, checks both kernels against a
  double-precision reference, then times them.
- `Makefile` — runs `exocc` and compiles the generated C.
- `benchmarks/` — square SGEMM comparison against TVM, BLIS, and OpenBLAS,
  including the plotting script and the generated figure.

## Differences from the AVX2 example

NEON registers are 128 bits wide, so an `f32` vector holds 4 lanes instead of
the 8 lanes of AVX2. To keep the same register pressure as the AVX2 example
(12 output registers for the `C` tile), the output tile is `6x8` instead of
`6x16`:

| | AVX2 example | NEON example |
| --- | --- | --- |
| vector width | 8 | 4 |
| output tile | 6x16 | 6x8 |
| output registers | 12 | 12 |
| load / store | `mm256_loadu_ps` / `mm256_storeu_ps` | `neon_vld_4xf32` / `neon_vst_4xf32` |
| broadcast | `mm256_broadcast_ss` | `neon_broadcast_4xf32` |
| multiply-add | `mm256_fmadd_ps` | `neon_vfmadd_4xf32_4xf32` |

The scheduling steps themselves mirror `x86_matmul.py` one-for-one:
reorder loops, stage `C`, split dimensions for vectorization, map loads/stores
to NEON intrinsics, stage `B`, broadcast `A`, and map the FMA.

## Build and run

Make sure the `exo-lang` environment is active (or use `conda run`):

```sh
conda activate exo-lang
cd examples/neon_matmul
make
./neon_matmul          # K = 2048, 1000 iterations
./neon_matmul 4096     # custom K
./neon_matmul 4096 50  # custom K and iteration count
```

`exocc` must be on `PATH`; it is used to generate `neon_matmul.c` and
`neon_matmul.h` from `arm_matmul.py`.

## What the harness checks

1. Computes `C = A * B` with a double-precision reference implementation.
2. Runs both Exo kernels on the same inputs and compares each result against
   the reference (absolute tolerance `1e-4`, relative tolerance `1e-4`).
3. Warms up and times each kernel, reporting microseconds per call, GFLOP/s,
   and the speedup of the scheduled kernel over the unscheduled one.

Example output on an Apple Silicon machine (`K = 2048`, 1000 iterations):

```
Verification:
  rank_k_reduce_6x8            PASS (max abs err 2.480e-05, max rel err 1.490e-05)
  rank_k_reduce_6x8_scheduled  PASS (max abs err 2.670e-05, max rel err 2.882e-05)

Performance:
  rank_k_reduce_6x8                77.08 us/call       2.55 GFLOP/s
  rank_k_reduce_6x8_scheduled       2.23 us/call      88.28 GFLOP/s
  speedup                          34.61x
```

Timings depend on the machine and CPU frequency governor; the verification
result should always be `PASS`.

## Comparison with TVM, BLIS, and OpenBLAS

The `benchmarks/` directory compares square SGEMM (`C = A * B`, with
`M = N = K`) across four backends, all running **single-threaded**:

| backend | what is measured |
| --- | --- |
| Exo | the `6x8` NEON microkernel above, used as the compute core of a C tiling driver |
| TVM | a TE matmul with an S-TIR schedule (32x32 output tiling, K unrolled by 8, 4-lane NEON vectorization) |
| BLIS | `cblas_sgemm` from a local BLIS build |
| OpenBLAS | `cblas_sgemm` from a local OpenBLAS build |

The Exo driver packs `B` once into `Kx8` panels and processes `M` in 6-row
blocks; partial tiles are zero-padded so the microkernel handles them. Packing
and tiling time is included in the measurement, just as BLIS/OpenBLAS include
their own packing costs.

The TVM schedule is a single hand-chosen configuration, not an autotuned
schedule. Its dip at 1024/2048 is a power-of-two cache-aliasing effect of the
compact row-major layout.

### Prerequisites and running

- BLIS is expected at `~/GitRepo/blis` and OpenBLAS at `~/GitRepo/openblas`
  (override with `BLIS_DIR=` / `OPENBLAS_DIR=` when building the benchmarks).
- TVM is expected in the `tvm` conda environment; override the interpreter with
  `--tvm-python` or the `TVM_PYTHON` environment variable.
- `matplotlib` is needed in the `exo-lang` environment
  (`pip install matplotlib`).

```sh
conda activate exo-lang
cd examples/neon_matmul/benchmarks
python run_benchmarks.py                 # builds and runs everything
python run_benchmarks.py --skip-tvm      # C backends only
python run_benchmarks.py --sizes 128 512 2048
```

The script writes `matmul_benchmark.csv` and `matmul_benchmark.png`
(X axis: square matrix size; Y axis: GFLOP/s). Example results on an Apple M4:

```
  size    exo      tvm     blis  openblas      (GFLOP/s)
    64  74.15    95.41    77.34     32.48
   128  77.77    92.43   103.98     79.21
   256  84.55    91.26   112.97     89.80
   512  87.17    89.88   120.43     91.64
   768  88.18    89.59   122.38    112.91
  1024  87.82    74.10   122.06    113.15
  1536  88.87    85.81   123.56    112.24
  2048  88.42    52.78   122.94    112.07
```

Measurements are single-run and depend on the CPU's thermal/frequency state,
so expect a few percent run-to-run variation (more at small sizes).
