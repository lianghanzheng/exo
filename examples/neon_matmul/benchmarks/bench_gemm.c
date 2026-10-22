/*
 * Square SGEMM benchmark for one backend at a time.
 *
 * Compile with exactly one of:
 *   -DUSE_EXO        Exo 6x8 NEON microkernel + tiling/packing driver
 *   -DUSE_BLIS       BLIS CBLAS
 *   -DUSE_OPENBLAS   OpenBLAS CBLAS
 *
 * Usage:
 *   bench_gemm [--target SECONDS] N1 N2 ...
 *
 * Prints one TSV line per size:
 *   backend  N  seconds_per_call  GFLOPS
 */

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(USE_EXO)
#include "neon_matmul.h"
#define BACKEND "exo"
#elif defined(USE_BLIS)
#include "blis.h"
#include "cblas.h"
#define BACKEND "blis"
#elif defined(USE_OPENBLAS)
#include "cblas.h"
#define BACKEND "openblas"
extern void openblas_set_num_threads(int);
#else
#error "define one of USE_EXO, USE_BLIS, USE_OPENBLAS"
#endif

#define WARMUP_ROUNDS 3

static uint32_t rng_state = 0x9e3779b9u;

static float next_float(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return ((rng_state >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f;
}

static void *xmalloc(size_t bytes) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, bytes) != 0 || ptr == NULL) {
    fprintf(stderr, "failed to allocate %zu bytes\n", bytes);
    exit(1);
  }
  return ptr;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Keep the core busy briefly so the first measurement is not taken while the
 * CPU is still ramping up to its boost frequency. */
static void cpu_warmup(void) {
  volatile float x = 1.0f;
  double start = now_seconds();
  while (now_seconds() - start < 0.1) {
    for (int i = 0; i < 100000; i++)
      x = x * 1.0000001f + 0.0000001f;
  }
}

#if defined(USE_EXO)
/*
 * Use the 6x8 Exo microkernel as the compute core of a square GEMM.
 * B is packed once into Kx8 column panels; M is processed in 6-row blocks.
 * Partial tiles are zero-padded so the microkernel can handle them too.
 */
static float *g_bpacked = NULL;
static float *g_apadded = NULL;

static void exo_sgemm(int M, int N, int K, const float *restrict A,
                      const float *restrict B, float *restrict C) {
  int npanels = (N + 7) / 8;

  for (int p = 0; p < npanels; p++) {
    int col = p * 8;
    int nb = N - col < 8 ? N - col : 8;
    float *dst = g_bpacked + (size_t)p * K * 8;
    for (int k = 0; k < K; k++) {
      const float *src = B + (size_t)k * N + col;
      for (int j = 0; j < 8; j++)
        dst[(size_t)k * 8 + j] = j < nb ? src[j] : 0.0f;
    }
  }

  for (int row = 0; row < M; row += 6) {
    int mb = M - row < 6 ? M - row : 6;
    const float *Ab;

    if (mb == 6) {
      Ab = A + (size_t)row * K;
    } else {
      for (int i = 0; i < 6; i++) {
        for (int k = 0; k < K; k++)
          g_apadded[(size_t)i * K + k] =
              i < mb ? A[(size_t)(row + i) * K + k] : 0.0f;
      }
      Ab = g_apadded;
    }

    for (int p = 0; p < npanels; p++) {
      int col = p * 8;
      int nb = N - col < 8 ? N - col : 8;
      float Ctmp[48];

      for (int t = 0; t < 48; t++)
        Ctmp[t] = 0.0f;

      rank_k_reduce_6x8_scheduled(NULL, (int_fast32_t)K, Ab,
                                  g_bpacked + (size_t)p * K * 8, Ctmp);

      for (int i = 0; i < mb; i++) {
        for (int j = 0; j < nb; j++)
          C[(size_t)(row + i) * N + col + j] = Ctmp[i * 8 + j];
      }
    }
  }
}
#endif

static void run_gemm(int n, const float *A, const float *B, float *C) {
#if defined(USE_EXO)
  exo_sgemm(n, n, n, A, B, C);
#else
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n,
              B, n, 0.0f, C, n);
#endif
}

static void reference_gemm(int n, const float *A, const float *B, float *C) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double acc = 0.0;
      for (int k = 0; k < n; k++)
        acc += (double)A[(size_t)i * n + k] * (double)B[(size_t)k * n + j];
      C[(size_t)i * n + j] = (float)acc;
    }
  }
}

static int verify(int n, const float *A, const float *B, const float *C) {
  float *ref = xmalloc((size_t)n * n * sizeof(float));
  reference_gemm(n, A, B, ref);

  int failures = 0;
  double max_abs = 0.0;
  for (size_t idx = 0; idx < (size_t)n * n; idx++) {
    double diff = fabs((double)C[idx] - (double)ref[idx]);
    if (diff > max_abs)
      max_abs = diff;
    if (diff > 1e-3 + 1e-3 * fabs((double)ref[idx]))
      failures++;
  }
  free(ref);
  if (failures) {
    fprintf(stderr, "%s: verification failed at N=%d (%d mismatches, max abs err %.3e)\n",
            BACKEND, n, failures, max_abs);
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  double target_seconds = 0.2;
  int argi = 1;

  if (argc > 1 && strcmp(argv[1], "--target") == 0) {
    if (argc < 3) {
      fprintf(stderr, "usage: %s [--target SECONDS] N1 N2 ...\n", argv[0]);
      return 1;
    }
    target_seconds = atof(argv[2]);
    argi = 3;
  }
  if (argi >= argc) {
    fprintf(stderr, "usage: %s [--target SECONDS] N1 N2 ...\n", argv[0]);
    return 1;
  }

#if defined(USE_BLIS)
  bli_thread_set_num_threads(1);
  fprintf(stderr, "%s: threads=%ld\n", BACKEND,
          (long)bli_thread_get_num_threads());
#elif defined(USE_OPENBLAS)
  openblas_set_num_threads(1);
  fprintf(stderr, "%s: threads=%d\n", BACKEND, openblas_get_num_threads());
#endif

  cpu_warmup();

  int max_n = 0;
  for (int i = argi; i < argc; i++) {
    int n = atoi(argv[i]);
    if (n > max_n)
      max_n = n;
  }

#if defined(USE_EXO)
  g_bpacked = xmalloc((size_t)max_n * max_n * sizeof(float));
  g_apadded = xmalloc((size_t)6 * max_n * sizeof(float));
#endif

  for (int i = argi; i < argc; i++) {
    int n = atoi(argv[i]);
    if (n <= 0) {
      fprintf(stderr, "invalid size: %s\n", argv[i]);
      return 1;
    }

    float *A = xmalloc((size_t)n * n * sizeof(float));
    float *B = xmalloc((size_t)n * n * sizeof(float));
    float *C = xmalloc((size_t)n * n * sizeof(float));

    for (size_t idx = 0; idx < (size_t)n * n; idx++)
      A[idx] = next_float();
    for (size_t idx = 0; idx < (size_t)n * n; idx++)
      B[idx] = next_float();
    for (size_t idx = 0; idx < (size_t)n * n; idx++)
      C[idx] = 0.0f;

    for (int w = 0; w < WARMUP_ROUNDS; w++)
      run_gemm(n, A, B, C);

    /* Verify small/medium sizes against a double-precision reference. */
    if (n <= 512 && !verify(n, A, B, C)) {
      free(A);
      free(B);
      free(C);
      return 1;
    }

    double t0 = now_seconds();
    run_gemm(n, A, B, C);
    double t1 = now_seconds();
    double estimate = t1 - t0;
    if (estimate <= 0.0)
      estimate = 1e-9;

    int iters = (int)(target_seconds / estimate);
    if (iters < 5)
      iters = 5;
    if (iters > 500000)
      iters = 500000;

    t0 = now_seconds();
    for (int it = 0; it < iters; it++)
      run_gemm(n, A, B, C);
    t1 = now_seconds();

    double seconds = (t1 - t0) / (double)iters;
    double gflops = 2.0 * (double)n * n * n / (seconds * 1e9);
    printf("%s\t%d\t%.9g\t%.6f\n", BACKEND, n, seconds, gflops);
    fflush(stdout);

    free(A);
    free(B);
    free(C);
  }

#if defined(USE_EXO)
  free(g_bpacked);
  free(g_apadded);
#endif
  return 0;
}
