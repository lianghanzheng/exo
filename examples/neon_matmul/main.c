#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neon_matmul.h"

#define M 6
#define N 8
#define DEFAULT_K 2048
#define DEFAULT_ITERS 1000
#define WARMUP_ITERS 20

typedef void (*matmul_fn)(void *, int_fast32_t, const float *, const float *,
                          float *);

static void *xmalloc(size_t bytes) {
  void *ptr = malloc(bytes);
  if (ptr == NULL) {
    fprintf(stderr, "out of memory allocating %zu bytes\n", bytes);
    exit(1);
  }
  return ptr;
}

/* Deterministic xorshift RNG so repeated runs are comparable. */
static uint32_t rng_state = 0x12345678u;

static float next_float(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return ((rng_state >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f;
}

static void initialize(int_fast32_t K, float *A, float *B) {
  for (int_fast32_t i = 0; i < M * K; i++)
    A[i] = next_float();
  for (int_fast32_t i = 0; i < K * N; i++)
    B[i] = next_float();
}

/* Double-precision reference: C = A * B. */
static void reference_matmul(int_fast32_t K, const float *A, const float *B,
                             float *C) {
  for (int_fast32_t i = 0; i < M; i++) {
    for (int_fast32_t j = 0; j < N; j++) {
      double acc = 0.0;
      for (int_fast32_t k = 0; k < K; k++)
        acc += (double)A[i * K + k] * (double)B[k * N + j];
      C[i * N + j] = (float)acc;
    }
  }
}

/* Returns 1 on success. Also reports the maximum absolute/relative error. */
static int check_result(const char *name, const float *ref, const float *got,
                        double *max_abs_out, double *max_rel_out) {
  double max_abs = 0.0;
  double max_rel = 0.0;
  int failures = 0;
  const double atol = 1e-4;
  const double rtol = 1e-4;

  for (int idx = 0; idx < M * N; idx++) {
    double expect = (double)ref[idx];
    double actual = (double)got[idx];
    double diff = fabs(actual - expect);
    double rel = diff / fmax(fabs(expect), 1e-6);
    if (diff > max_abs)
      max_abs = diff;
    if (rel > max_rel)
      max_rel = rel;
    if (diff > atol + rtol * fabs(expect)) {
      if (failures < 5)
        fprintf(stderr,
                "  %s mismatch [%d]: got %.8f, expected %.8f (abs err %.3e)\n",
                name, idx, actual, expect, diff);
      failures++;
    }
  }

  *max_abs_out = max_abs;
  *max_rel_out = max_rel;
  return failures == 0;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double benchmark(matmul_fn fn, int_fast32_t K, const float *A,
                        const float *B, float *C, int iters) {
  for (int i = 0; i < WARMUP_ITERS; i++)
    fn(NULL, K, A, B, C);

  double start = now_seconds();
  for (int i = 0; i < iters; i++)
    fn(NULL, K, A, B, C);
  double elapsed = now_seconds() - start;

  return elapsed / (double)iters;
}

static void report_benchmark(const char *name, double seconds,
                             int_fast32_t K) {
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double usec = seconds * 1e6;
  double gflops = flops / (seconds * 1e9);
  printf("  %-28s %9.2f us/call   %8.2f GFLOP/s\n", name, usec, gflops);
}

int main(int argc, char **argv) {
  int_fast32_t K = DEFAULT_K;
  int iters = DEFAULT_ITERS;

  if (argc > 1)
    K = (int_fast32_t)strtol(argv[1], NULL, 10);
  if (argc > 2)
    iters = (int)strtol(argv[2], NULL, 10);
  if (K <= 0 || iters <= 0) {
    fprintf(stderr, "usage: %s [K] [iterations]\n", argv[0]);
    return 1;
  }

  float *A = xmalloc((size_t)M * K * sizeof(float));
  float *B = xmalloc((size_t)K * N * sizeof(float));
  float *C_ref = xmalloc((size_t)M * N * sizeof(float));
  float *C_plain = xmalloc((size_t)M * N * sizeof(float));
  float *C_neon = xmalloc((size_t)M * N * sizeof(float));

  printf("Matrix multiply microkernel: C[%d,%d] += A[%d,%" PRIdFAST32
         "] * B[%" PRIdFAST32 ",%d]\n",
         M, N, M, K, K, N);
  printf("K = %" PRIdFAST32 ", iterations = %d, %.0f FLOPs/call\n\n", K, iters,
         2.0 * M * N * K);

  initialize(K, A, B);
  reference_matmul(K, A, B, C_ref);

  /* Verify both the unscheduled and the scheduled Exo procedure. */
  for (int i = 0; i < M * N; i++) {
    C_plain[i] = 0.0f;
    C_neon[i] = 0.0f;
  }
  rank_k_reduce_6x8(NULL, K, A, B, C_plain);
  rank_k_reduce_6x8_scheduled(NULL, K, A, B, C_neon);

  double max_abs, max_rel;
  int ok_plain =
      check_result("rank_k_reduce_6x8", C_ref, C_plain, &max_abs, &max_rel);
  printf("Verification:\n");
  printf("  %-28s %s (max abs err %.3e, max rel err %.3e)\n",
         "rank_k_reduce_6x8", ok_plain ? "PASS" : "FAIL", max_abs, max_rel);

  int ok_neon = check_result("rank_k_reduce_6x8_scheduled", C_ref, C_neon,
                             &max_abs, &max_rel);
  printf("  %-28s %s (max abs err %.3e, max rel err %.3e)\n",
         "rank_k_reduce_6x8_scheduled", ok_neon ? "PASS" : "FAIL", max_abs,
         max_rel);

  if (!ok_plain || !ok_neon) {
    fprintf(stderr, "result verification failed\n");
    return 1;
  }

  /* Performance: the results are overwritten, so no need to clear C. */
  double t_plain = benchmark(rank_k_reduce_6x8, K, A, B, C_plain, iters);
  double t_neon =
      benchmark(rank_k_reduce_6x8_scheduled, K, A, B, C_neon, iters);

  printf("\nPerformance:\n");
  report_benchmark("rank_k_reduce_6x8", t_plain, K);
  report_benchmark("rank_k_reduce_6x8_scheduled", t_neon, K);
  printf("  %-28s %9.2fx\n", "speedup", t_plain / t_neon);

  free(A);
  free(B);
  free(C_ref);
  free(C_plain);
  free(C_neon);
  return 0;
}
