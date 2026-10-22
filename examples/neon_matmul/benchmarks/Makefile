CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -Wno-unused-parameter
EXO_DIR ?= ..
BLIS_DIR ?= $(HOME)/GitRepo/blis
OPENBLAS_DIR ?= $(HOME)/GitRepo/openblas

.PHONY: all bench clean
all: bench_exo bench_blis bench_openblas

bench: all
	python run_benchmarks.py

bench_exo: bench_gemm.c $(EXO_DIR)/neon_matmul.c $(EXO_DIR)/neon_matmul.h
	$(CC) $(CFLAGS) -DUSE_EXO -I$(EXO_DIR) -o $@ bench_gemm.c $(EXO_DIR)/neon_matmul.c -lm

bench_blis: bench_gemm.c
	$(CC) $(CFLAGS) -DUSE_BLIS -isystem $(BLIS_DIR)/include/arm64 -o $@ bench_gemm.c \
		$(BLIS_DIR)/lib/arm64/libblis.a -lpthread -lm

bench_openblas: bench_gemm.c
	$(CC) $(CFLAGS) -DUSE_OPENBLAS -isystem $(OPENBLAS_DIR) -o $@ bench_gemm.c \
		$(OPENBLAS_DIR)/libopenblas.a -lpthread -lm

clean:
	$(RM) bench_exo bench_blis bench_openblas
