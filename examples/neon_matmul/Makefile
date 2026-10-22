CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -Wno-unused-parameter
LDLIBS ?= -lm
EXOCC ?= exocc

neon_matmul: main.o neon_matmul.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

neon_matmul.c neon_matmul.h: arm_matmul.py
	$(EXOCC) -o . --stem neon_matmul $<

main.o: main.c neon_matmul.h
neon_matmul.o: neon_matmul.c

.PHONY: run clean
run: neon_matmul
	./neon_matmul

clean:
	$(RM) neon_matmul neon_matmul.c neon_matmul.h neon_matmul.d *.o
	$(RM) -r __pycache__
