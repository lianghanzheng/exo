from __future__ import annotations

import os
import sys

from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *

# Hide output when running through exocc.
if __name__ != "__main__" and hasattr(os, "devnull"):
    sys.stdout = open(os.devnull, "w")


# Algorithm definition
@proc
def rank_k_reduce_6x8(
    K: size, A: f32[6, K] @ DRAM, B: f32[K, 8] @ DRAM, C: f32[6, 8] @ DRAM
):
    for i in seq(0, 6):
        for j in seq(0, 8):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


print("=============Original Matmul==============")
print(rank_k_reduce_6x8)

# In this example, we want the computation to be "output stationary", which means,
# we want to preallocate all the output registers at the start.
#
# NEON exposes 32 128-bit vector registers. A 6x8 f32 tile occupies
# 6 * (8 / 4) = 12 registers, leaving plenty of room for A and B.
neon = rename(rank_k_reduce_6x8, "rank_k_reduce_6x8_scheduled")
neon = reorder_loops(neon, "j k")
neon = reorder_loops(neon, "i k")

# The staging of C will cause us to consume 12 of the vector registers
neon = divide_loop(neon, "for j in _: _", 4, ["jo", "ji"], perfect=True)
neon = stage_mem(neon, "for k in _:_", "C[0:6, 0:8]", "C_reg")
neon = simplify(neon)

# Reshape C_reg so we can map it into vector registers
neon = divide_dim(neon, "C_reg:_", 1, 4)
neon = repeat(divide_loop)(neon, "for i1 in _: _", 4, ["i2", "i3"], perfect=True)
neon = simplify(neon)

# Map C_reg operations to vector instructions
neon = set_memory(neon, "C_reg:_", Neon)
neon = replace_all(neon, neon_vld_4xf32)
neon = replace_all(neon, neon_vst_4xf32)
neon = simplify(neon)

# B is easy, it is just two vector loads
neon = stage_mem(neon, "for i in _:_", "B[k, 0:8]", "B_reg")
neon = simplify(neon)
neon = divide_loop(neon, "for i0 in _: _ #1", 4, ["io", "ii"], perfect=True)
neon = divide_dim(neon, "B_reg:_", 0, 4)
neon = set_memory(neon, "B_reg:_", Neon)
neon = simplify(neon)
neon = replace_all(neon, neon_vld_4xf32)
neon = simplify(neon)

# The final part is staging A, which we broadcast into a vector register.
neon = bind_expr(neon, "A[i, k]", "A_reg")
neon = expand_dim(neon, "A_reg", 4, "ji")
neon = lift_alloc(neon, "A_reg", n_lifts=2)
neon = fission(neon, neon.find("A_reg[ji] = _").after(), n_lifts=2)
neon = remove_loop(neon, "for jo in _: _")
neon = set_memory(neon, "A_reg:_", Neon)
neon = replace_all(neon, neon_broadcast_4xf32)

# Replace the multiply-add instructions with NEON instructions
neon = replace_all(neon, neon_vfmadd_4xf32_4xf32)
neon = simplify(neon)

print("=============Optimized Matmul==============")
print(neon)
