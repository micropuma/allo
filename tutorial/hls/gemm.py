import allo
from allo.ir.types import float32
import numpy as np

M, N, K = 32, 32, 32

def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0.0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C

# 针对gemm计算做调度
s = allo.customize(gemm)
s.reorder("k", "j")
# 给C数组在最外层循环中添加buffer
s.buffer_at(s.C, axis="i")
mod = s.build(target="vitis_hls", mode="hw_emu", project="gemm.prj")

np_A = np.random.random((M, K)).astype(np.float32)
np_B = np.random.random((K, N)).astype(np.float32)
allo_C = np.zeros((M, N), dtype=np.float32)
mod(np_A, np_B, allo_C)
np.testing.assert_allclose(allo_C, np.matmul(np_A, np_B), rtol=1e-5, atol=1e-5)
