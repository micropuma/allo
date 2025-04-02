import allo
import numpy as np

# 矩阵维度
M, N, K = 32, 32, 32

# 生成随机输入数据
np_A = np.random.random((M, K)).astype(np.float32)
np_B = np.random.random((K, N)).astype(np.float32)
allo_C = np.zeros((M, N), dtype=np.float32)

# 载入生成的加速器
mod = allo.load("./gemm.prj")  # 载入 Allo 生成的硬件模块
mod(np_A, np_B, allo_C)  # 执行计算

# 验证计算结果
np.testing.assert_allclose(allo_C, np.matmul(np_A, np_B), rtol=1e-5, atol=1e-5)
print("GEMM computation successful!")

