import allo
from allo.ir.types import int32

M, N = 1024, 1024


def matrix_add(A: int32[M, N]) -> int32[M, N]:
    B: int32[M, N] = 0
    for i, j in allo.grid(M, N):
        B[i, j] = A[i, j] + 1
    return B

import inspect

src = inspect.getsource(matrix_add)
print(src)

import ast, astpretty

tree = ast.parse(src)
astpretty.pprint(tree, indent=2, show_offsets=False)

s = allo.customize(matrix_add, verbose=True)
print(s)


