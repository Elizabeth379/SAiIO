import numpy as np


def solve(A: np.ndarray, P: int, Q: int) -> tuple[np.ndarray, np.ndarray, tuple[int]]:
    B = np.zeros(A.shape, dtype=np.int32)
    C = np.zeros(A.shape, dtype=np.int32)

    for p in range(P):
        for q in range(Q + 1):
            if p == 0:
                B[p, q] = A[p, q]
                C[p, q] = q
            else:
                max_value = 0
                max_index = 0

                for i in range(q + 1):
                    temp = max(A[p, i] + B[p - 1, q - i], max_value)

                    if max_value != temp:
                        max_value = temp
                        max_index = i

                B[p, q] = max_value
                C[p, q] = max_index

    q = Q
    p = P
    result = []
    while p > 0:
        result.append((p, C[p - 1, q]))
        q -= C[p - 1, q]
        p -= 1

    return B, C, result


if __name__ == '__main__':
    P = 3;
    Q = 3
    A = np.array([[0, 1, 2, 3],
                  [0, 0, 1, 2],
                  [0, 2, 2, 3]], dtype=np.int32)

    B, C, result = solve(A, P, Q)

    print(B)
    print(C)
    print(result)
