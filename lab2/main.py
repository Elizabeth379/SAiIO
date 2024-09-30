import numpy as np
import math


def first_phase(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    m, n = A.shape
    n_set = set(range(1, n + 1))

    mask = (b < 0)
    b[mask] *= -1
    A[mask] *= -1

    c_aux = np.concatenate((np.zeros(n), np.full(m, -1)))
    A_aux = np.concatenate((A, np.eye(m)), axis=1)
    x_aux = np.concatenate((np.zeros(n), b))
    B_aux = [i + n for i in range(1, m + 1)]

    x_aux, B_aux = second_phase(A_aux, c_aux, x_aux, B_aux)

    if np.any(x_aux[-m:] != 0):
        raise Exception('This problem is infeasible!')

    x = x_aux[:n]
    B = B_aux

    while True:
        if set(B).issubset(n_set):
            return x, A, B

        j_k = max(B)
        k = B.index(j_k)

        j_list = list(n_set - set(B))
        A_B = A_aux[:, np.array(B) - 1]
        A_B_inverted = np.linalg.inv(A_B)

        l = [(j, A_B_inverted.dot(A_aux[:, j - 1])) for j in j_list]

        was = False
        for j, l_j in l:
            if l_j[k] != 0:
                B[k] = j
                was = True
                break

        if not was:
            i = j_k - n - 1

            A_aux = np.delete(A_aux, i, axis=0)
            A = np.delete(A, i, axis=0)

            b = np.delete(b, i)
            B.remove(j_k)


def second_phase(A: np.ndarray, c_T: np.ndarray, x_T: np.ndarray, B: list) -> tuple[np.ndarray, np.ndarray]:
    while True:
        A_B = A[:, np.array(B) - 1]
        A_B_inverted = np.linalg.inv(A_B)

        c_B_T = c_T[np.array(B) - 1]
        u_T = c_B_T.dot(A_B_inverted)
        delta_T = u_T.dot(A) - c_T

        if np.all(delta_T >= 0):
            return x_T, B

        j0 = np.where(delta_T < 0)[0][0]
        delta = delta_T[j0]

        z = A_B_inverted.dot(A[:, j0])
        theta = np.array([x_T[j - 1] / z[i] if z[i] > 0 else np.inf for i, j in enumerate(B)])
        theta0 = np.min(theta)

        if theta0 == np.inf:
            raise Exception('The target functionality of the task is not limited from ' \
                            'above on a set of acceptable plans!')

        k = theta.tolist().index(theta0)
        j_k = B[k]
        B[k] = j0 + 1

        x_T[j0] = theta0
        for i, j in enumerate(B):
            if j != j0 + 1:
                x_T[j - 1] = x_T[j - 1] - theta0 * z[i]
        x_T[j_k - 1] = 0


def get_fractional_part(number: float) -> float:
    return number - int(number) if number >= 0 else number - math.floor(number)


def homori_method(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    m, n = A.shape

    x, _, B = first_phase(c, A, b)

    n_set = set(range(1, n + 1))
    N = list(n_set - set(B))

    integer_mask = x == x.astype(np.int32)
    if all(integer_mask):
        return x
    else:
        float_basis_indices = [i for i in np.array(B) - 1 if not integer_mask[i]]
        float_index = float_basis_indices[-1]
        float_element = x[float_index]

        k = B.index(float_index + 1)

        A_B = A[:, np.array(B) - 1]
        A_N = A[:, np.array(N) - 1]

        A_B_inverted = np.linalg.inv(A_B)
        Q = A_B_inverted @ A_N

        l = Q[k]

        l_i = 0
        result = []
        for i in range(1, n + 1):
            if i in B:
                result.append(0)
            elif i in N:
                result.append(get_fractional_part(l[l_i]))
                l_i += 1
        result.append(-1)

        return np.array(result), get_fractional_part(float_element)


if __name__ == '__main__':
    c = np.array([0, 1, 0, 0])
    A = np.array([[3, 2, 1, 0],
                  [-3, 2, 0, 1]])
    b = np.array([6, 0])

    result = homori_method(c, A, b)

    print(result)
