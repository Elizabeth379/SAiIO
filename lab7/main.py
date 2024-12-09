import numpy as np
from termcolor import colored
from methods import hungarian_algorithm


def print_result(C: np.ndarray, positions: list[tuple[int]]) -> None:
    n, m = C.shape
    for i in range(n):
        row = []
        for j in range(m):
            if (i, j) in positions:
                row.append(colored(f"{C[i, j]:2}", "yellow", attrs=["bold"]))
            else:
                row.append(f"{C[i, j]:2}")
        print(" ".join(row))


if __name__ == "__main__":
    C = np.array([
        [7, 2, 1, 9, 4],
        [9, 6, 9, 5, 5],
        [3, 8, 3, 1, 8],
        [7, 9, 4, 2, 2],
        [8, 4, 7, 4, 8]
    ])

    positions = hungarian_algorithm(C)
    print_result(C, positions)