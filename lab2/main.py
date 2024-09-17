import numpy as np
import math
from scipy.optimize import linprog  # Импортируем функцию linprog

# Симплекс-метод для решения линейной задачи
def simplex_method(A, b, c):
    m, n = A.shape
    B = list(range(n - m, n))  # базисные индексы
    N = list(range(n - m))  # небазисные индексы

    # Начальное базисное решение
    A_b_inv = np.linalg.inv(A[:, B])
    x_b = np.dot(A_b_inv, b)
    x = np.zeros(n)
    x[B] = x_b

    # Итерации симплекс-метода
    while True:
        c_b = c[B]
        c_n = c[N]

        # Найдем вектор теневых цен (потенциалов)
        y = np.dot(c_b, A_b_inv)

        # Вычислим оценку редуцированных затрат
        delta = c_n - np.dot(y, A[:, N])

        # Условие оптимальности (если все редуцированные затраты >= 0)
        if all(d >= 0 for d in delta):
            return x, B

        # Найдем входящую переменную
        q = N[np.argmin(delta)]

        # Найдем направление изменения решения
        d_b = np.dot(A_b_inv, A[:, q])
        if all(d <= 0 for d in d_b):
            return None, "Целевая функция не ограничена"

        # Найдем выходящую переменную
        theta = min(x[B] / d_b[d_b > 0])
        p = B[np.argmin(x[B] / d_b[d_b > 0])]

        # Обновляем базис
        x[B] -= theta * d_b
        x[q] = theta
        B[B.index(p)] = q
        N[N.index(q)] = p

        # Обновляем обратную матрицу базиса
        A_b_inv = np.linalg.inv(A[:, B])

# Метод отсекающего ограничения Гомори
def gomory_cutting_plane_method(A, b, c):
    m, n = A.shape

    while True:
        # Решаем текущую задачу линейного программирования
        res = linprog(c, A_eq=A, b_eq=b, method='simplex')

        if res.status != 0:
            return "Задача несовместна или целевая функция неограничена сверху"

        x = res.x

        # Проверяем, является ли текущее решение целым
        if all(math.isclose(xi, round(xi)) for xi in x[:n]) and not np.allclose(x[:n], 0):
            return f"Оптимальный план: {x[:n]}"

        # Поиск дробной компоненты решения
        try:
            fractional_index = next(i for i in range(n) if not math.isclose(x[i], round(x[i])))
        except StopIteration:
            return f"Все переменные уже целые, но решение не найдено"

        # Добавляем отсекающее ограничение Гомори
        gomory_cut = x[fractional_index] - math.floor(x[fractional_index])
        new_constraint = np.floor(A[fractional_index]) - A[fractional_index]
        b = np.append(b, -gomory_cut)
        A = np.vstack([A, new_constraint])

        print(f"Добавлено отсекающее ограничение: {new_constraint} <= {gomory_cut}")



# Пример использования
A = np.array([
    [3, 2, 1, 0],
    [-3, 2, 0, 1]
])  # Матрица A
b = np.array([6, 0])  # Вектор b
c = np.array([0, 1, 0, 0])  # Вектор c

solution = gomory_cutting_plane_method(A, b, c)
print(solution)
