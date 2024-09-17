import numpy as np

def simplex(c, A, b):
    """
    Симплекс-метод для решения задачи ЛП
    maximize c^T x
    subject to Ax <= b, x >= 0
    """
    num_vars = len(c)
    num_constraints = len(b)

    # Формируем симплекс-таблицу
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

    # Заполняем строку целевой функции
    tableau[0, :num_vars] = -c

    # Заполняем таблицу коэффициентами ограничений
    for i in range(num_constraints):
        tableau[i + 1, :num_vars] = A[i]
        tableau[i + 1, num_vars + i] = 1
        tableau[i + 1, -1] = b[i]

    # Итерации симплекс-метода
    while np.any(tableau[0, :-1] < 0):  # Пока есть отрицательные коэффициенты в строке целевой функции
        # Находим разрешающий столбец
        pivot_col = np.argmin(tableau[0, :-1])

        # Находим разрешающую строку
        ratios = []
        for i in range(1, len(tableau)):
            if tableau[i, pivot_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
        pivot_row = np.argmin(ratios) + 1

        # Пивотирование
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Извлекаем решение
    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[1:, i]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == num_constraints - 1:
            solution[i] = tableau[np.argmax(col) + 1, -1]

    return solution, tableau[0, -1]


def gomory_cut(tableau):
    """
    Создание отсечения Гомори для дробного решения.
    """
    # Находим строку с дробной правой частью
    fractional_row = None
    for i in range(1, len(tableau)):
        if tableau[i, -1] % 1 != 0:
            fractional_row = tableau[i]
            break

    if fractional_row is None:
        raise ValueError("Не удалось найти дробное решение для построения отсечения Гомори.")

    # Формирование нового ограничения на основе дробной части
    new_constraint = np.zeros(len(fractional_row) - 1)
    for j in range(len(new_constraint)):
        if fractional_row[j] % 1 != 0:
            new_constraint[j] = np.floor(fractional_row[j]) - fractional_row[j]
        else:
            new_constraint[j] = 0

    new_constraint = np.append(new_constraint, -fractional_row[-1])
    return new_constraint


def solve_gomory(c, A, b):
    """
    Решение задачи ЛП с использованием метода отсекающих ограничений Гомори.
    """
    solution, objective_value = simplex(c, A, b)

    while not np.all(np.floor(solution) == solution):  # Пока решение не целочисленное
        print("Текущее решение:", solution)

        # Добавляем отсечение Гомори
        num_vars = len(c)
        num_constraints = len(b)
        tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
        tableau[0, :num_vars] = -c

        for i in range(num_constraints):
            tableau[i + 1, :num_vars] = A[i]
            tableau[i + 1, num_vars + i] = 1
            tableau[i + 1, -1] = b[i]

        new_constraint = gomory_cut(tableau)

        # Добавляем новое ограничение в таблицу
        A = np.vstack([A, new_constraint[:-1]])
        b = np.append(b, new_constraint[-1])

        # Перезапускаем симплекс-метод с новыми ограничениями
        solution, objective_value = simplex(c, A, b)

    return solution, objective_value


# Пример задачи
c = np.array([3, 2])  # Целевая функция
A = np.array([[2, 1],  # Ограничения
              [1, 2]])
b = np.array([4, 3])

solution, objective_value = solve_gomory(c, A, b)
print(f"Оптимальное целочисленное решение: {solution}")
print(f"Оптимальное значение целевой функции: {objective_value}")
