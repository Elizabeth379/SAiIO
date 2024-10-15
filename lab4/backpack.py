def backpack(w, v, cap):
    matrix = [[0 for _ in range(cap + 1)] for _ in range(len(w) + 1)]

    for row in range(len(w)):
        for curr_cap in range(cap + 1):
            if w[row] <= curr_cap:
                matrix[row + 1][curr_cap] = max(v[row] + matrix[row][curr_cap - w[row]], matrix[row][curr_cap])
            else:
                matrix[row + 1][curr_cap] = matrix[row][curr_cap]

    return matrix[-1][-1]