"""
Return true if a matrix exists that satisfies the following conditions:

Given a vector a, b, check if there exists a matrix of len(a) by len(b) such that:
1. for every i in len(a), a[i] = sum(matrix[i])
2. for every j in len(b), b[j] = sum(matrix[:, j])
"""


def is_solution(matrix, a, b):
    n_row = len(a)
    n_col = len(b)
    current_row_sum = [0] * n_row
    current_col_sum = [0] * n_col
    for i in range(n_row):
        # check row sum across the matrix
        for j in range(n_col):
            current_row_sum[i] += matrix[i][j]
            current_col_sum[j] += matrix[i][j]
        if current_row_sum[i] != a[i]:
            return False
    for j in range(n_col):
        if current_col_sum[j] != b[j]:
            return False
    return True


def dfs_old(current_matrix, current_row, current_col, visited, a, b):
    if (
        current_row == len(a) - 1
        and current_col == len(b) - 1
        and is_solution(current_matrix, a, b)
    ):
        return True
    if current_row < 0 or current_row >= len(a):
        return False
    if current_col < 0 or current_col >= len(b):
        return False
    if visited[current_row][current_col] == 1:  # dont visit if already visited
        return False

    for cand_value in [0, 1]:
        if (
            visited[current_row][current_col] == 0
        ):  # if not visited, try to visit this cell
            current_matrix[current_row][current_col] = cand_value
            visited[current_row][current_col] = 1
            left = dfs_old(current_matrix, current_row, current_col - 1, visited, a, b)
            right = dfs_old(current_matrix, current_row, current_col + 1, visited, a, b)
            up = dfs_old(current_matrix, current_row - 1, current_col, visited, a, b)
            down = dfs_old(current_matrix, current_row + 1, current_col, visited, a, b)
            if any([left, right, up, down]):
                return True, current_matrix
            # backtrack
            visited[current_row][current_col] = 0  # reset_visited for next candidate
            current_matrix[current_row][current_col] = 0
    return False


def dfs(current_matrix, current_row, current_col, a, b):
    if is_solution(current_matrix, a, b):
        return True
    if current_row < 0 or current_row >= len(a):
        return False
    if current_col < 0 or current_col >= len(b):
        return False

    for cand_value in [0, 1]:
        current_matrix[current_row][current_col] = cand_value
        right = dfs(current_matrix, current_row, current_col + 1, a, b)
        down = dfs(current_matrix, current_row + 1, current_col, a, b)
        if any([right, down]):
            return True
        # reset
        current_matrix[current_row][current_col] = 0
    return False


def backtrack_solution(matrix, row, col):
    if sum(a) != sum(b):
        return False
    if row == len(a):
        return is_solution(
            matrix, a, b
        )  # if we explored all cells, check if solution is valid
    # calculate next position
    next_row = row + 1 if col == len(b) - 1 else row
    next_col = 0 if col == len(b) - 1 else col + 1
    # try both options for current cell
    for val in [0, 1]:
        matrix[row][col] = val
        if backtrack_solution(matrix, next_row, next_col):
            return True
        matrix[row][col] = 0
    return False


def solution_old(a, b):
    current_matrix = [[0] * len(b) for _ in range(len(a))]
    visited = [[0] * len(b) for _ in range(len(a))]
    return dfs_old(current_matrix, 0, 0, visited, a, b)


def solution(a, b):
    current_matrix = [[0] * len(b) for _ in range(len(a))]
    # visited = [[0] * len(b) for _ in range(len(a))]
    return dfs(current_matrix, 0, 0, a, b)


def backtrack(a, b):
    matrix = [[0] * len(b) for _ in range(len(a))]
    return backtrack_solution(matrix, 0, 0)


if __name__ == "__main__":
    a = [3, 2]
    b = [2, 1, 1, 0, 1]
    n_row = len(a)
    n_col = len(b)
    # n_row by n_col matrix
    matrix = [
        [1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
    ]

    print(is_solution(matrix, a, b))
    print(solution_old(a, b))
    print(solution(a, b))
    print(backtrack(a, b))

    a = [3, 2]
    b = [2, 1, 1, 1, 1]
    print(solution_old(a, b))
    print(backtrack(a, b))
