def is_safe(pos, queens, rows_used, cols_used):
    """
    Checks if a queen can be safely placed at pos (row, col).
    """
    r, c = pos

    if r in rows_used or c in cols_used:
        return False

    for qr, qc in queens:
        if abs(qr - r) == 1 and abs(qc - c) <= 1:
            return False  # Adjacent (including diagonals)

    return True


def solve_queens(maze_map, num_clusters):
    """
    Solves the LinkedIn Queens puzzle using backtracking.

    Args:
        maze_map (dict): (row, col) → cluster_id
        num_clusters (int): number of color clusters

    Returns:
        list of (row, col): queen positions, or [] if no solution
    """
    from collections import defaultdict

    # Group cells by cluster
    cluster_cells = defaultdict(list)
    for pos, cluster_id in maze_map.items():
        cluster_cells[cluster_id].append(pos)

    # Backtracking state
    solution = []
    rows_used = set()
    cols_used = set()

    def backtrack(cluster_idx):
        if cluster_idx == num_clusters:
            return True

        for cell in cluster_cells[cluster_idx]:
            if is_safe(cell, solution, rows_used, cols_used):
                solution.append(cell)
                rows_used.add(cell[0])
                cols_used.add(cell[1])

                if backtrack(cluster_idx + 1):
                    return True

                # Backtrack
                solution.pop()
                rows_used.remove(cell[0])
                cols_used.remove(cell[1])

        return False

    success = backtrack(0)
    return solution if success else []