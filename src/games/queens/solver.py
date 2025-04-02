from collections import defaultdict


def is_safe(pos, queens, rows_used, cols_used):
    """
    Determines if a queen can be safely placed at the specified position.

    Args:
        pos (tuple): The (row, col) position to check.
        queens (list): Current list of placed queen positions.
        rows_used (set): Set of rows already occupied by queens.
        cols_used (set): Set of columns already occupied by queens.

    Returns:
        bool: True if the position is safe, False otherwise.
    """
    r, c = pos

    if r in rows_used or c in cols_used:
        return False

    # Check adjacency (including diagonals)
    for qr, qc in queens:
        if abs(qr - r) == 1 and abs(qc - c) <= 1:
            return False

    return True


def solve_queens(maze_map, num_clusters):
    """
    Solves the LinkedIn Queens puzzle using backtracking.

    Each cluster must contain exactly one queen. Queens must not be in the same
    row, column, or adjacent to each other (including diagonals).

    Args:
        maze_map (dict): Mapping of (row, col) → cluster_id.
        num_clusters (int): Number of distinct color clusters.

    Returns:
        list of tuple: A list of (row, col) queen positions if a solution is found;
                       otherwise, an empty list.
    """
    # Group all positions by cluster
    cluster_cells = defaultdict(list)
    for pos, cluster_id in maze_map.items():
        cluster_cells[cluster_id].append(pos)

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