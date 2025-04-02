ICON1 = "icon1"
ICON2 = "icon2"


def solve_tango_puzzle(rows, cols,
                       equals_constraints,
                       cross_constraints,
                       row_quota,
                       col_quota,
                       initial_grid=None):
    """
    Solves a Tango puzzle using backtracking.

    The solution respects the following constraints:
      - Each cell must be either 'icon1' or 'icon2'.
      - No more than two identical icons may appear consecutively in any row or column.
      - Each row must contain exactly row_quota[r] instances of 'icon1'.
      - Each column must contain exactly col_quota[c] instances of 'icon1'.
      - Cells linked by 'equals' constraints must contain the same icon.
      - Cells linked by 'cross' constraints must contain different icons.

    Args:
        rows (int): Number of rows in the board.
        cols (int): Number of columns in the board.
        equals_constraints (list): List of cell pairs that must be equal.
        cross_constraints (list): List of cell pairs that must differ.
        row_quota (list): Number of 'icon1' values required per row.
        col_quota (list): Number of 'icon1' values required per column.
        initial_grid (list of list, optional): Pre-filled grid. Defaults to None.

    Returns:
        list of list: Solved grid or an empty list if no valid solution exists.
    """
    from copy import deepcopy

    grid = deepcopy(initial_grid) if initial_grid else [[None] * cols for _ in range(rows)]

    def backtrack(r, c):
        if c == cols:
            if not row_fits_quota(grid, r, row_quota[r]):
                return False
            return backtrack(r + 1, 0)

        if r == rows:
            return columns_fit_quota(grid, col_quota)

        if grid[r][c] in (ICON1, ICON2):
            if satisfies_constraints(grid, r, c,
                                     equals_constraints,
                                     cross_constraints,
                                     row_quota,
                                     col_quota):
                return backtrack(r, c + 1)
            return False

        for icon in (ICON1, ICON2):
            grid[r][c] = icon
            if satisfies_constraints(grid, r, c,
                                     equals_constraints,
                                     cross_constraints,
                                     row_quota,
                                     col_quota):
                if backtrack(r, c + 1):
                    return True
            grid[r][c] = None

        return False

    return grid if backtrack(0, 0) else []


def row_fits_quota(grid, row_idx, needed_icon1):
    """
    Checks if a row contains exactly the required number of 'icon1' values.

    Args:
        grid (list of list): Puzzle grid.
        row_idx (int): Row index to check.
        needed_icon1 (int): Expected number of 'icon1' in the row.

    Returns:
        bool: True if quota is satisfied, False otherwise.
    """
    actual_icon1 = sum(1 for cell in grid[row_idx] if cell == ICON1)
    return actual_icon1 == needed_icon1


def columns_fit_quota(grid, col_quota):
    """
    Validates whether each column meets its 'icon1' quota.

    Args:
        grid (list of list): Fully assigned puzzle grid.
        col_quota (list): Number of 'icon1' values required per column.

    Returns:
        bool: True if all column quotas are satisfied, False otherwise.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for c in range(cols):
        icon1_count = sum(1 for r in range(rows) if grid[r][c] == ICON1)
        if icon1_count != col_quota[c]:
            return False
    return True


def satisfies_constraints(grid, r, c,
                          equals_constraints,
                          cross_constraints,
                          row_quota,
                          col_quota):
    """
    Validates all constraints after assigning a value at (r, c).

    Args:
        grid (list of list): Current puzzle grid.
        r (int): Row index of the last changed cell.
        c (int): Column index of the last changed cell.
        equals_constraints (list): Pairs of cells that must match.
        cross_constraints (list): Pairs of cells that must differ.
        row_quota (list): Icon1 quota per row.
        col_quota (list): Icon1 quota per column.

    Returns:
        bool: True if all constraints are currently satisfied.
    """
    val = grid[r][c]
    if val not in (ICON1, ICON2):
        return True

    rows = len(grid)
    cols = len(grid[0])

    if sum(1 for x in grid[r] if x == ICON1) > row_quota[r]:
        return False

    if sum(1 for rr in range(rows) if grid[rr][c] == ICON1) > col_quota[c]:
        return False

    if has_run_of_three(grid[r]):
        return False

    if has_run_of_three([grid[rr][c] for rr in range(rows)]):
        return False

    for (r1, c1), (r2, c2) in equals_constraints:
        v1, v2 = grid[r1][c1], grid[r2][c2]
        if v1 is not None and v2 is not None and v1 != v2:
            return False

    for (r1, c1), (r2, c2) in cross_constraints:
        v1, v2 = grid[r1][c1], grid[r2][c2]
        if v1 is not None and v2 is not None and v1 == v2:
            return False

    return True


def has_run_of_three(cells):
    """
    Detects if a sequence contains three identical consecutive values.

    Args:
        cells (list): Sequence of icons.

    Returns:
        bool: True if three in a row are identical, False otherwise.
    """
    consecutive = 1
    prev = cells[0]

    for i in range(1, len(cells)):
        curr = cells[i]
        if curr is not None and curr == prev and prev is not None:
            consecutive += 1
        else:
            consecutive = 1
        prev = curr

        if consecutive == 3:
            return True

    return False