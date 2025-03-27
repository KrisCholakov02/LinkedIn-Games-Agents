import copy

ICON1 = "icon1"
ICON2 = "icon2"


def solve_tango_puzzle(rows,
                       cols,
                       equals_constraints,
                       cross_constraints,
                       row_quota,
                       col_quota):
    """
    Solves a Tango puzzle via backtracking.

    Puzzle constraints:
      1) Each cell must be icon1 or icon2.
      2) No more than two identical icons consecutively (horizontal or vertical).
      3) Each row/column must contain row_quota[r] (resp. col_quota[c]) icon1s.
         (The rest are icon2s.)
      4) Equals constraints => both cells must hold the same icon.
      5) Cross constraints => both cells must hold different icons.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        equals_constraints (list of ((r1,c1), (r2,c2))):
            Pairs of cell coordinates that must be the same icon.
        cross_constraints (list of ((r1,c1), (r2,c2))):
            Pairs of cell coordinates that must be different icons.
        row_quota (list of int):
            row_quota[r] = how many icon1 must appear in row r.
        col_quota (list of int):
            col_quota[c] = how many icon1 must appear in column c.

    Returns:
        grid (list of list): A 2D solution grid with "icon1" or "icon2" in each cell.
                             If no solution is found, returns an empty list.
    """
    # Initialize puzzle grid with None = unfilled
    grid = [[None for _ in range(cols)] for _ in range(rows)]

    def backtrack(r, c):
        """ Tries to place icon1 or icon2 in grid[r][c], then moves on. """
        if c == cols:
            # move to next row
            return backtrack(r + 1, 0)
        if r == rows:
            # All rows filled => solution
            return True

        # Try both icons
        for icon in (ICON1, ICON2):
            grid[r][c] = icon
            if satisfies_constraints(grid, r, c,
                                     equals_constraints,
                                     cross_constraints,
                                     row_quota,
                                     col_quota):
                if backtrack(r, c + 1):
                    return True
            # revert
            grid[r][c] = None

        return False

    # Start recursion
    success = backtrack(0, 0)
    if not success:
        # If puzzle is guaranteed to be solvable, we never reach here,
        # but in case it fails, return an empty solution or raise an error
        return []
    return grid


def satisfies_constraints(grid,
                          r, c,
                          equals_constraints,
                          cross_constraints,
                          row_quota,
                          col_quota):
    """
    Checks if the partial assignment in grid[r][c] is consistent with:
      - row/col icon1 quotas
      - no more than 2 identical icons consecutively in row/column
      - equals constraints (pairs must hold the same icon)
      - cross constraints (pairs must differ)

    Returns:
        True if valid so far, False if constraint is violated.
    """
    val = grid[r][c]
    if val not in (ICON1, ICON2):
        return True  # cell is not assigned or invalid

    rows = len(grid)
    cols = len(grid[0])

    # 1) Row icon1 usage not to exceed row_quota[r]
    row_icon1_count = sum(1 for x in grid[r] if x == ICON1)
    if row_icon1_count > row_quota[r]:
        return False

    # 2) Column icon1 usage not to exceed col_quota[c]
    col_icon1_count = 0
    for rr in range(rows):
        if grid[rr][c] == ICON1:
            col_icon1_count += 1
    if col_icon1_count > col_quota[c]:
        return False

    # 3) No runs of 3 same icons in row r
    if has_run_of_three(grid[r]):
        return False

    # 4) No runs of 3 same icons in column c
    #    Build the column as a list
    column_cells = [grid[row][c] for row in range(rows)]
    if has_run_of_three(column_cells):
        return False

    # 5) '=' constraints => same icon
    for (r1, c1), (r2, c2) in equals_constraints:
        # If both assigned, must match
        if grid[r1][c1] is not None and grid[r2][c2] is not None:
            if grid[r1][c1] != grid[r2][c2]:
                return False

    # 6) 'cross' constraints => different icons
    for (r1, c1), (r2, c2) in cross_constraints:
        if grid[r1][c1] is not None and grid[r2][c2] is not None:
            if grid[r1][c1] == grid[r2][c2]:
                return False

    return True


def has_run_of_three(cells):
    """
    Returns True if `cells` (list of icon1/icon2/None) has
    any run of 3 identical icons consecutively.
    """
    consecutive = 1
    prev = cells[0]
    for i in range(1, len(cells)):
        curr = cells[i]
        if curr is not None and curr == prev:
            consecutive += 1
        else:
            consecutive = 1
        prev = curr
        if consecutive == 3:
            return True
    return False
