ICON1 = "icon1"
ICON2 = "icon2"

def solve_tango_puzzle(rows, cols,
                       equals_constraints,
                       cross_constraints,
                       row_quota,
                       col_quota,
                       initial_grid=None):
    """
    Solves a Tango puzzle with backtracking, respecting pre-filled cells
    and ensuring each row/column has exactly row_quota[r], col_quota[c]
    'icon1' (and the rest 'icon2').

    Constraints:
      1) Each cell is either icon1 or icon2 (or pre-filled).
      2) No more than 2 identical icons consecutively in any row/column.
      3) row_quota[r] icon1 in row r, col_quota[c] icon1 in col c.
      4) equals => same icon, cross => different icons.

    Returns:
      A 2D list solution grid if successful, otherwise [] if unsolvable.
    """
    from copy import deepcopy

    if initial_grid is None:
        grid = [[None] * cols for _ in range(rows)]
    else:
        grid = deepcopy(initial_grid)

    def backtrack(r, c):
        if c == cols:
            # End of row r => check if row r exactly meets row_quota
            # If not, no point continuing
            if not row_fits_quota(grid, r, row_quota[r]):
                return False
            # Move to next row, col=0
            return backtrack(r + 1, 0)

        if r == rows:
            # All rows are filled => final check for columns
            return columns_fit_quota(grid, col_quota) and True

        # If cell is pre-filled, just validate constraints
        if grid[r][c] in (ICON1, ICON2):
            if satisfies_constraints(grid, r, c,
                                     equals_constraints,
                                     cross_constraints,
                                     row_quota,
                                     col_quota):
                return backtrack(r, c + 1)
            else:
                return False

        # Otherwise, try icon1 or icon2
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

    if backtrack(0, 0):
        return grid
    return []

def row_fits_quota(grid, row_idx, needed_icon1):
    """
    Checks if row row_idx has exactly needed_icon1 'icon1'.
    If row is not fully assigned, it might still have None – in that case
    we treat that as no match. If you want to ensure no 'None' remain,
    the solver ensures no None if properly assigned.
    """
    row = grid[row_idx]
    actual_icon1 = sum(1 for cell in row if cell == ICON1)
    return (actual_icon1 == needed_icon1)

def columns_fit_quota(grid, col_quota):
    """
    After the entire grid is filled, checks each column c
    has exactly col_quota[c] 'icon1'. If any mismatch, return False.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for c in range(cols):
        icon1_count = 0
        for r in range(rows):
            if grid[r][c] == ICON1:
                icon1_count += 1
        if icon1_count != col_quota[c]:
            return False
    return True

def satisfies_constraints(grid, r, c,
                          equals_constraints,
                          cross_constraints,
                          row_quota,
                          col_quota):
    """
    Checks partial constraints so far:
      - row/col not exceeding icon1 quota
      - no triple consecutive icons in row/col
      - '=' => same, cross => differ if both assigned
    """
    val = grid[r][c]
    if val not in (ICON1, ICON2):
        return True  # not assigned

    rows = len(grid)
    cols = len(grid[0])

    # Row partial check: do not exceed row_quota
    icon1_in_row = sum(1 for x in grid[r] if x == ICON1)
    if icon1_in_row > row_quota[r]:
        return False

    # Column partial check: do not exceed col_quota
    icon1_in_col = sum(1 for rr in range(rows) if grid[rr][c] == ICON1)
    if icon1_in_col > col_quota[c]:
        return False

    # No triple consecutive horizontally
    if has_run_of_three(grid[r]):
        return False

    # No triple consecutive vertically
    col_cells = [grid[rr][c] for rr in range(rows)]
    if has_run_of_three(col_cells):
        return False

    # equals => same
    for (r1, c1), (r2, c2) in equals_constraints:
        if grid[r1][c1] is not None and grid[r2][c2] is not None:
            if grid[r1][c1] != grid[r2][c2]:
                return False

    # cross => differ
    for (r1, c1), (r2, c2) in cross_constraints:
        if grid[r1][c1] is not None and grid[r2][c2] is not None:
            if grid[r1][c1] == grid[r2][c2]:
                return False

    return True

def has_run_of_three(cells):
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