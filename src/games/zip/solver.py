def solve_zip_puzzle(cell_map, walls_map, grid_size):
    """
    Solves the Zip puzzle via backtracking.

    The puzzle is represented by:
      - cell_map: dict mapping (row, col) -> content, where content is either "empty" or a digit as a string.
      - walls_map: dict mapping adjacent cell pairs (e.g. ((r,c),(r, c+1)) for vertical, ((r,c),(r+1,c)) for horizontal)
                   to a boolean (True if a wall exists, disallowing movement).
      - grid_size: tuple (num_rows, num_cols)

    The goal is to find a Hamiltonian path that visits every cell exactly once, starting at the cell
    labeled "1" and ending at the cell with the highest number. Moreover, when the path passes through
    pre-filled (numbered) cells, the digits must appear in strictly increasing order.

    Returns:
        list: A list of (row, col) positions representing the valid path, or [] if no solution exists.
    """
    num_rows, num_cols = grid_size
    total_cells = num_rows * num_cols

    # Identify start and end cells from the pre-filled numbers.
    start = None
    end = None
    max_num = -1
    for pos, content in cell_map.items():
        if content != "empty":
            try:
                num = int(content)
            except ValueError:
                continue
            if num == 1:
                start = pos
            if num > max_num:
                max_num = num
                end = pos
    if start is None or end is None:
        print("Error: Start (cell with 1) or end (cell with highest number) is missing.")
        return []

    # Allowed moves: up, down, left, right.
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_move_allowed(cur, nxt):
        # Check boundaries.
        r, c = nxt
        if r < 0 or r >= num_rows or c < 0 or c >= num_cols:
            return False
        # Disallow move if a wall exists between cur and nxt.
        if (((cur, nxt) in walls_map and walls_map[(cur, nxt)]) or
            ((nxt, cur) in walls_map and walls_map[(nxt, cur)])):
            return False
        return True

    best_solution = None

    def dfs(cur, path, last_number):
        nonlocal best_solution
        # If a complete path is found, check that it ends at the correct cell and last number equals max_num.
        if len(path) == total_cells:
            if cur == end and last_number == max_num:
                best_solution = path.copy()
            return

        for dr, dc in moves:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in path:
                continue
            if not is_move_allowed(cur, nxt):
                continue
            # Enforce number ordering if the next cell is pre-filled.
            nxt_content = cell_map.get(nxt, "empty")
            new_last = last_number
            if nxt_content != "empty":
                try:
                    nxt_num = int(nxt_content)
                except ValueError:
                    continue
                if nxt_num <= last_number:
                    continue
                new_last = nxt_num
            path.append(nxt)
            dfs(nxt, path, new_last)
            if best_solution is not None:
                return
            path.pop()

    dfs(start, [start], 1)
    return best_solution if best_solution is not None else []