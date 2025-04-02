def solve_zip_puzzle(cell_map, walls_map, grid_size):
    """
    Solves the Zip puzzle using depth-first search (DFS) with backtracking.

    Rules:
        - Each cell must be visited exactly once (Hamiltonian path).
        - The path must begin at the cell labeled "1" and end at the cell with the highest number.
        - Movement between adjacent cells is allowed unless blocked by a wall.
        - If a cell contains a digit, it must be visited in strictly increasing numeric order.

    Args:
        cell_map (dict): Maps (row, col) to either a digit (str) or "empty".
        walls_map (dict): Maps ((r1, c1), (r2, c2)) → bool indicating wall presence (True = wall).
        grid_size (tuple): (num_rows, num_cols)

    Returns:
        list[(row, col)]: Ordered list of cell positions representing the valid path, or [] if unsolvable.
    """
    num_rows, num_cols = grid_size
    total_cells = num_rows * num_cols

    # Locate starting cell ("1") and ending cell (maximum number)
    start = None
    end = None
    max_num = -1

    for pos, value in cell_map.items():
        if value == "empty":
            continue
        try:
            num = int(value)
        except ValueError:
            continue
        if num == 1:
            start = pos
        if num > max_num:
            max_num = num
            end = pos

    if start is None or end is None:
        print("Error: Missing required start or end cell (with numbers).")
        return []

    # Legal movements: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_move_allowed(current, neighbor):
        r, c = neighbor
        if not (0 <= r < num_rows and 0 <= c < num_cols):
            return False
        if walls_map.get((current, neighbor), False) or walls_map.get((neighbor, current), False):
            return False
        return True

    best_solution = None

    def dfs(current, path, last_num):
        nonlocal best_solution

        if len(path) == total_cells:
            if current == end and last_num == max_num:
                best_solution = path.copy()
            return

        for dr, dc in moves:
            neighbor = (current[0] + dr, current[1] + dc)
            if neighbor in path:
                continue
            if not is_move_allowed(current, neighbor):
                continue

            content = cell_map.get(neighbor, "empty")
            updated_last = last_num

            if content != "empty":
                try:
                    number = int(content)
                except ValueError:
                    continue
                if number <= last_num:
                    continue
                updated_last = number

            path.append(neighbor)
            dfs(neighbor, path, updated_last)
            if best_solution is not None:
                return
            path.pop()

    dfs(start, [start], 1)
    return best_solution if best_solution is not None else []