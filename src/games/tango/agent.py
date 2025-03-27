from time import sleep
from src.core.base_game_agent import BaseGameAgent
from src.games.tango.solver import solve_tango_puzzle
from src.games.tango.placer import place_solution as place_tango_solution

from src.games.tango.recognizer import recognize_tango_board
from src.utils.screenshot import take_screenshot
import cv2


class LinkedInTangoAgent(BaseGameAgent):
    def __init__(self, driver_path, headless=False):
        super().__init__(driver_path, headless)
        self.launch_driver()
        self.num_cols = None

    def navigate_and_prepare(self, username: str, password: str) -> bool:
        self.login(username, password)
        self.navigate_to_game("Tango")
        sleep(1)
        return self._handle_game_state()

    def capture_board(self):
        output_path = "img/tango_screenshot.png"
        take_screenshot(self.driver, output_path, board_class="lotka-board")
        image = cv2.imread(output_path)
        if image is None:
            raise RuntimeError(f"Failed to load image from {output_path}")
        print("[✓] Captured and loaded Tango board screenshot.")
        return image

    def recognize(self, image):
        # Now returns (cell_map, sign_map, rows, cols)
        cell_map, sign_map, rows, cols = recognize_tango_board(image, debug=True)
        self.num_cols = cols
        self.num_rows = rows

        print(f"[✓] Recognized board with {rows} rows × {cols} columns.")
        print(f"[✓] Found {len(set(cell_map.values()))} cell clusters and {len(sign_map)} sign(s).")
        print(f"[✓] Cell map: {cell_map}")
        print(f"[✓] Sign map: {sign_map}")

        # Return the cell map and the sign map instead of an edge map
        return cell_map, sign_map

    def solve(self, recognized_data):
        """
        Solve the Tango puzzle using the external solver from solver.py
        recognized_data => (cell_map, sign_map)
        """
        cell_map, sign_map = recognized_data
        rows = self.num_rows
        cols = self.num_cols

        # Build equals_constraints & cross_constraints from sign_map
        equals_constraints = []
        cross_constraints = []
        # sign_map items might store pairs of adjacent cells that sign covers
        # e.g. sign_item['cell_pairs'] = [((r1, c1), (r2, c2))]
        # sign_item['sign_label'] => 'equals' or 'cross'
        for item in sign_map:
            label = item['sign_label']
            if 'cell_pairs' in item:  # or however you store it
                for pair in item['cell_pairs']:
                    if label == 'equals':
                        equals_constraints.append(pair)
                    elif label == 'cross':
                        cross_constraints.append(pair)

        # Each row/column must have the same # of icon1 & icon2 => half icon1, half icon2
        # For an NxN puzzle with N even => row_quota[r] = N//2
        row_quota = [cols // 2] * rows
        col_quota = [rows // 2] * cols

        # Call the puzzle solver
        solution = solve_tango_puzzle(rows, cols,
                                      equals_constraints,
                                      cross_constraints,
                                      row_quota,
                                      col_quota)

        if not solution:
            print("[!] No solution found, which should be unlikely for a valid puzzle.")
        else:
            print("[✓] Puzzle solved!")
            for row_sol in solution:
                print(row_sol)
        return solution


    def place_solution(self, solution):
        """
        Places the puzzle solution on the board by clicking each cell
        until it shows the correct icon.
        """
        if not solution:
            print("[!] No solution to place.")
            return

        print("[i] Placing the puzzle solution on the board...")
        try:
            place_tango_solution(
                driver=self.driver,
                solution_grid=solution,
                skip_locked=True,
                debug=True   # set to False for less logging
            )
            print("[✓] Placed the puzzle solution successfully.")
            sleep(1000)
        except RuntimeError as e:
            print(f"[✗] Placing solution failed: {e}")