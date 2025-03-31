from time import sleep

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
        """
        1) Finds and clicks the main 'Clear' button (id='aux-controls-clear').
        2) Waits for the confirm modal; finds the 'Clear' button inside that modal
           via the <span> text='Clear', and clicks it.
        3) Takes a screenshot of the board (class='lotka-board'), saves to
           'img/tango_screenshot.png', then loads and returns the cv2 image.
        """
        import time
        import cv2
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        # First Clear Button
        try:
            main_clear_button = self.driver.find_element(By.ID, "aux-controls-clear")
            main_clear_button.click()
            time.sleep(0.1)
            print("[i] Clicked the main 'Clear' button before capturing board.")
        except Exception as e:
            print(f"[!] Could not find/click main Clear button: {e}")

        # Second Clear Button (Confirm in Modal)
        try:
            wait = WebDriverWait(self.driver, 5)
            confirm_button = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//button[contains(@class,'artdeco-modal__confirm-dialog-btn') and .//span[text()='Clear']]"
                ))
            )
            confirm_button.click()
            time.sleep(0.1)
            print("[i] Confirmed 'Clear' in the modal before capturing board.")
        except Exception as e:
            print(f"[!] Could not find/click confirm dialog's Clear button: {e}")

        # Now do the actual screenshot of the board
        output_path = "img/tango_screenshot.png"
        take_screenshot(self.driver, output_path, board_class="lotka-board")

        # Load with cv2
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
        # Print the cell map in a grid-like format
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                row_str += f"{cell_map[(r, c)]} "
            print(row_str)
        # Print the sign map
        for sign_item in sign_map:
            print(sign_item)

        # Return the cell map and the sign map instead of an edge map
        return cell_map, sign_map

    def solve(self, recognized_data):
        """
        Solve the Tango puzzle, respecting pre-filled icons in cell_map.
        recognized_data => (cell_map, sign_map)

        cell_map => {(r,c): "empty"/"icon1"/"icon2"}
        sign_map => list of dicts, each with:
          {
            'sign_label': 'equals'/'cross',
            'cell_pairs': [((r1,c1),(r2,c2))]
          }
        """
        cell_map, sign_map = recognized_data
        rows = self.num_rows
        cols = self.num_cols

        equals_constraints = []
        cross_constraints = []

        for item in sign_map:
            label = item['sign_label']
            if 'cell_pairs' in item:
                for (r1, c1), (r2, c2) in item['cell_pairs']:
                    # skip invalid or out-of-range
                    if (r1 is None or c1 is None or r2 is None or c2 is None or
                            r1 < 0 or r1 >= rows or c1 < 0 or c1 >= cols or
                            r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols):
                        continue

                    if label == 'equals':
                        equals_constraints.append(((r1, c1), (r2, c2)))
                    elif label == 'cross':
                        cross_constraints.append(((r1, c1), (r2, c2)))

        # For NxN with N even => half icon1, half icon2
        row_quota = [cols // 2] * rows
        col_quota = [rows // 2] * cols

        # Build initial_grid, filling pre-filled cells from cell_map
        initial_grid = []
        for r in range(rows):
            row_list = []
            for c in range(cols):
                val = cell_map.get((r, c), "empty")
                if val == "icon1" or val == "icon2":
                    row_list.append(val)  # locked cell
                else:
                    row_list.append(None)  # solver can fill
            initial_grid.append(row_list)

        solution = solve_tango_puzzle(rows, cols,
                                      equals_constraints,
                                      cross_constraints,
                                      row_quota,
                                      col_quota,
                                      initial_grid)

        if not solution:
            print("[!] No solution found (puzzle might be invalid).")
            return []
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