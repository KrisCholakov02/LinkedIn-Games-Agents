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
    """
    Agent for automating the LinkedIn 'Tango' logic puzzle.
    Handles login, board clearing, screenshot capture, recognition, solving, and interaction.
    """

    def __init__(self, driver_path: str, headless: bool = False):
        """
        Initializes the Tango agent and launches the browser.

        Args:
            driver_path (str): Path to the Chrome WebDriver.
            headless (bool): Whether to run the browser in headless mode.
        """
        super().__init__(driver_path, headless)
        self.launch_driver()
        self.num_cols = None
        self.num_rows = None

    def navigate_and_prepare(self, username: str, password: str) -> bool:
        """
        Logs into LinkedIn and navigates to the Tango game.

        Args:
            username (str): LinkedIn username.
            password (str): LinkedIn password.

        Returns:
            bool: True if puzzle is ready to be solved, False otherwise.
        """
        self.login(username, password)
        self.navigate_to_game("Tango")
        sleep(1)
        return self._handle_game_state()

    def capture_board(self):
        """
        Clears the board and captures a screenshot of the Tango puzzle.

        Returns:
            image (np.ndarray): The captured puzzle board image.

        Raises:
            RuntimeError: If the screenshot could not be loaded.
        """
        # Click the main "Clear" button
        try:
            main_clear_button = self.driver.find_element(By.ID, "aux-controls-clear")
            main_clear_button.click()
            sleep(0.1)
            print("[i] Clicked the main 'Clear' button.")
        except Exception as e:
            print(f"[!] Failed to click main Clear button: {e}")

        # Confirm "Clear" in the modal
        try:
            wait = WebDriverWait(self.driver, 5)
            confirm_button = wait.until(EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(@class,'artdeco-modal__confirm-dialog-btn') and .//span[text()='Clear']]"
            )))
            confirm_button.click()
            sleep(0.1)
            print("[i] Confirmed 'Clear' in the modal.")
        except Exception as e:
            print(f"[!] Failed to confirm Clear in modal: {e}")

        # Take screenshot
        output_path = "img/tango_screenshot.png"
        take_screenshot(self.driver, output_path, board_class="lotka-board")
        image = cv2.imread(output_path)

        if image is None:
            raise RuntimeError(f"Failed to load image from {output_path}")

        print("[✓] Captured and loaded Tango board screenshot.")
        return image

    def recognize(self, image):
        """
        Recognizes the current state of the board from the screenshot.

        Args:
            image (np.ndarray): Screenshot of the puzzle board.

        Returns:
            tuple:
                - cell_map (dict): {(row, col): "empty"/"icon1"/"icon2"}
                - sign_map (list): List of constraint objects between cell pairs.
        """
        cell_map, sign_map, rows, cols = recognize_tango_board(image, debug=True)
        self.num_rows = rows
        self.num_cols = cols

        print(f"[✓] Recognized board with {rows} rows × {cols} columns.")
        print(f"[✓] Found {len(set(cell_map.values()))} cell clusters and {len(sign_map)} sign(s).")

        # Print the cell map visually
        for r in range(rows):
            row_str = " ".join(str(cell_map[(r, c)]) for c in range(cols))
            print(row_str)

        for sign in sign_map:
            print(sign)

        return cell_map, sign_map

    def solve(self, recognized_data):
        """
        Solves the Tango puzzle based on recognized board state and constraints.

        Args:
            recognized_data (tuple): (cell_map, sign_map)

        Returns:
            list: Solved grid (2D list of "icon1"/"icon2") or [] if no solution found.
        """
        cell_map, sign_map = recognized_data
        rows, cols = self.num_rows, self.num_cols

        equals_constraints = []
        cross_constraints = []

        for item in sign_map:
            label = item['sign_label']
            if 'cell_pairs' in item:
                for (r1, c1), (r2, c2) in item['cell_pairs']:
                    if (None in [r1, c1, r2, c2] or
                            r1 < 0 or r1 >= rows or c1 < 0 or c1 >= cols or
                            r2 < 0 or r2 >= rows or c2 < 0 or c2 >= cols):
                        continue
                    if label == 'equals':
                        equals_constraints.append(((r1, c1), (r2, c2)))
                    elif label == 'cross':
                        cross_constraints.append(((r1, c1), (r2, c2)))

        row_quota = [cols // 2] * rows
        col_quota = [rows // 2] * cols

        # Build initial grid
        initial_grid = [
            [
                cell_map.get((r, c)) if cell_map.get((r, c)) in {"icon1", "icon2"} else None
                for c in range(cols)
            ]
            for r in range(rows)
        ]

        solution = solve_tango_puzzle(
            rows, cols,
            equals_constraints,
            cross_constraints,
            row_quota,
            col_quota,
            initial_grid
        )

        if not solution:
            print("[!] No solution found (puzzle may be invalid).")
            return []
        else:
            print("[✓] Puzzle solved:")
            for row in solution:
                print(row)
            return solution

    def place_solution(self, solution):
        """
        Places the solved solution onto the board using DOM interaction.

        Args:
            solution (list): 2D list of "icon1"/"icon2" strings.
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
                debug=True  # Set to False to reduce console output
            )
            print("[✓] Placed the puzzle solution successfully.")
            sleep(10)
        except RuntimeError as e:
            print(f"[✗] Failed to place solution: {e}")