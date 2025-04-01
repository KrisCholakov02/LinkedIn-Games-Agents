import os
from time import sleep

import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.core.base_game_agent import BaseGameAgent
from src.games.zip.solver import solve_zip_puzzle
from src.games.zip.placer import place_solution as place_zip_solution
from src.games.zip.recognizer import recognize_zip_board
from src.utils.screenshot import take_screenshot
import cv2


class LinkedInZipAgent(BaseGameAgent):
    def __init__(self, driver_path, headless=False):
        super().__init__(driver_path, headless)
        self.launch_driver()
        self.num_cols = None

    def navigate_and_prepare(self, username: str, password: str) -> bool:
        self.login(username, password)
        self.navigate_to_game("Zip")
        sleep(1)
        return self._handle_game_state()

    def capture_board(self):
        try:
            main_clear_button = self.driver.find_element(By.ID, "aux-controls-clear")
            main_clear_button.click()
            sleep(0.1)
            print("[i] Clicked the main 'Clear' button before capturing board.")
        except Exception as e:
            print(f"[!] Could not find/click main Clear button: {e}")

        try:
            wait = WebDriverWait(self.driver, 5)
            confirm_button = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//button[contains(@class,'artdeco-modal__confirm-dialog-btn') and .//span[text()='Clear']]"
                ))
            )
            confirm_button.click()
            sleep(0.1)
            print("[i] Confirmed 'Clear' in the modal before capturing board.")
        except Exception as e:
            print(f"[!] Could not find/click confirm dialog's Clear button: {e}")

        output_path = "img/zip_screenshot.png"
        take_screenshot(self.driver, output_path, board_class="trail-board")
        image = cv2.imread(output_path)
        if image is None:
            raise RuntimeError(f"Failed to load image from {output_path}")
        print("[✓] Captured and loaded Zip board screenshot.")
        return image

    def recognize(self, image):
        """
        Recognizes the Zip board by detecting the grid, reading digits with OCR,
        and detecting walls between cells.

        Returns:
            tuple: (cell_map, walls_map, grid_size)
              - cell_map: dict mapping (row, col) to recognized content (digit as string or "empty")
              - walls_map: dict mapping adjacent cell pairs to a boolean (True if wall present)
              - grid_size: (num_rows, num_cols)
        """
        # Get cell map, walls map, and grid size from the recognition pipeline.
        cell_map, walls_map, grid_size = recognize_zip_board(image, debug=True)
        self.num_rows, self.num_cols = grid_size
        print(f"[✓] Recognized board with {self.num_rows} rows × {self.num_cols} cols.")

        # Print recognized cell values in a grid format.
        for r in range(self.num_rows):
            row_str = ""
            for c in range(self.num_cols):
                row_str += f"{cell_map.get((r, c), 'empty')} "
            print(row_str)

        # --- Visualize Walls ---
        # Recompute grid lines based on the image dimensions and grid size.
        h_lines = list(np.linspace(0, image.shape[0], self.num_rows + 1, dtype=int))
        v_lines = list(np.linspace(0, image.shape[1], self.num_cols + 1, dtype=int))

        # Create a copy of the board image for overlaying wall markers.
        vis_img = image.copy()

        # Iterate over walls_map: keys are tuples ((r,c), (r2,c2)).
        # Vertical wall: between cells in the same row, adjacent columns.
        # Horizontal wall: between cells in the same column, adjacent rows.
        for ((r, c), (r2, c2)), is_wall in walls_map.items():
            if is_wall:
                if r == r2 and c2 == c + 1:
                    # Vertical wall between cell (r,c) and (r, c+1)
                    x = v_lines[c2]  # Boundary line between these two cells.
                    y1 = h_lines[r]
                    y2 = h_lines[r + 1]
                    cv2.line(vis_img, (x, y1), (x, y2), (0, 0, 255), 2)
                elif c == c2 and r2 == r + 1:
                    # Horizontal wall between cell (r,c) and (r+1, c)
                    y = h_lines[r2]
                    x1 = v_lines[c]
                    x2 = v_lines[c + 1]
                    cv2.line(vis_img, (x1, y), (x2, y), (0, 0, 255), 2)

        # Save the visualization for debugging.
        debug_vis_path = "img/debug/zip_walls_visualization.png"
        cv2.imwrite(debug_vis_path, vis_img)
        print(f"Saved walls visualization to '{debug_vis_path}'.")

        return cell_map, walls_map, grid_size

    def solve(self, recognized_data):
        """
        Solves the Zip puzzle by finding a Hamiltonian path that visits every cell exactly once,
        starting at the cell labeled "1" and ending at the cell with the highest number.
        The path must obey wall constraints (no movement through a wall) and the numerical order
        in pre-filled cells must be strictly increasing.

        recognized_data: tuple (cell_map, walls_map, grid_size) where:
          - cell_map: dict mapping (row, col) -> recognized content ("empty" or a digit as string)
          - walls_map: dict mapping adjacent cell pairs to a boolean (True if wall exists)
          - grid_size: (num_rows, num_cols)

        After solving, the method visualizes the solution path on the board screenshot by drawing
        connecting lines between the centers of the corresponding cells and saves the debug image.
        """
        cell_map, walls_map, grid_size = recognized_data
        solution = solve_zip_puzzle(cell_map, walls_map, grid_size)

        if not solution:
            print("[!] No solution found, which should be unlikely for a valid puzzle.")
            return solution

        print("[✓] Puzzle solved!")
        # Load the board image for visualization.
        board_img = cv2.imread("img/zip_screenshot.png")
        if board_img is None:
            print("[!] Could not load board image for visualization.")
            return solution

        num_rows, num_cols = grid_size
        h, w = board_img.shape[:2]
        # Compute grid boundaries and cell centers.
        h_lines = np.linspace(0, h, num_rows + 1, dtype=int)
        v_lines = np.linspace(0, w, num_cols + 1, dtype=int)
        cell_centers = {}
        for r in range(num_rows):
            for c in range(num_cols):
                center_x = (v_lines[c] + v_lines[c + 1]) // 2
                center_y = (h_lines[r] + h_lines[r + 1]) // 2
                cell_centers[(r, c)] = (center_x, center_y)

        # Draw the solution path as red lines connecting cell centers.
        for i in range(len(solution) - 1):
            pt1 = cell_centers[solution[i]]
            pt2 = cell_centers[solution[i + 1]]
            cv2.line(board_img, pt1, pt2, (0, 0, 255), 2)

        # Save the visualization.
        vis_path = "img/debug/zip_solution.png"
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, board_img)
        print(f"Saved solution visualization to '{vis_path}'.")

        return solution

    def place_solution(self, solution):
        if not solution:
            print("[!] No solution to place.")
            return

        print("[i] Placing the puzzle solution on the board...")
        try:
            place_zip_solution(self.driver, solution, self.num_cols, debug=True)
            print("[✓] Placed the puzzle solution successfully.")
            sleep(1000)
        except RuntimeError as e:
            print(f"[✗] Placing solution failed: {e}")