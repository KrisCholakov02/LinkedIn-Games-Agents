import os
from time import sleep

import numpy as np
import cv2
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.core.base_game_agent import BaseGameAgent
from src.games.zip.solver import solve_zip_puzzle
from src.games.zip.placer import place_solution as place_zip_solution
from src.games.zip.recognizer import recognize_zip_board
from src.utils.screenshot import take_screenshot


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
        """
        Clears the current puzzle state and captures a screenshot of the board.
        Returns:
            image (np.ndarray): Loaded screenshot as a BGR image.
        """
        try:
            clear_btn = self.driver.find_element(By.ID, "aux-controls-clear")
            clear_btn.click()
            sleep(0.1)
            print("[i] Clicked the main 'Clear' button.")
        except Exception as e:
            print(f"[!] Failed to click 'Clear' button: {e}")

        try:
            wait = WebDriverWait(self.driver, 5)
            confirm_btn = wait.until(EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(@class,'artdeco-modal__confirm-dialog-btn') and .//span[text()='Clear']]"
            )))
            confirm_btn.click()
            sleep(0.1)
            print("[i] Confirmed reset in modal dialog.")
        except Exception as e:
            print(f"[!] Failed to confirm 'Clear' action: {e}")

        output_path = "img/zip_screenshot.png"
        take_screenshot(self.driver, output_path, board_class="trail-board")
        image = cv2.imread(output_path)
        if image is None:
            raise RuntimeError(f"Failed to load screenshot from '{output_path}'.")

        print("[✓] Board screenshot captured successfully.")
        return image

    def recognize(self, image):
        """
        Runs recognition on the Zip puzzle board.

        Returns:
            tuple:
              - cell_map: {(row, col): str}, either a digit or 'empty'
              - walls_map: {((r1, c1), (r2, c2)): bool}, True if wall blocks path
              - grid_size: (rows, cols)
        """
        cell_map, walls_map, grid_size = recognize_zip_board(image, debug=True)
        self.num_rows, self.num_cols = grid_size
        print(f"[✓] Recognized {self.num_rows} rows × {self.num_cols} columns.")

        # Print detected digits for manual verification
        for r in range(self.num_rows):
            print(" ".join(cell_map.get((r, c), "empty") for c in range(self.num_cols)))

        # Overlay wall visualization
        h_lines = np.linspace(0, image.shape[0], self.num_rows + 1, dtype=int)
        v_lines = np.linspace(0, image.shape[1], self.num_cols + 1, dtype=int)
        vis_img = image.copy()

        for ((r1, c1), (r2, c2)), is_wall in walls_map.items():
            if not is_wall:
                continue
            if r1 == r2 and c2 == c1 + 1:
                x = v_lines[c2]
                y1, y2 = h_lines[r1], h_lines[r1 + 1]
                cv2.line(vis_img, (x, y1), (x, y2), (0, 0, 255), 2)
            elif c1 == c2 and r2 == r1 + 1:
                y = h_lines[r2]
                x1, x2 = v_lines[c1], v_lines[c1 + 1]
                cv2.line(vis_img, (x1, y), (x2, y), (0, 0, 255), 2)

        os.makedirs("img/debug", exist_ok=True)
        debug_path = "img/debug/zip_walls_visualization.png"
        cv2.imwrite(debug_path, vis_img)
        print(f"Saved wall overlay to '{debug_path}'.")

        return cell_map, walls_map, grid_size

    def solve(self, recognized_data):
        """
        Solves the Zip puzzle by finding a valid Hamiltonian path.

        The path must:
            - Begin at cell labeled '1' and end at the highest number.
            - Visit all cells exactly once.
            - Avoid crossing any wall.
            - Maintain increasing order across any pre-filled digits.

        Returns:
            list: Ordered list of (row, col) cell positions forming the solution path.
        """
        cell_map, walls_map, grid_size = recognized_data
        solution = solve_zip_puzzle(cell_map, walls_map, grid_size)

        if not solution:
            print("[!] No valid solution found.")
            return []

        print("[✓] Puzzle solved successfully.")

        board_img = cv2.imread("img/zip_screenshot.png")
        if board_img is None:
            print("[!] Failed to load screenshot for path visualization.")
            return solution

        num_rows, num_cols = grid_size
        h, w = board_img.shape[:2]
        h_lines = np.linspace(0, h, num_rows + 1, dtype=int)
        v_lines = np.linspace(0, w, num_cols + 1, dtype=int)

        cell_centers = {
            (r, c): (
                (v_lines[c] + v_lines[c + 1]) // 2,
                (h_lines[r] + h_lines[r + 1]) // 2
            )
            for r in range(num_rows)
            for c in range(num_cols)
        }

        for i in range(len(solution) - 1):
            pt1 = cell_centers[solution[i]]
            pt2 = cell_centers[solution[i + 1]]
            cv2.line(board_img, pt1, pt2, (0, 0, 255), 2)

        vis_path = "img/debug/zip_solution.png"
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, board_img)
        print(f"Saved solution path visualization to '{vis_path}'.")

        return solution

    def place_solution(self, solution):
        """
        Places the solved path onto the board by simulating clicks.

        Args:
            solution (list): Ordered list of (row, col) positions in the solution.
        """
        if not solution:
            print("[!] No solution available to place.")
            return

        print("[i] Placing solution onto the board...")
        try:
            place_zip_solution(self.driver, solution, self.num_cols, debug=True)
            print("[✓] Solution placed successfully.")
            sleep(1000)
        except RuntimeError as e:
            print(f"[✗] Error while placing solution: {e}")