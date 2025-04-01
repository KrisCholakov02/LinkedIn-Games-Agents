from time import sleep

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.core.base_game_agent import BaseGameAgent
# from src.games.zip.solver import solve_zip_puzzle
# from src.games.zip.placer import place_solution as place_zip_solution
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
        Recognizes the Zip board by detecting the grid and using OCR to read digits.

        Returns:
            tuple: (cell_map, grid_size)
              - cell_map: dict mapping (row, col) to recognized content (digit as string or "empty")
              - grid_size: (num_rows, num_cols)
        """
        cell_map, grid_size = recognize_zip_board(image, debug=True)
        self.num_rows, self.num_cols = grid_size
        print(f"[✓] Recognized board with {self.num_rows} rows × {self.num_cols} cols.")
        for r in range(self.num_rows):
            row_str = ""
            for c in range(self.num_cols):
                row_str += f"{cell_map.get((r, c), 'empty')} "
            print(row_str)
        return cell_map, grid_size

    def solve(self, recognized_data):
        # TODO: Replace with actual solver logic
        # solution = solve_zip_puzzle(recognized_data)
        # return solution
        raise NotImplementedError("Solver for Zip is not implemented yet.")

    def place_solution(self, solution):
        if not solution:
            print("[!] No solution to place.")
            return

        print("[i] Placing the puzzle solution on the board...")
        try:
            # place_zip_solution(self.driver, solution, skip_locked=True, debug=True)
            print("[✓] Placed the puzzle solution successfully.")
            sleep(1000)
        except RuntimeError as e:
            print(f"[✗] Placing solution failed: {e}")