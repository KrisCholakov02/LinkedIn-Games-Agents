from time import sleep
from src.core.base_game_agent import BaseGameAgent
from selenium.webdriver.common.by import By

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
        cell_map, rows, cols = recognize_tango_board(image, debug=True)
        self.num_cols = cols
        self.num_rows = rows
        print(f"[✓] Recognized board with {rows} rows × {cols} columns.")
        print(f"[✓] Found {len(set(cell_map.values()))} cell clusters.")
        return cell_map

    def solve(self, maze_map):
        # To be implemented: placeholder
        raise NotImplementedError("Solver for Tango is not implemented yet.")

    def place_solution(self, solution):
        # To be implemented: placeholder
        raise NotImplementedError("Placer for Tango is not implemented yet.")