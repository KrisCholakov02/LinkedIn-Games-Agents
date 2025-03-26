from time import sleep

from src.core.base_game_agent import BaseGameAgent
from selenium.webdriver.common.by import By

import cv2
from src.games.queens.recognizer import recognize_maze
from src.utils.screenshot import take_screenshot
from src.games.queens.solver import solve_queens
from src.games.queens.placer import place_queens_by_dom


class LinkedInQueensAgent(BaseGameAgent):
    def __init__(self, driver_path, headless=False):
        super().__init__(driver_path, headless)
        self.launch_driver()
        self.num_cols = None  # to be set during recognition

    def navigate_and_prepare(self, username: str, password: str) -> bool:
        self.login(username, password)
        self.navigate_to_game("Queens")
        sleep(2)
        return self._handle_game_state()

    def capture_board(self):
        output_path = "img/screenshot_test.png"
        take_screenshot(self.driver, output_path, board_class="queens-board")
        image = cv2.imread(output_path)
        if image is None:
            raise RuntimeError(f"Failed to load image from {output_path}")
        print("[✓] Captured and loaded board screenshot.")
        return image

    def recognize(self, image):
        maze_map, color_map = recognize_maze(image)
        self.num_cols = max(pos[1] for pos in maze_map.keys()) + 1
        return maze_map, color_map

    def solve(self, maze_map):
        num_clusters = len(set(maze_map.values()))
        queen_positions = solve_queens(maze_map, num_clusters)
        if not queen_positions:
            print("[✗] No valid solution found.")
        else:
            print(f"[✓] Found {len(queen_positions)} queen positions.")
        return queen_positions

    def place_queens(self, queen_positions):
        if self.num_cols is None:
            raise ValueError("num_cols must be determined before placing queens.")
        place_queens_by_dom(self.driver, queen_positions, self.num_cols)
        print("[✓] Queens placed on board.")