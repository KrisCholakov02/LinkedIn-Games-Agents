from time import sleep

from selenium.webdriver.common.by import By
from src.core.base_game_agent import BaseGameAgent

import cv2
from src.games.queens.recognizer import recognize_maze
from src.utils.screenshot import take_screenshot
from src.games.queens.solver import solve_queens
from src.games.queens.placer import place_queens_by_dom


class LinkedInQueensAgent(BaseGameAgent):
    """
    Agent for automating the LinkedIn 'Queens' puzzle game.
    Handles the full pipeline: login, board capture, recognition,
    solving, and placing queens on the board.
    """

    def __init__(self, driver_path: str, headless: bool = False):
        """
        Initializes the Queens agent and launches the browser.

        Args:
            driver_path (str): Path to the Chrome WebDriver.
            headless (bool): Whether to run the browser in headless mode.
        """
        super().__init__(driver_path, headless)
        self.launch_driver()
        self.num_cols = None  # Will be set after recognition

    def navigate_and_prepare(self, username: str, password: str) -> bool:
        """
        Logs in and navigates to the Queens game. Prepares the game if not already solved.

        Args:
            username (str): LinkedIn username.
            password (str): LinkedIn password.

        Returns:
            bool: True if the puzzle is ready to be solved, False otherwise.
        """
        self.login(username, password)
        self.navigate_to_game("Queens")
        sleep(1)
        return self._handle_game_state()

    def capture_board(self):
        """
        Captures a screenshot of the game board and loads it as an image.

        Returns:
            image (np.ndarray): The captured board image.

        Raises:
            RuntimeError: If the screenshot could not be loaded.
        """
        output_path = "img/screenshot_test.png"
        take_screenshot(self.driver, output_path, board_class="queens-board")
        image = cv2.imread(output_path)
        if image is None:
            raise RuntimeError(f"Failed to load image from {output_path}")
        print("[✓] Captured and loaded board screenshot.")
        return image

    def recognize(self, image):
        """
        Recognizes the puzzle structure and color clusters from the board image.

        Args:
            image (np.ndarray): The board screenshot.

        Returns:
            tuple: (maze_map, color_map) where:
                - maze_map (dict): Mapping of (row, col) positions to cluster IDs.
                - color_map (dict): Mapping of cluster IDs to their representative color.
        """
        maze_map, color_map = recognize_maze(image)
        self.num_cols = max(pos[1] for pos in maze_map.keys()) + 1
        return maze_map, color_map

    def solve(self, maze_map):
        """
        Solves the Queens puzzle by finding valid queen positions.

        Args:
            maze_map (dict): Mapping of cell positions to cluster IDs.

        Returns:
            list: List of (row, col) tuples representing queen positions.
        """
        num_clusters = len(set(maze_map.values()))
        queen_positions = solve_queens(maze_map, num_clusters)

        if not queen_positions:
            print("[✗] No valid solution found.")
        else:
            print(f"[✓] Found {len(queen_positions)} queen positions.")

        return queen_positions

    def place_queens(self, queen_positions):
        """
        Places queens on the board using DOM interaction.

        Args:
            queen_positions (list): List of (row, col) positions to place queens.

        Raises:
            ValueError: If num_cols has not been initialized.
        """
        if self.num_cols is None:
            raise ValueError("num_cols must be determined before placing queens.")

        place_queens_by_dom(self.driver, queen_positions, self.num_cols)
        print("[✓] Queens placed on board.")