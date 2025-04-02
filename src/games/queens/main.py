from time import sleep

from src.games.queens.agent import LinkedInQueensAgent
from config import DRIVER_PATH


def run_game(username: str, password: str):
    """
    Entry point for running the LinkedIn 'Queens' puzzle automation.

    This function:
        - Logs into LinkedIn.
        - Navigates to the Queens game.
        - Captures and analyzes the game board.
        - Solves the puzzle.
        - Places the queens on the board.

    Args:
        username (str): LinkedIn username.
        password (str): LinkedIn password.
    """
    agent = LinkedInQueensAgent(driver_path=DRIVER_PATH)

    if agent.navigate_and_prepare(username=username, password=password):
        screenshot = agent.capture_board()
        maze_map, color_map = agent.recognize(screenshot)
        queen_positions = agent.solve(maze_map)
        agent.place_queens(queen_positions)
        sleep(10)  # Give time to visually verify the result

    agent.quit()