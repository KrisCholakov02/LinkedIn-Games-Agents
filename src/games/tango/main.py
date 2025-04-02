from src.games.tango.agent import LinkedInTangoAgent
from config import DRIVER_PATH


def run_game(username: str, password: str):
    """
    Entry point for solving the LinkedIn 'Tango' puzzle.

    This function:
        - Logs into LinkedIn.
        - Navigates to the Tango game.
        - Clears and captures the board.
        - Recognizes the board state and constraints.
        - Solves the puzzle.
        - Places the solution on the board.

    Args:
        username (str): LinkedIn username.
        password (str): LinkedIn password.
    """
    agent = LinkedInTangoAgent(driver_path=DRIVER_PATH)

    if agent.navigate_and_prepare(username, password):
        board_img = agent.capture_board()
        game_map = agent.recognize(board_img)
        solution = agent.solve(game_map)
        agent.place_solution(solution)

    agent.quit()