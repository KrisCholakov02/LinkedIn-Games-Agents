from src.games.zip.agent import LinkedInZipAgent
from config import DRIVER_PATH


def run_game(username: str, password: str):
    """
    Entry point for running the Zip puzzle automation.

    Steps:
        1. Launches browser and logs in.
        2. Navigates to the Zip game and resets the board.
        3. Captures the board and runs OCR + wall detection.
        4. Solves the puzzle using pathfinding logic.
        5. Places the solution on the board via browser interactions.
    """
    agent = LinkedInZipAgent(driver_path=DRIVER_PATH)

    if agent.navigate_and_prepare(username, password):
        board_img = agent.capture_board()
        game_data = agent.recognize(board_img)
        solution = agent.solve(game_data)
        agent.place_solution(solution)

    agent.quit()