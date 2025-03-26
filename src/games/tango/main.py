from src.games.tango.agent import LinkedInTangoAgent
from config import DRIVER_PATH

def run_game(username: str, password: str):
    agent = LinkedInTangoAgent(driver_path=DRIVER_PATH)
    if agent.navigate_and_prepare(username, password):
        board_img = agent.capture_board()
        game_map = agent.recognize(board_img)
        solution = agent.solve(game_map)
        agent.place_solution(solution)
    agent.quit()