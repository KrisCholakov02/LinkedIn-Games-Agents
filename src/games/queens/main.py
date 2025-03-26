from time import sleep

from src.games.queens.agent import LinkedInQueensAgent
from config import DRIVER_PATH

def run_game(username: str, password: str):
    agent = LinkedInQueensAgent(driver_path=DRIVER_PATH)
    if agent.navigate_and_prepare(username=username, password=password):
        screenshot = agent.capture_board()
        maze_map, color_map = agent.recognize(screenshot)
        queen_positions = agent.solve(maze_map)
        agent.place_queens(queen_positions)
        sleep(10)
    agent.quit()