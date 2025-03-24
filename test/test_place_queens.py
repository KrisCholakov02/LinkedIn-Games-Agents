from time import sleep

from src.comp_vision.recognize import detect_grid_size_from_lines
from test_solve import test_solve
from src.web_automation.place_queens import place_queens_by_dom

def test_place_queens():
    agent, queen_positions = test_solve()

    # Call placement
    place_queens_by_dom(agent.driver, queen_positions, num_cols=max(max(pos) for pos in queen_positions) + 1)

    print("Queens placed successfully!")

    return agent

if __name__ == '__main__':
    agent = test_place_queens()
    sleep(20)
    agent.close()