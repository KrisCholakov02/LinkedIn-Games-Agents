from config import DRIVER_PATH
from src.web_automation.navigation import navigate_to_queens_puzzle

def run_navigation_flow():
    agent = navigate_to_queens_puzzle(
        driver_path=DRIVER_PATH,
        username="REMOVED",
        password="REMOVED",
        headless=False
    )

    return agent

if __name__ == '__main__':
    agent = run_navigation_flow()
    agent.close()