from time import sleep

import cv2

from test_navigation import run_navigation_flow
from src.comp_vision.screenshot import take_screenshot


def test_take_screenshot():
    """
    Runs the navigation flow and takes a screenshot of the puzzle page.
    """
    agent = run_navigation_flow()
    sleep(5)  # Allow puzzle to load

    take_screenshot(agent.driver, "img/screenshot_test.png")
    print("Screenshot complete.")

    # Return the image for further processing
    screenshot_img = cv2.imread("img/screenshot_test.png")
    return agent, screenshot_img


if __name__ == '__main__':
    agent = test_take_screenshot()
    agent.close()