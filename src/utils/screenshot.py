import cv2
import numpy as np
from selenium.webdriver.common.by import By
import time


def take_screenshot(driver, output_path: str) -> None:
    """
    Captures and saves a screenshot of only the puzzle board by rendering
    the DOM element directly (no pixel math needed).

    Args:
        driver (selenium.webdriver): The active Selenium WebDriver instance.
        output_path (str): The file path to save the screenshot (e.g., "img/board.png").
    """
    time.sleep(1)

    board_element = driver.find_element(By.CLASS_NAME, "queens-board")

    # Scroll into view just to be safe (optional)
    driver.execute_script("arguments[0].scrollIntoView(true);", board_element)
    time.sleep(0.2)

    # Use Selenium's built-in per-element screenshot
    png_data = board_element.screenshot_as_png

    # Decode and save using OpenCV
    np_img = np.frombuffer(png_data, np.uint8)
    board_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    cv2.imwrite(output_path, board_img)
    print(f"Board screenshot saved to: {output_path}")