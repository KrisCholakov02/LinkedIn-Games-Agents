import math
import os
import cv2
import numpy as np
import time
from selenium.webdriver.common.by import By

def take_screenshot(driver, output_path: str, board_class) -> None:
    """
    Captures and saves a screenshot of the puzzle board using DOM-based coordinates.

    Args:
        driver (WebDriver): Selenium WebDriver instance.
        output_path (str): Path to save the screenshot.
        board_class (str): CSS class name of the board element.
    """
    time.sleep(0.3)

    # Scroll the board element into view
    board_element = driver.find_element(By.CLASS_NAME, board_class)
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", board_element)

    # Get viewport-relative coordinates of the board element
    rect = driver.execute_script("""
        const rect = arguments[0].getBoundingClientRect();
        return {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height
        };
    """, board_element)

    # Capture full viewport screenshot
    screenshot_png = driver.get_screenshot_as_png()
    np_img = np.frombuffer(screenshot_png, np.uint8)
    full_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convert and round coordinates to int
    x, y, w, h = map(math.ceil, (rect['x'], rect['y'], rect['width'], rect['height']))

    # Crop the image using the viewport coordinates
    cropped = full_img[y:y + h, x:x + w]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, cropped)
    if success:
        print(f"[✓] Cropped puzzle screenshot saved to: {output_path}")
    else:
        print(f"[✗] Failed to save screenshot to: {output_path}")