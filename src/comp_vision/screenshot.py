import cv2
import numpy as np


def take_screenshot(driver, output_path: str) -> None:
    """
    Takes a screenshot of the current browser window using the provided WebDriver
    and saves it to the specified path.

    Args:
        driver (selenium.webdriver): The active Selenium WebDriver instance.
        output_path (str): The file path to save the screenshot (e.g., "img/screenshot_test.png").
    """
    # Capture screenshot as PNG binary
    screenshot_png = driver.get_screenshot_as_png()

    # Decode into OpenCV image format
    np_img = np.frombuffer(screenshot_png, np.uint8)
    screenshot_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Save image to disk
    cv2.imwrite(output_path, screenshot_img)
    print(f"Screenshot saved to: {output_path}")