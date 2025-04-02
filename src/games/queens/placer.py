from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def place_queens_by_dom(driver, queen_positions, num_cols):
    """
    Places queens on the board by interacting with DOM elements,
    using the 'data-cell-idx' attribute to target grid cells.

    Args:
        driver (selenium.webdriver): An active Selenium WebDriver instance.
        queen_positions (list of tuple): List of (row, col) positions for each queen.
        num_cols (int): Number of columns in the grid (used to calculate cell index).
    """
    wait = WebDriverWait(driver, 10)

    for row, col in queen_positions:
        try:
            cell_idx = row * num_cols + col
            selector = f'div[data-cell-idx="{cell_idx}"]'

            cell_element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            cell_element.click()
            cell_element.click()  # Double-click to place a queen

            print(f"[✓] Placed queen at (row={row}, col={col}, idx={cell_idx})")
        except Exception as e:
            print(f"[✗] Failed to place queen at (row={row}, col={col}) — {e}")