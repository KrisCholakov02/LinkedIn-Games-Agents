from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def place_solution(driver, solution_grid, num_cols, debug=False):
    """
    Executes the placement of the Zip puzzle solution on the board.

    Each cell in the provided solution path is clicked in order to trace the correct path.

    Args:
        driver (selenium.webdriver): Active WebDriver instance.
        solution_grid (list of (row, col)): Ordered sequence of cell coordinates forming the path.
        num_cols (int): Total number of columns in the grid (used for linear index computation).
        debug (bool): If True, prints detailed click information including screen coordinates.
    """
    if not solution_grid:
        print("[!] No solution to place.")
        return

    wait = WebDriverWait(driver, 5)

    for row, col in solution_grid:
        idx = row * num_cols + col
        selector = f"div[data-cell-idx='{idx}']"
        cell_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        cell_elem.click()

        if debug:
            loc = cell_elem.location
            size = cell_elem.size
            center = (int(loc['x'] + size['width'] / 2), int(loc['y'] + size['height'] / 2))
            print(f"[i] Clicked cell ({row}, {col}) [idx={idx}] at screen center {center}")

        sleep(0.01)

    if debug:
        print("[✓] Completed input of solution path.")