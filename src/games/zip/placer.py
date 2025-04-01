from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def place_solution(driver, solution_grid, num_cols, debug=False):
    """
    Places the Zip puzzle solution on the board by clicking the cells in the given order.
    The solution_grid is an ordered list of (row, col) positions representing the path.
    The function clicks each cell in the solution path.

    Args:
        driver: Selenium WebDriver instance.
        solution_grid (list of (row, col)): Ordered list of cell coordinates forming the path.
        num_cols (int): Number of columns in the grid (to compute cell indices).
        debug (bool): If True, prints debug information.
    """
    if not solution_grid:
        print("[!] No solution to place.")
        return

    wait = WebDriverWait(driver, 5)

    for cell in solution_grid:
        idx = cell[0] * num_cols + cell[1]
        selector = f"div[data-cell-idx='{idx}']"
        cell_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        cell_elem.click()
        if debug:
            cell_loc = cell_elem.location
            cell_size = cell_elem.size
            cell_center = (int(cell_loc['x'] + cell_size['width'] / 2),
                           int(cell_loc['y'] + cell_size['height'] / 2))
            print(f"[i] Clicked cell {cell} (idx {idx}) center: {cell_center}")
        sleep(0.05)

    if debug:
        print("[✓] Finished placing solution by clicking cells.")