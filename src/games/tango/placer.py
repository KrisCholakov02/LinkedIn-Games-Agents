import os
import io
import time
import cv2
from PIL import Image
from glob import glob

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def clear_puzzle(driver, debug=False):
    """
    Clicks the 'Clear' button and confirms the action via modal to reset the puzzle.

    Args:
        driver (webdriver): Active Selenium WebDriver.
        debug (bool): If True, prints debug information.
    """
    try:
        main_clear_button = driver.find_element(By.ID, "aux-controls-clear")
        main_clear_button.click()
        time.sleep(0.1)
        if debug:
            print("[i] Clicked the main 'Clear' button.")
    except Exception as e:
        if debug:
            print(f"[!] Failed to click main Clear button: {e}")
        return

    try:
        wait = WebDriverWait(driver, 5)
        confirm_button = wait.until(EC.element_to_be_clickable((
            By.XPATH,
            "//button[contains(@class,'artdeco-modal__confirm-dialog-btn') and .//span[text()='Clear']]"
        )))
        confirm_button.click()
        time.sleep(0.1)
        if debug:
            print("[i] Confirmed Clear in modal.")
    except Exception as e:
        if debug:
            print(f"[!] Failed to confirm Clear in modal: {e}")


def load_first_image_in_directory(dir_path):
    """
    Loads the first PNG or JPG file in the specified directory.

    Args:
        dir_path (str): Directory path to search.

    Returns:
        np.ndarray or None: Loaded grayscale image or None if no image found.
    """
    if not os.path.isdir(dir_path):
        return None
    candidates = glob(os.path.join(dir_path, "*.png")) + glob(os.path.join(dir_path, "*.jpg"))
    if not candidates:
        return None
    return cv2.imread(candidates[0], cv2.IMREAD_GRAYSCALE)


def load_cluster_references():
    """
    Loads icon reference templates for classification.

    Returns:
        dict: {'icon1': np.ndarray or None, 'icon2': np.ndarray or None}
    """
    return {
        'icon1': load_first_image_in_directory("img/debug/cells/icon1"),
        'icon2': load_first_image_in_directory("img/debug/cells/icon2")
    }


def template_match_score(cell_gray, template_gray):
    """
    Calculates similarity score between two grayscale images using template matching.

    Returns:
        float: Match score (0.0 to 1.0).
    """
    th, tw = template_gray.shape[:2]
    ch, cw = cell_gray.shape[:2]
    if th > ch or tw > cw:
        return 0.0
    result = cv2.matchTemplate(cell_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


def classify_single_cell(image, refs, threshold=0.8):
    """
    Classifies a cell image as 'icon1', 'icon2', or 'empty' using template matching.

    Args:
        image (np.ndarray): Cell image in BGR or grayscale.
        refs (dict): Icon templates.
        threshold (float): Matching threshold.

    Returns:
        str: One of 'icon1', 'icon2', or 'empty'.
    """
    if image is None or (refs['icon1'] is None and refs['icon2'] is None):
        return "empty"
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    score1 = template_match_score(image, refs['icon1']) if refs['icon1'] is not None else 0.0
    score2 = template_match_score(image, refs['icon2']) if refs['icon2'] is not None else 0.0

    best = max(score1, score2)
    if best < threshold:
        return "empty"
    return "icon1" if score1 > score2 else "icon2"


def capture_cell_screenshot(driver, cell_element, output_path):
    """
    Captures and saves a screenshot of a single cell's bounding box.

    Args:
        driver (webdriver): Active Selenium WebDriver.
        cell_element (WebElement): Target cell.
        output_path (str): Where to save the screenshot.

    Returns:
        str or None: Path to saved screenshot or None on failure.
    """
    try:
        png_data = driver.get_screenshot_as_png()
        im = Image.open(io.BytesIO(png_data))
        loc = cell_element.location
        sz = cell_element.size
        bbox = (loc['x'], loc['y'], loc['x'] + sz['width'], loc['y'] + sz['height'])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        im.crop(bbox).save(output_path)
        return output_path
    except Exception as e:
        print(f"[!] Error capturing cell screenshot: {e}")
        return None


def classify_cell_state(driver, cell_element, refs, filename="cell_check.png"):
    """
    Captures and classifies the icon inside a cell.

    Returns:
        str: One of 'icon1', 'icon2', or 'empty'.
    """
    path = f"img/temp/{filename}"
    cap_path = capture_cell_screenshot(driver, cell_element, path)
    if cap_path is None:
        return "empty"
    img = cv2.imread(cap_path)
    return classify_single_cell(img, refs) if img is not None else "empty"


def detect_click_map(driver, rows, cols, refs, debug=False):
    """
    Determines how many clicks are needed to set each icon type.

    Returns:
        dict: {'icon1': clicks, 'icon2': clicks}
    """
    wait = WebDriverWait(driver, 2)
    test_cell = None

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            selector = f"div[data-cell-idx='{idx}']"
            try:
                cell_elem = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                if cell_elem.get_attribute("aria-disabled") == "true":
                    continue
                if classify_cell_state(driver, cell_elem, refs, filename="click_map_before.png") == "empty":
                    test_cell = cell_elem
                    if debug:
                        print(f"[i] Found test cell at (r={r}, c={c}), idx={idx}.")
                    break
            except:
                continue
        if test_cell:
            break

    if not test_cell:
        if debug:
            print("[!] No empty test cell found. Default: icon1→1, icon2→2")
        return {"icon1": 1, "icon2": 2}

    test_cell.click()
    time.sleep(0.1)
    classify_cell_state(driver, test_cell, refs, filename="click1_icon.png")

    test_cell.click()
    time.sleep(0.1)
    classify_cell_state(driver, test_cell, refs, filename="click2_icon.png")

    click1_img = cv2.imread("img/temp/click1_icon.png", cv2.IMREAD_GRAYSCALE)
    score = template_match_score(click1_img, refs['icon1']) if click1_img is not None else 0.0

    if debug:
        print(f"[DEBUG] score(click1_icon vs icon1) = {score:.3f}")

    return {"icon1": 1, "icon2": 2} if score >= 0.8 else {"icon2": 1, "icon1": 2}


def place_solution(driver, solution_grid, skip_locked=True, debug=False):
    """
    Places a solved Tango puzzle on the board.

    Args:
        driver (webdriver): Active Selenium WebDriver.
        solution_grid (list of list): Final solution with icon names.
        skip_locked (bool): Whether to skip locked cells.
        debug (bool): If True, prints debug info.
    """
    rows = len(solution_grid)
    if rows == 0:
        if debug:
            print("[i] No solution to place.")
        return

    cols = len(solution_grid[0])
    clear_puzzle(driver, debug=debug)

    refs = load_cluster_references()
    click_map = detect_click_map(driver, rows, cols, refs, debug=debug)
    clear_puzzle(driver, debug=debug)

    wait = WebDriverWait(driver, 2)

    for r in range(rows):
        for c in range(cols):
            desired_icon = solution_grid[r][c]
            if desired_icon not in {"icon1", "icon2"}:
                if debug:
                    print(f"[i] Skipping cell({r},{c}) => '{desired_icon}'.")
                continue

            idx = r * cols + c
            selector = f"div[data-cell-idx='{idx}']"

            try:
                cell_elem = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            except Exception as e:
                if debug:
                    print(f"[!] Failed to access cell({r},{c}) idx={idx}: {e}")
                continue

            if skip_locked and cell_elem.get_attribute("aria-disabled") == "true":
                if debug:
                    print(f"[i] Cell({r},{c}) locked. Skipping.")
                continue

            needed_clicks = click_map.get(desired_icon, 2)
            if debug:
                print(f"[i] Setting cell({r},{c}) to '{desired_icon}' with {needed_clicks} click(s).")

            for _ in range(needed_clicks):
                cell_elem.click()
                time.sleep(0.05)

    if debug:
        print("[✓] Finished placing the solution.")