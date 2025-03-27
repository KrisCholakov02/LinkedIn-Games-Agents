import os
import time
import cv2
from PIL import Image
import io
from glob import glob

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def clear_puzzle(driver, debug=False):
    """
    Clicks the main 'Clear' button (id='aux-controls-clear'), then waits for
    the confirm modal's "Clear" button by finding a <span> with text='Clear'
    inside a primary confirm-dialog button. Then clicks that to finalize.
    If either step fails, logs a warning if debug=True.
    """
    try:
        main_clear_button = driver.find_element(By.ID, "aux-controls-clear")
        main_clear_button.click()
        time.sleep(0.3)
        if debug:
            print("[i] Clicked the primary 'Clear' button.")
    except Exception as e:
        if debug:
            print(f"[!] Could not find/click main Clear button: {e}")
        return

    try:
        # We look for a <span> with text='Clear', inside the confirm-dialog button
        # e.g. <button ... class="artdeco-modal__confirm-dialog-btn"><span>Clear</span></button>
        wait = WebDriverWait(driver, 5)
        confirm_clear_button = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(@class,'artdeco-modal__confirm-dialog-btn') and .//span[text()='Clear']]"
            ))
        )
        confirm_clear_button.click()
        time.sleep(0.3)
        if debug:
            print("[i] Confirmed Clear in the modal.")
    except Exception as e:
        if debug:
            print(f"[!] Could not find/click the confirm dialog's Clear button: {e}")

def load_first_image_in_directory(dir_path):
    if not os.path.isdir(dir_path):
        return None
    candidates = glob(os.path.join(dir_path, "*.png")) + glob(os.path.join(dir_path, "*.jpg"))
    if not candidates:
        return None
    return cv2.imread(candidates[0], cv2.IMREAD_GRAYSCALE)

def load_cluster_references():
    ref_icon1 = load_first_image_in_directory("img/debug/cells/icon1")
    ref_icon2 = load_first_image_in_directory("img/debug/cells/icon2")
    return {'icon1': ref_icon1, 'icon2': ref_icon2}

def template_match_score(cell_gray, template_gray):
    th, tw = template_gray.shape[:2]
    ch, cw = cell_gray.shape[:2]
    if th > ch or tw > cw:
        return 0.0
    result = cv2.matchTemplate(cell_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val

def classify_single_cell(image, refs, threshold=0.8):
    if image is None or (refs['icon1'] is None and refs['icon2'] is None):
        return "empty"
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score_icon1 = 0.0
    if refs['icon1'] is not None:
        score_icon1 = template_match_score(image, refs['icon1'])
    score_icon2 = 0.0
    if refs['icon2'] is not None:
        score_icon2 = template_match_score(image, refs['icon2'])
    best_score = max(score_icon1, score_icon2)
    if best_score < threshold:
        return "empty"
    return "icon1" if score_icon1 > score_icon2 else "icon2"

def capture_cell_screenshot(driver, cell_element, output_path):
    try:
        png_data = driver.get_screenshot_as_png()
        im = Image.open(io.BytesIO(png_data))
        loc = cell_element.location
        sz = cell_element.size
        left, top = loc['x'], loc['y']
        right, bottom = left + sz['width'], top + sz['height']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        im_cropped = im.crop((left, top, right, bottom))
        im_cropped.save(output_path)
        return output_path
    except Exception as e:
        print(f"[!] Error capturing cell screenshot: {e}")
        return None

def classify_cell_state(driver, cell_element, refs):
    path = "img/temp/cell_check.png"
    capture_cell_screenshot(driver, cell_element, path)
    img = cv2.imread(path)
    return classify_single_cell(img, refs) if img is not None else "empty"

def detect_click_map(driver, rows, cols, refs, debug=False):
    wait = WebDriverWait(driver, 2)
    test_cell = None
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            sel = f"div[data-cell-idx='{idx}']"
            try:
                cell_elem = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
                if cell_elem.get_attribute("aria-disabled") == "true":
                    continue
                if classify_cell_state(driver, cell_elem, refs) == "empty":
                    test_cell = cell_elem
                    if debug:
                        print(f"[i] Found empty test cell at (r={r}, c={c}), idx={idx}.")
                    break
            except:
                pass
        if test_cell:
            break
    if not test_cell:
        if debug:
            print("[!] No empty cell found; defaulting icon1->1, icon2->2.")
        return {"icon1": 1, "icon2": 2}

    test_cell.click()
    time.sleep(0.2)
    first_click_icon = classify_cell_state(driver, test_cell, refs)
    if debug:
        print(f"[i] Single-click => '{first_click_icon}'")
    if first_click_icon == "icon1":
        return {"icon1": 1, "icon2": 2}
    elif first_click_icon == "icon2":
        return {"icon2": 1, "icon1": 2}
    if debug:
        print("[!] First click did not yield icon1/icon2; defaulting map.")
    return {"icon1": 1, "icon2": 2}

def place_solution(driver, solution_grid, skip_locked=True, debug=False):
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
            if desired_icon not in ("icon1", "icon2"):
                if debug:
                    print(f"[i] Skipping cell({r},{c}) => '{desired_icon}'.")
                continue
            needed_clicks = click_map.get(desired_icon, 2)
            idx = r * cols + c
            sel = f"div[data-cell-idx='{idx}']"
            try:
                cell_elem = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
            except Exception as e:
                if debug:
                    print(f"[!] Could not locate cell({r},{c}) idx={idx}: {e}")
                continue
            if skip_locked and cell_elem.get_attribute("aria-disabled") == "true":
                if debug:
                    print(f"[i] Cell({r},{c}) locked. Skipping.")
                continue
            if debug:
                print(f"[i] Setting cell({r},{c}) => '{desired_icon}' with {needed_clicks} click(s).")
            for _ in range(needed_clicks):
                cell_elem.click()
                time.sleep(0.05)
    if debug:
        print("[✓] Finished placing solution.")