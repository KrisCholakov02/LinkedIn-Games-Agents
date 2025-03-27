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
    the confirm modal's "Clear" button (a <span> with text='Clear' inside a
    primary confirm-dialog button), then clicks that to finalize.
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
    """
    Standard classification function: compares the cell image
    to icon1.png and icon2.png, picking the best if > threshold.
    """
    if image is None or (refs['icon1'] is None and refs['icon2'] is None):
        return "empty"
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score_icon1, score_icon2 = 0.0, 0.0
    if refs['icon1'] is not None:
        score_icon1 = template_match_score(image, refs['icon1'])
    if refs['icon2'] is not None:
        score_icon2 = template_match_score(image, refs['icon2'])
    best = max(score_icon1, score_icon2)
    if best < threshold:
        return "empty"
    return "icon1" if score_icon1 > score_icon2 else "icon2"

def capture_cell_screenshot(driver, cell_element, output_path):
    """
    Captures a screenshot of just the bounding box region of one cell element.
    """
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

def classify_cell_state(driver, cell_element, refs, filename="cell_check.png"):
    path = f"img/temp/{filename}"
    cap_path = capture_cell_screenshot(driver, cell_element, path)
    if cap_path is None:
        return "empty"
    img = cv2.imread(cap_path)
    if img is None:
        return "empty"
    return classify_single_cell(img, refs)

def detect_click_map(driver, rows, cols, refs, debug=False):
    """
    1) Finds one empty cell.
    2) Clicks it once -> saves 'click1_icon.png',
       compares only to icon1 ref to decide if it surpasses threshold => icon1 or else => icon2
    3) Clicks it again -> saves 'click2_icon.png' for debugging only
    """
    wait = WebDriverWait(driver, 2)
    test_cell = None

    # find a single empty cell
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            selector = f"div[data-cell-idx='{idx}']"
            try:
                cell_elem = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                if cell_elem.get_attribute("aria-disabled") == "true":
                    continue
                before_state = classify_cell_state(driver, cell_elem, refs, filename="click_map_before.png")
                if before_state == "empty":
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
            print("[!] No empty cell found. Default => icon1->1, icon2->2")
        return {"icon1": 1, "icon2": 2}

    # 1) First click => see how well it matches icon1
    test_cell.click()
    time.sleep(0.2)
    click1_path = "img/temp/click1_icon.png"
    classify_cell_state(driver, test_cell, refs, filename="click1_icon.png")

    # 2) Second click => for debugging
    test_cell.click()
    time.sleep(0.2)
    classify_cell_state(driver, test_cell, refs, filename="click2_icon.png")

    # load the first screenshot again
    click1_img = cv2.imread(click1_path, cv2.IMREAD_GRAYSCALE)
    icon1_ref = refs['icon1']
    score1_icon1 = 0.0
    if click1_img is not None and icon1_ref is not None:
        score1_icon1 = template_match_score(click1_img, icon1_ref)

    # optionally print the second screenshot's comparison for debugging
    click2_path = "img/temp/click2_icon.png"
    click2_img = cv2.imread(click2_path, cv2.IMREAD_GRAYSCALE)
    score2_icon1 = 0.0
    if click2_img is not None and icon1_ref is not None:
        score2_icon1 = template_match_score(click2_img, icon1_ref)

    if debug:
        print(f"[DEBUG] score(click1_icon.png vs icon1) = {score1_icon1:.3f}")
        print(f"[DEBUG] score(click2_icon.png vs icon1) = {score2_icon1:.3f}")

    # threshold
    threshold = 0.8
    # if first click => strong match with icon1 => "icon1" => 1 click, else => "icon2" => 1 click
    if score1_icon1 >= threshold:
        if debug:
            print("[i] 1 click => icon1, so we do icon1->1, icon2->2")
        return {"icon1": 1, "icon2": 2}
    else:
        if debug:
            print("[i] 1 click => not icon1 => icon2->1, icon1->2")
        return {"icon2": 1, "icon1": 2}

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
            selector = f"div[data-cell-idx='{idx}']"

            try:
                cell_elem = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
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