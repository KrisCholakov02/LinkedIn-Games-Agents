import os
import numpy as np
import cv2
import pytesseract

def detect_grid_size_from_lines(image, debug=True):
    """
    Detects the number of rows and columns using greyish grid lines.

    Args:
        image (np.ndarray): BGR puzzle image.
        debug (bool): If True, saves an image with drawn grid lines.

    Returns:
        tuple: (num_rows, num_cols, horizontal_lines_pos, vertical_lines_pos)
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=5
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    horizontal_projection = np.sum(horizontal_lines, axis=1)
    vertical_projection = np.sum(vertical_lines, axis=0)

    h_threshold = np.max(horizontal_projection) * 0.5
    v_threshold = np.max(vertical_projection) * 0.5

    h_indices = np.where(horizontal_projection > h_threshold)[0]
    v_indices = np.where(vertical_projection > v_threshold)[0]

    def group_lines(indices, min_gap=5):
        grouped = []
        if len(indices) == 0:
            return grouped
        group = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] > min_gap:
                grouped.append(group)
                group = []
            group.append(indices[i])
        grouped.append(group)
        return [int(np.mean(g)) for g in grouped]

    horizontal_lines_pos = group_lines(h_indices)
    vertical_lines_pos = group_lines(v_indices)

    if debug:
        debug_img = original.copy()
        for y in horizontal_lines_pos:
            cv2.line(debug_img, (0, y), (debug_img.shape[1], y), (0, 255, 0), 2)
        for x in vertical_lines_pos:
            cv2.line(debug_img, (x, 0), (x, debug_img.shape[0]), (255, 0, 0), 2)
        os.makedirs("img/debug", exist_ok=True)
        cv2.imwrite("img/debug/zip_grid.png", debug_img)
        print("Saved grid debug image to 'img/debug/zip_grid.png'.")

    num_rows = len(horizontal_lines_pos) - 1
    num_cols = len(vertical_lines_pos) - 1
    if num_rows < 1 or num_cols < 1:
        raise ValueError("Could not detect valid grid lines.")

    return num_rows, num_cols, horizontal_lines_pos, vertical_lines_pos

def extract_cell_images(image, h_lines, v_lines, margin=10):
    """
    Divides the board image into individual cell images.

    Args:
        image (np.ndarray): BGR image.
        h_lines (list): y-coordinates of detected horizontal lines.
        v_lines (list): x-coordinates of detected vertical lines.
        margin (int): Margin to avoid grid lines.

    Returns:
        list of tuples: [((row, col), cell_img), ...]
    """
    cell_images = []
    os.makedirs("img/debug/cells", exist_ok=True)
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]
            y1_inner = min(y1 + margin, y2)
            y2_inner = max(y2 - margin, y1_inner + 1)
            x1_inner = min(x1 + margin, x2)
            x2_inner = max(x2 - margin, x1_inner + 1)
            cell_img = image[y1_inner:y2_inner, x1_inner:x2_inner]
            cell_images.append(((i, j), cell_img))
            cv2.imwrite(f"img/debug/cells/cell_{i}_{j}.png", cell_img)
    return cell_images

def recognize_digit_ocr(cell_img, r_c, debug=False):
    """
    Recognizes digits in the cell image using OCR.

    Args:
        cell_img (np.ndarray): BGR or grayscale cell image.
        r_c (tuple): (row, col) for naming debug files.
        debug (bool): If True, prints OCR result.

    Returns:
        str: Recognized digit as a string, or "empty" if not detected.
    """
    if len(cell_img.shape) == 3 and cell_img.shape[2] == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    height, width = gray.shape[:2]
    crop_h = int(height * 0.2)
    crop_w = int(width * 0.2)
    gray = gray[crop_h:height - crop_h, crop_w:width - crop_w]

    debug_path = f"img/temp/ocr_cell_{r_c[0]}_{r_c[1]}.png"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    cv2.imwrite(debug_path, gray)
    if debug:
        print(f"[OCR] Saved binary cell image to '{debug_path}'.")

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(gray, config=config).strip()
    if debug:
        print(f"[OCR] Recognized text: '{text}'")
    return text if text != "" else "empty"

def detect_walls(image, h_lines, v_lines, wall_thickness=3, intensity_threshold=50, margin=2, debug=True):
    """
    Detects walls between cells by sampling the inner border regions.

    Args:
        image (np.ndarray): Original BGR board image.
        h_lines (list): y-coordinates of detected horizontal grid lines.
        v_lines (list): x-coordinates of detected vertical grid lines.
        wall_thickness (int): Half-width (in pixels) of the border region to sample.
        intensity_threshold (int): Threshold below which the border is considered a wall.
        margin (int): Margin from cell borders to avoid overlap with cell content.
        debug (bool): If True, saves debug images for the wall regions.

    Returns:
        dict: Mapping of adjacent cell pairs to a boolean indicating presence of a wall.
    """
    walls = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    os.makedirs("img/debug/walls", exist_ok=True)

    num_rows = len(h_lines) - 1
    num_cols = len(v_lines) - 1

    for r in range(num_rows):
        for j in range(num_cols - 1):
            x_center = v_lines[j + 1]
            x1 = max(x_center - wall_thickness, 0)
            x2 = min(x_center + wall_thickness, gray.shape[1])
            y1 = h_lines[r] + margin
            y2 = h_lines[r + 1] - margin
            if y2 <= y1:
                continue
            border_region = gray[y1:y2, x1:x2]
            avg_intensity = np.mean(border_region)
            is_wall = avg_intensity < intensity_threshold
            walls[((r, j), (r, j + 1))] = is_wall
            if debug:
                debug_path = f"img/debug/walls/vertical_r{r}_c{j}_{j + 1}.png"
                cv2.imwrite(debug_path, border_region)
                print(
                    f"[WALL] Vertical wall between (r={r},c={j}) and (r={r},c={j + 1}): avg intensity = {avg_intensity:.1f} -> {'WALL' if is_wall else 'no wall'}")

    for c in range(num_cols):
        for r in range(num_rows - 1):
            y_center = h_lines[r + 1]
            y1 = max(y_center - wall_thickness, 0)
            y2 = min(y_center + wall_thickness, gray.shape[0])
            x1 = v_lines[c] + margin
            x2 = v_lines[c + 1] - margin
            if x2 <= x1:
                continue
            border_region = gray[y1:y2, x1:x2]
            avg_intensity = np.mean(border_region)
            is_wall = avg_intensity < intensity_threshold
            walls[((r, c), (r + 1, c))] = is_wall
            if debug:
                debug_path = f"img/debug/walls/horizontal_r{r}_{r + 1}_c{c}.png"
                cv2.imwrite(debug_path, border_region)
                print(
                    f"[WALL] Horizontal wall between (r={r},c={c}) and (r={r + 1},c={c}): avg intensity = {avg_intensity:.1f} -> {'WALL' if is_wall else 'no wall'}")
    return walls

def recognize_zip_board(image, debug=True):
    """
    Full recognition pipeline for the Zip game.

    Args:
        image (np.ndarray): BGR screenshot of the puzzle board.
        debug (bool): If True, prints debug info and saves intermediate images.

    Returns:
        tuple:
          - cell_map: dict mapping (row, col) to recognized content (digit as string or "empty")
          - walls_map: dict mapping adjacent cell pairs to a boolean (True if wall present)
          - grid_size: (num_rows, num_cols)
    """
    num_rows, num_cols, h_lines, v_lines = detect_grid_size_from_lines(image, debug=debug)
    if debug:
        print(f"Detected grid: {num_rows} rows × {num_cols} cols")

    cell_images = extract_cell_images(image, h_lines, v_lines, margin=10)

    cell_map = {}
    for ((r, c), cell_img) in cell_images:
        recognized = recognize_digit_ocr(cell_img, (r, c), debug=debug)
        cell_map[(r, c)] = recognized
        if debug:
            print(f"Cell ({r}, {c}) recognized as: {recognized}")

    walls_map = detect_walls(image, h_lines, v_lines, wall_thickness=3, intensity_threshold=50, margin=2, debug=debug)

    return cell_map, walls_map, (num_rows, num_cols)