import os
import numpy as np
import cv2
import pytesseract


def detect_grid_size_from_lines(image, debug=True):
    """
    Detects number of rows and columns using thick black lines in a grid.

    Args:
        image (np.ndarray): BGR puzzle image.
        debug (bool): If True, saves an image with drawn grid lines.

    Returns:
        (int, int): (num_rows, num_cols)
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to extract dark (black) lines
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to isolate horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Compute projection profiles
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

    return num_rows, num_cols


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


def recognize_digit_ocr(cell_img, debug=False):
    """
    Uses pytesseract OCR to recognize digits in the cell image.

    Args:
        cell_img (np.ndarray): BGR or grayscale cell image.
        debug (bool): If True, prints the OCR result.

    Returns:
        str: The recognized digit (as a string) or "empty" if no digit is found.
    """

    # Convert to grayscale if needed.
    if len(cell_img.shape) == 3 and cell_img.shape[2] == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    # Optionally, apply thresholding to improve OCR performance.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use pytesseract with configuration for digits only.
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config).strip()
    if debug:
        print(f"[OCR] Recognized text: '{text}'")

    # If no text is found, assume the cell is empty.
    return text if text != "" else "empty"


def recognize_zip_board(image, debug=True):
    """
    Full recognition pipeline for the Zip game:
      1. Detects grid lines to determine the number of rows and columns.
      2. Extracts individual cell images.
      3. Recognizes the content of each cell using OCR (digits) – if a digit is detected, returns that;
         otherwise, returns "empty".

    Args:
        image (np.ndarray): BGR screenshot of the puzzle board.
        debug (bool): If True, prints debug information and saves intermediate images.

    Returns:
        tuple:
          - cell_map: dict mapping (row, col) to recognized content (digit as string or "empty")
          - grid_size: (num_rows, num_cols)
    """
    num_rows, num_cols = detect_grid_size_from_lines(image, debug=debug)
    if debug:
        print(f"Detected grid: {num_rows} rows × {num_cols} cols")

    # Create evenly spaced line positions based on image dimensions.
    h_lines = list(np.linspace(0, image.shape[0], num_rows + 1, dtype=int))
    v_lines = list(np.linspace(0, image.shape[1], num_cols + 1, dtype=int))

    cell_images = extract_cell_images(image, h_lines, v_lines, margin=10)

    cell_map = {}
    for ((r, c), cell_img) in cell_images:
        recognized = recognize_digit_ocr(cell_img, debug=debug)
        cell_map[(r, c)] = recognized
        if debug:
            print(f"Cell ({r},{c}) recognized as: {recognized}")

    return cell_map, (num_rows, num_cols)