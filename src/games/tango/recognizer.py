import os
import numpy as np
import cv2
from sklearn.cluster import KMeans


def detect_grid_size_from_faded_lines(image, debug=True):
    """
    Detects the horizontal and vertical grid lines (their y and x positions, respectively)
    from a partially faded board image. Uses morphological operations and projections
    to find strong row/column separators. Returns two lists of integer coordinates:
    horizontal_lines_pos (y-coordinates) and vertical_lines_pos (x-coordinates).
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold to highlight lines
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=5
    )

    # Morphological kernels to detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Extract candidate horizontal and vertical lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Project each line map to find strong line positions
    horizontal_projection = np.sum(horizontal_lines, axis=1)
    vertical_projection = np.sum(vertical_lines, axis=0)

    # Heuristic thresholds (30% of max) to decide which rows/cols are lines
    h_threshold = np.max(horizontal_projection) * 0.3
    v_threshold = np.max(vertical_projection) * 0.3

    h_indices = np.where(horizontal_projection > h_threshold)[0]
    v_indices = np.where(vertical_projection > v_threshold)[0]

    # Group consecutive indices into line positions
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
        # Return the mean index of each group
        return [int(np.mean(g)) for g in grouped]

    horizontal_lines_pos = group_lines(h_indices)
    vertical_lines_pos = group_lines(v_indices)

    # Optional debug visualization
    if debug:
        debug_img = original.copy()
        for y in horizontal_lines_pos:
            cv2.line(debug_img, (0, y), (debug_img.shape[1], y), (0, 255, 0), 2)
        for x in vertical_lines_pos:
            cv2.line(debug_img, (x, 0), (x, debug_img.shape[0]), (255, 0, 0), 2)

        os.makedirs("img/debug", exist_ok=True)
        cv2.imwrite("img/debug/tango_grid.png", debug_img)
        print("[✓] Saved grid debug image to 'img/debug/tango_grid.png'.")

    # Basic check: need at least two horizontal and vertical lines to form a grid
    if len(horizontal_lines_pos) < 2 or len(vertical_lines_pos) < 2:
        raise ValueError("[✗] Failed to detect valid grid dimensions.")

    return horizontal_lines_pos, vertical_lines_pos


def crop_cell_inners(image, h_lines, v_lines, margin=10):
    """
    Crops out the interior portion of each cell to gather individual cell images.
    The margin parameter shrinks the cell boundary inward to avoid lines.

    Args:
        image (np.ndarray): Input BGR image of the board.
        h_lines (list of int): Sorted y-positions of horizontal grid lines.
        v_lines (list of int): Sorted x-positions of vertical grid lines.
        margin (int): Pixels to shrink from each side to avoid lines.

    Returns:
        A list of tuples [((row_index, col_index), cell_img), ...].
    """
    cell_images = []
    os.makedirs("img/debug/cells", exist_ok=True)

    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]

            # Shrink inwards by 'margin' to avoid the lines
            y1_inner = min(y1 + margin, y2)
            y2_inner = max(y2 - margin, y1_inner + 1)
            x1_inner = min(x1 + margin, x2)
            x2_inner = max(x2 - margin, x1_inner + 1)

            cell_img = image[y1_inner:y2_inner, x1_inner:x2_inner]
            cell_images.append(((i, j), cell_img))

            # Save debug image for each cell
            cv2.imwrite(f"img/debug/cells/cell_{i}_{j}.png", cell_img)

    return cell_images


def determine_optimal_k(features, max_k=3):
    """
    Small helper function to decide how many clusters (k) to use in a KMeans model,
    based on the "elbow" or relative improvement in inertia.
    """
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # Compare successive drops in inertia
    deltas = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]
    if len(deltas) == 0:
        return 1
    if len(deltas) == 1 or deltas[1] < 0.1 * deltas[0]:
        return 2
    return 3


def classify_images(image_tuples, max_clusters):
    """
    Generic function to extract basic pixel features from the given images,
    then cluster them with KMeans. Returns the assigned cluster label for
    each corresponding image in image_tuples.

    Args:
        image_tuples: List of ((pos), image) pairs.
        max_clusters: The upper bound for the cluster search (1..max_clusters).

    Returns:
        A list of cluster labels in the same order as image_tuples.
    """
    features = []
    for (_, img) in image_tuples:
        resized = cv2.resize(img, (24, 24))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        norm = blurred / 255.0
        features.append(norm.flatten())

    features_np = np.array(features)
    k = determine_optimal_k(features_np, max_k=max_clusters)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_np)
    return labels


def multi_scale_template_match(board_gray, tmpl_gray, threshold=0.7, scale_factors=None):
    """
    Performs multi-scale template matching for a single sign template
    on a given board image.

    Args:
        board_gray (np.ndarray): Grayscale board image (large).
        tmpl_gray (np.ndarray): Grayscale sign template (small).
        threshold (float): Minimum correlation threshold for a match.
        scale_factors (list): e.g., [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    Returns:
        A list of bounding boxes [ (x1, y1, x2, y2) ] for every found match.
    """
    if scale_factors is None:
        # The range below is just an example; adjust as necessary.
        scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    matches = []
    original_w = tmpl_gray.shape[1]
    original_h = tmpl_gray.shape[0]

    for scale in scale_factors:
        # Resize the template
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        # Avoid invalid or too small sizes
        if new_w < 5 or new_h < 5:
            continue

        scaled_tmpl = cv2.resize(tmpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Perform template matching at this scale
        result = cv2.matchTemplate(board_gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            x1, y1 = pt[0], pt[1]
            x2, y2 = x1 + new_w, y1 + new_h
            matches.append((x1, y1, x2, y2))

    return matches


def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    A simple Non-Max Suppression (NMS) to merge overlapping bounding boxes
    so we don't repeat the same match multiple times. This is optional,
    but can be useful if the same sign is found multiple times at slightly
    shifted positions.

    Args:
        boxes (list): List of bounding boxes [(x1, y1, x2, y2)].
        overlap_thresh (float): IoU threshold for merging.

    Returns:
        A list of filtered bounding boxes with duplicates removed.
    """
    if not boxes:
        return []

    # Convert to float type for computations
    boxes_array = np.array(boxes, dtype=float)

    # Coordinates
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]

    # Compute area of each box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Sort by the lower-right y value of the bounding box
    order = np.argsort(y2)

    suppressed = [False] * len(boxes)
    keep = []

    for i in range(len(boxes)):
        idx = order[i]
        if suppressed[idx]:
            continue

        keep.append(idx)
        for j in range(i + 1, len(boxes)):
            idx2 = order[j]
            if suppressed[idx2]:
                continue

            # Calculate IoU overlap
            xx1 = max(x1[idx], x1[idx2])
            yy1 = max(y1[idx], y1[idx2])
            xx2 = min(x2[idx], x2[idx2])
            yy2 = min(y2[idx], y2[idx2])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap_area = w * h
            iou = overlap_area / (areas[idx] + areas[idx2] - overlap_area)

            if iou > overlap_thresh:
                suppressed[idx2] = True

    # Return only the bounding boxes that survived
    return [boxes[k] for k in keep]


def detect_signs_on_grid(image, sign_templates, h_lines, v_lines, threshold=0.7, debug=True):
    """
    Uses multi-scale template matching to find each sign (e.g., crosses or equals)
    in the given board image. Determines which cells they lie in based on the
    detected grid lines.

    Args:
        image (np.ndarray): BGR board image.
        sign_templates (dict): { 'sign_label': cv2_image_of_sign, ... }
        h_lines (list): Sorted y-coordinates of horizontal grid lines.
        v_lines (list): Sorted x-coordinates of vertical grid lines.
        threshold (float): Match threshold for cv2.matchTemplate (lower for fuzzier matches).
        debug (bool): If True, saves debug images with bounding boxes.

    Returns:
        sign_map: A list of dictionaries, each containing:
          {
            'sign_label': str,
            'bounding_box': (x1, y1, x2, y2),
            'grid_position': (row_index, col_index)
          }
    """
    # If sign_templates is None or empty, skip detection
    if not sign_templates:
        return []

    board_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_img = image.copy() if debug else None
    sign_results = []

    def find_line_index(val, lines):
        """Return i such that lines[i] <= val < lines[i+1], or None if out of range."""
        for i in range(len(lines) - 1):
            if lines[i] <= val < lines[i + 1]:
                return i
        return None

    for label, tmpl in sign_templates.items():
        if tmpl is None or tmpl.size == 0:
            # If the template is empty or not loaded, skip it
            print(f"[!] Warning: Template for '{label}' is empty. Skipping.")
            continue

        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

        # --- Multi-scale matching for each sign template ---
        raw_boxes = multi_scale_template_match(
            board_gray, tmpl_gray, threshold=threshold, scale_factors=None
        )

        # Remove duplicates via simple NMS
        final_boxes = non_max_suppression(raw_boxes, overlap_thresh=0.3)

        for (x1, y1, x2, y2) in final_boxes:
            # Center of bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Determine which cell it's in via grid lines
            row_idx = find_line_index(cy, h_lines)
            col_idx = find_line_index(cx, v_lines)

            if row_idx is None or col_idx is None:
                grid_pos = (None, None)
            else:
                grid_pos = (row_idx, col_idx)

            sign_info = {
                'sign_label': label,
                'bounding_box': (x1, y1, x2, y2),
                'grid_position': grid_pos
            }
            sign_results.append(sign_info)

            if debug and debug_img is not None:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if debug and debug_img is not None:
        os.makedirs("img/debug/signs", exist_ok=True)
        cv2.imwrite("img/debug/signs/detected_signs.png", debug_img)
        print("[✓] Saved sign-detection debug image to 'img/debug/signs/detected_signs.png'.")

    return sign_results


def recognize_tango_board(image, sign_templates=None, debug=True):
    """
    Main entry point for detecting the board layout, cell content clusters,
    and locating sign symbols (e.g., crosses and equals) using multi-scale template matching.

    Args:
        image (np.ndarray): BGR input of the entire board.
        sign_templates (dict or None):
            A dictionary of named sign images for template matching,
            e.g. { 'cross': cross_img, 'equals': equals_img, ... }
        debug (bool): If True, will write debug images to disk.

    Returns:
        - cell_map: Dictionary mapping (row, col) -> cluster_label for each cell
        - sign_map: List of sign detections (each with sign_label, bounding_box, grid_position)
        - rows: Number of grid rows
        - cols: Number of grid cols
    """
    # Fallback: If user didn't provide sign templates, load them from disk
    if sign_templates is None:
        sign_templates = {
            'cross': cv2.imread("src/games/tango/cross.png"),
            'equals': cv2.imread("src/games/tango/equal.png"),
        }

    # 1) Detect grid lines
    h_lines, v_lines = detect_grid_size_from_faded_lines(image, debug=debug)
    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    # 2) Crop individual cells and cluster them
    cell_images = crop_cell_inners(image, h_lines, v_lines, margin=10)
    cell_labels = classify_images(cell_images, max_clusters=3)
    cell_map = {pos: label for (pos, _), label in zip(cell_images, cell_labels)}

    if debug:
        # Save the cells by cluster (purely for visualization/debug)
        for (pos, img), label in zip(cell_images, cell_labels):
            cluster_dir = f"img/debug/cells/cluster_{label}"
            os.makedirs(cluster_dir, exist_ok=True)
            r, c = pos
            cv2.imwrite(f"{cluster_dir}/cell_{r}_{c}.png", img)

    # 3) Detect signs (crosses, equals, etc.) by multi-scale template matching
    sign_map = []
    if sign_templates is not None:
        print("[i] Detecting sign symbols on the board...")
        sign_map = detect_signs_on_grid(
            image, sign_templates, h_lines, v_lines, threshold=0.8, debug=debug
        )

    return cell_map, sign_map, rows, cols