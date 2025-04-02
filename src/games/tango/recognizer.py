import os
import numpy as np
import cv2
from sklearn.cluster import KMeans

def detect_grid_size_from_faded_lines(image, debug=True):
    """
    Detects the grid size from an image with faded lines.

    Args:
        image (np.ndarray): Input image.
        debug (bool): If True, saves debug images.

    Returns:
        tuple: Lists of y-coordinates of horizontal lines and x-coordinates of vertical lines.
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

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

    h_threshold = np.max(horizontal_projection) * 0.3
    v_threshold = np.max(vertical_projection) * 0.3

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
        cv2.imwrite("img/debug/tango_grid.png", debug_img)
        print("[✓] Saved grid debug image to 'img/debug/tango_grid.png'.")

    if len(horizontal_lines_pos) < 2 or len(vertical_lines_pos) < 2:
        raise ValueError("[✗] Failed to detect valid grid dimensions.")

    return horizontal_lines_pos, vertical_lines_pos

def crop_cell_inners(image, h_lines, v_lines, margin=10):
    """
    Crops the inner parts of cells from the grid.

    Args:
        image (np.ndarray): Input image.
        h_lines (list): Y-coordinates of horizontal lines.
        v_lines (list): X-coordinates of vertical lines.
        margin (int): Margin to apply when cropping.

    Returns:
        list: List of tuples containing cell coordinates and cropped cell images.
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

def determine_optimal_k(features, max_k=3):
    """
    Determines the optimal number of clusters for KMeans.

    Args:
        features (np.ndarray): Feature array.
        max_k (int): Maximum number of clusters to consider.

    Returns:
        int: Optimal number of clusters.
    """
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    deltas = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]
    if len(deltas) == 0:
        return 1
    if len(deltas) == 1 or deltas[1] < 0.1 * deltas[0]:
        return 2
    return 3

def classify_images(image_tuples, max_clusters):
    """
    Classifies images into clusters using KMeans.

    Args:
        image_tuples (list): List of tuples containing cell coordinates and images.
        max_clusters (int): Maximum number of clusters.

    Returns:
        np.ndarray: Array of cluster labels.
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
    Performs multi-scale template matching.

    Args:
        board_gray (np.ndarray): Grayscale board image.
        tmpl_gray (np.ndarray): Grayscale template image.
        threshold (float): Matching threshold.
        scale_factors (list): List of scale factors.

    Returns:
        list: List of bounding boxes for matched regions.
    """
    if scale_factors is None:
        scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    matches = []
    original_w = tmpl_gray.shape[1]
    original_h = tmpl_gray.shape[0]

    for scale in scale_factors:
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        if new_w < 5 or new_h < 5:
            continue

        scaled_tmpl = cv2.resize(tmpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(board_gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            x1, y1 = pt[0], pt[1]
            x2, y2 = x1 + new_w, y1 + new_h
            matches.append((x1, y1, x2, y2))

    return matches

def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    Applies non-maximum suppression to bounding boxes.

    Args:
        boxes (list): List of bounding boxes.
        overlap_thresh (float): Overlap threshold.

    Returns:
        list: List of bounding boxes after suppression.
    """
    if not boxes:
        return []

    boxes_array = np.array(boxes, dtype=float)
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
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

    return [boxes[k] for k in keep]

def detect_signs_on_grid(image, sign_templates, h_lines, v_lines, threshold=0.7, debug=True):
    """
    Detects signs on the grid using template matching.

    Args:
        image (np.ndarray): BGR board image.
        sign_templates (dict): Dictionary of sign templates.
        h_lines (list): Y-coordinates of horizontal lines.
        v_lines (list): X-coordinates of vertical lines.
        threshold (float): Matching threshold.
        debug (bool): If True, saves debug images.

    Returns:
        list: List of dictionaries containing sign information.
    """
    if not sign_templates:
        return []

    board_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_img = image.copy() if debug else None
    sign_results = []

    def clamp_idx(val, upper):
        return max(0, min(val, upper - 1))

    def find_nearest_line(val, lines):
        nearest_idx = None
        nearest_dist = float('inf')
        for i, line_coord in enumerate(lines):
            dist = abs(line_coord - val)
            if dist < nearest_dist:
                nearest_idx = i
                nearest_dist = dist
        return nearest_idx, nearest_dist

    for label, tmpl_bgr in sign_templates.items():
        if tmpl_bgr is None or tmpl_bgr.size == 0:
            print(f"[!] Warning: Template for '{label}' is empty. Skipping.")
            continue

        tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
        raw_boxes = multi_scale_template_match(board_gray, tmpl_gray, threshold=threshold)

        final_boxes = non_max_suppression(raw_boxes, overlap_thresh=0.3)

        for (x1, y1, x2, y2) in final_boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            near_v_idx, dist_v = find_nearest_line(cx, v_lines)
            near_h_idx, dist_h = find_nearest_line(cy, h_lines)

            cell_pairs = []
            line_thresh = 6

            if dist_v < dist_h and dist_v < line_thresh:
                c_left = clamp_idx(near_v_idx - 1, len(v_lines) - 1)
                c_right = clamp_idx(near_v_idx, len(v_lines) - 1)

                row_idx = None
                for i in range(len(h_lines) - 1):
                    if h_lines[i] <= cy < h_lines[i + 1]:
                        row_idx = i
                        break

                if row_idx is not None:
                    cell_pairs.append(((row_idx, c_left), (row_idx, c_right)))

            elif dist_h <= dist_v and dist_h < line_thresh:
                r_top = clamp_idx(near_h_idx - 1, len(h_lines) - 1)
                r_bot = clamp_idx(near_h_idx, len(h_lines) - 1)

                col_idx = None
                for j in range(len(v_lines) - 1):
                    if v_lines[j] <= cx < v_lines[j + 1]:
                        col_idx = j
                        break

                if col_idx is not None:
                    cell_pairs.append(((r_top, col_idx), (r_bot, col_idx)))

            if not cell_pairs:
                cell_pairs = [((None, None), (None, None))]

            sign_info = {
                'sign_label': label,
                'bounding_box': (x1, y1, x2, y2),
                'cell_pairs': cell_pairs
            }
            sign_results.append(sign_info)

            if debug and debug_img is not None:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if debug and debug_img is not None:
        os.makedirs("img/debug/signs", exist_ok=True)
        cv2.imwrite("img/debug/signs/detected_signs.png", debug_img)
        print("[✓] Saved sign-detection debug image to 'img/debug/signs/detected_signs.png'.")

    return sign_results

def assign_cluster_names(cell_images, cell_labels):
    """
    Assigns names to clusters based on average brightness.

    Args:
        cell_images (list): List of tuples containing cell coordinates and images.
        cell_labels (np.ndarray): Array of cluster labels.

    Returns:
        dict: Dictionary mapping cluster labels to names.
    """
    cluster_to_imgs = {}
    for (pos, img), label in zip(cell_images, cell_labels):
        cluster_to_imgs.setdefault(label, []).append(img)

    cluster_brightness = {}
    for label, img_list in cluster_to_imgs.items():
        all_pixels = []
        for bgr in img_list:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            all_pixels.extend(gray.flatten())
        avg_brightness = np.mean(all_pixels)
        cluster_brightness[label] = avg_brightness

    sorted_clusters = sorted(cluster_brightness.items(), key=lambda x: x[1], reverse=True)
    distinct_clusters = len(sorted_clusters)
    name_map = {}

    if distinct_clusters == 1:
        only_label, only_brightness = sorted_clusters[0]
        if only_brightness > 200:
            name_map[only_label] = "empty"
        else:
            name_map[only_label] = "icon1"

    elif distinct_clusters == 2:
        (label0, bright0), (label1, bright1) = sorted_clusters
        name_map[label0] = "empty"
        name_map[label1] = "icon1"

    else:
        (label0, bright0) = sorted_clusters[0]
        (label1, bright1) = sorted_clusters[1]
        (label2, bright2) = sorted_clusters[2]
        name_map[label0] = "empty"
        name_map[label1] = "icon1"
        name_map[label2] = "icon2"

        for extra_label, _ in sorted_clusters[3:]:
            name_map[extra_label] = "icon2"

    return name_map

def recognize_tango_board(image, sign_templates=None, debug=True):
    """
    Recognizes the Tango board layout and cell content.

    Args:
        image (np.ndarray): BGR input of the entire board.
        sign_templates (dict or None): Dictionary of sign templates.
        debug (bool): If True, saves debug images.

    Returns:
        tuple: Cell map, sign map, number of rows, and number of columns.
    """
    if sign_templates is None:
        sign_templates = {
            'cross': cv2.imread("src/games/tango/cross.png"),
            'equals': cv2.imread("src/games/tango/equal.png"),
        }

    h_lines, v_lines = detect_grid_size_from_faded_lines(image, debug=debug)
    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    cell_images = crop_cell_inners(image, h_lines, v_lines, margin=10)
    cell_labels = classify_images(cell_images, max_clusters=3)
    cluster_name_map = assign_cluster_names(cell_images, cell_labels)

    cell_map = {}
    for ((r, c), _), label in zip(cell_images, cell_labels):
        cell_map[(r, c)] = cluster_name_map[label]

    if debug:
        for (pos, img), cluster_id in zip(cell_images, cell_labels):
            cluster_dir = f"img/debug/cells/{cluster_name_map[cluster_id]}"
            os.makedirs(cluster_dir, exist_ok=True)
            r, c = pos
            cv2.imwrite(f"{cluster_dir}/cell_{r}_{c}.png", img)

    sign_map = []
    if sign_templates:
        print("[i] Detecting sign symbols on the board...")
        sign_map = detect_signs_on_grid(image, sign_templates, h_lines, v_lines, threshold=0.8, debug=debug)

    return cell_map, sign_map, rows, cols