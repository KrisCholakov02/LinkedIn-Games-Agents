import os
import numpy as np
import cv2
from sklearn.cluster import KMeans


def detect_grid_size_from_faded_lines(image, debug=True):
    """
    Detects faded grid lines in the Tango puzzle board (horizontal and vertical)
    and returns their pixel coordinates.

    Args:
        image (np.ndarray): Input BGR image.
        debug (bool): Whether to save a debug image showing detected lines.

    Returns:
        tuple: (horizontal_line_positions, vertical_line_positions)
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
    Crops inner content of each cell by applying margin on all sides.

    Args:
        image (np.ndarray): Input BGR image.
        h_lines (list): Horizontal line positions.
        v_lines (list): Vertical line positions.
        margin (int): Padding from all sides (default 10 pixels).

    Returns:
        list: List of (position, cell_image) tuples.
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
    Automatically choose optimal number of clusters using simplified elbow method.

    Args:
        features (np.ndarray): Flattened grayscale features.
        max_k (int): Maximum clusters to test.

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


def classify_cells(cell_images, max_clusters=3):
    """
    Clusters the cells into 1–3 groups (e.g., empty, moon, sun).

    Args:
        cell_images (list): List of (pos, img) tuples.

    Returns:
        list: Cluster labels per cell.
    """
    features = []
    for (_, img) in cell_images:
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


def recognize_tango_board(image, debug=True):
    """
    Recognizes only the inner cells (Sun, Moon, Empty) of a Tango board.

    Args:
        image (np.ndarray): BGR screenshot of the puzzle.
        debug (bool): Whether to save debug output.

    Returns:
        tuple:
            - cell_map: {(row, col): cluster_id}
            - rows: int
            - cols: int
    """
    h_lines, v_lines = detect_grid_size_from_faded_lines(image, debug=debug)
    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    cell_images = crop_cell_inners(image, h_lines, v_lines, margin=10)
    cell_labels = classify_cells(cell_images)

    cell_map = {pos: label for (pos, _), label in zip(cell_images, cell_labels)}

    if debug:
        for (pos, img), label in zip(cell_images, cell_labels):
            cluster_dir = f"img/debug/cells/cluster_{label}"
            os.makedirs(cluster_dir, exist_ok=True)
            r, c = pos
            cv2.imwrite(f"{cluster_dir}/cell_{r}_{c}.png", img)

    return cell_map, rows, cols