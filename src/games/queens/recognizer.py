import numpy as np
import cv2
from sklearn.cluster import DBSCAN

def detect_grid_size_from_lines(image, debug=True):
    """
    Detects number of rows and columns using thick black lines in a grid.

    Args:
        image (np.ndarray): BGR puzzle image.
        debug (bool): If True, shows/saves image with drawn lines.

    Returns:
        (int, int): (num_rows, num_cols)
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to extract dark (black) lines
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to isolate lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Sum across axes to get projection profiles
    horizontal_projection = np.sum(horizontal_lines, axis=1)
    vertical_projection = np.sum(vertical_lines, axis=0)

    # Threshold projection to find strong lines only
    h_threshold = np.max(horizontal_projection) * 0.5
    v_threshold = np.max(vertical_projection) * 0.5

    h_indices = np.where(horizontal_projection > h_threshold)[0]
    v_indices = np.where(vertical_projection > v_threshold)[0]

    # Group close lines to single logical lines
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

        cv2.imwrite("img/debug_grid_lines.png", debug_img)
        print("Saved debug image with detected grid lines to 'debug_grid_lines.png'.")

    num_rows = len(horizontal_lines_pos) - 1
    num_cols = len(vertical_lines_pos) - 1

    if num_rows < 1 or num_cols < 1:
        raise ValueError("Could not detect valid grid lines.")

    return num_rows, num_cols

def extract_cell_colors(image, num_rows, num_cols):
    """
    Divides image into cells and extracts average center color (3x3 pixels) per cell.

    Args:
        image (np.ndarray): Cropped BGR image.
        num_rows (int): Number of grid rows.
        num_cols (int): Number of grid columns.

    Returns:
        cell_positions (list of (row, col)),
        cell_colors (np.ndarray of shape (n_cells, 3), RGB colors)
    """
    height, width = image.shape[:2]
    cell_h = height // num_rows
    cell_w = width // num_cols

    cell_positions = []
    cell_colors = []

    for row in range(num_rows):
        for col in range(num_cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell_img = image[y1:y2, x1:x2]

            # Define 3x3 center square
            cy, cx = cell_img.shape[0] // 2, cell_img.shape[1] // 2
            center_patch = cell_img[cy - 1:cy + 2, cx - 1:cx + 2]  # 3x3 patch

            avg_color_bgr = center_patch.mean(axis=(0, 1))
            avg_color_rgb = avg_color_bgr[::-1]  # BGR → RGB

            cell_positions.append((row, col))
            cell_colors.append(avg_color_rgb)


    return cell_positions, np.array(cell_colors)


def cluster_cell_colors(cell_colors, eps=10, min_samples=1):
    """
    Clusters cell colors using DBSCAN (no need to specify n_clusters).

    Args:
        cell_colors (np.ndarray): RGB vectors (shape: N x 3)
        eps (float): Maximum distance between two samples to be considered in the same neighborhood.
        min_samples (int): Minimum number of samples to form a core point.

    Returns:
        labels (np.ndarray): Cluster labels per cell.
        centers (np.ndarray): RGB values for each cluster center.
    """
    # Normalize color values to [0, 1] to make eps scale independent
    normalized = cell_colors / 255.0

    # Fit DBSCAN
    db = DBSCAN(eps=eps / 255.0, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(normalized)

    # Compute cluster centers manually
    unique_labels = sorted(set(labels) - {-1})  # Ignore noise (-1)
    centers = np.array([
        cell_colors[labels == label].mean(axis=0)
        for label in unique_labels
    ])

    # Re-map labels to 0-based cluster IDs
    label_map = {old: new for new, old in enumerate(unique_labels)}
    clean_labels = np.array([label_map[label] if label in label_map else -1 for label in labels])

    return clean_labels, centers


def recognize_maze(image, n_clusters=None):
    """
    Full recognition pipeline: crop borders, detect grid, cluster cell colors.

    Args:
        image (np.ndarray): BGR screenshot of maze area.
        n_clusters (int or None): Optional color cluster count.

    Returns:
        dict: (row, col) → cluster label
        dict: cluster label → RGB color
    """

    rows, cols = detect_grid_size_from_lines(image, debug=True)
    print(f"Detected grid size: {rows} rows × {cols} cols")

    positions, colors = extract_cell_colors(image, rows, cols)
    labels, centers = cluster_cell_colors(colors)  # don't pass n_clusters anymore

    maze_map = {positions[i]: int(labels[i]) for i in range(len(positions))}
    cluster_colors = {i: tuple(map(int, centers[i])) for i in range(len(centers))}

    print(f"Detected color clusters: {len(centers)}")

    return maze_map, cluster_colors