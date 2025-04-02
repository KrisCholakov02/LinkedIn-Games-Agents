import numpy as np
import cv2
from sklearn.cluster import DBSCAN


def detect_grid_size_from_lines(image, debug=True):
    """
    Detects the number of rows and columns in a grid using thick black lines.

    Args:
        image (np.ndarray): BGR puzzle image.
        debug (bool): If True, saves a debug image with detected lines drawn.

    Returns:
        tuple: (num_rows, num_cols)

    Raises:
        ValueError: If no valid grid lines are detected.
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to isolate grid lines
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
        cv2.imwrite("img/debug_grid_lines.png", debug_img)
        print("Saved debug image with detected grid lines to 'debug_grid_lines.png'.")

    num_rows = len(horizontal_lines_pos) - 1
    num_cols = len(vertical_lines_pos) - 1

    if num_rows < 1 or num_cols < 1:
        raise ValueError("Could not detect valid grid lines.")

    return num_rows, num_cols


def extract_cell_colors(image, num_rows, num_cols):
    """
    Divides the grid into cells and extracts the average center color of each.

    Args:
        image (np.ndarray): BGR image of the board.
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.

    Returns:
        tuple:
            - list of (row, col): Cell positions.
            - np.ndarray: RGB colors of each cell (shape: [n_cells, 3]).
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

            cy, cx = cell_img.shape[0] // 2, cell_img.shape[1] // 2
            center_patch = cell_img[cy - 1:cy + 2, cx - 1:cx + 2]  # 3x3 patch

            avg_color_bgr = center_patch.mean(axis=(0, 1))
            avg_color_rgb = avg_color_bgr[::-1]  # Convert BGR → RGB

            cell_positions.append((row, col))
            cell_colors.append(avg_color_rgb)

    return cell_positions, np.array(cell_colors)


def cluster_cell_colors(cell_colors, eps=10, min_samples=1):
    """
    Clusters cell colors using DBSCAN to group similar-colored cells.

    Args:
        cell_colors (np.ndarray): RGB values of cells (shape: N x 3).
        eps (float): Distance threshold for clustering (0–255 scale).
        min_samples (int): Minimum number of samples to form a cluster.

    Returns:
        tuple:
            - np.ndarray: Cluster label per cell (shape: N).
            - np.ndarray: RGB values for each cluster center (shape: K x 3).
    """
    normalized = cell_colors / 255.0  # Normalize to [0, 1] for DBSCAN

    db = DBSCAN(eps=eps / 255.0, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(normalized)

    unique_labels = sorted(set(labels) - {-1})  # Exclude noise
    centers = np.array([
        cell_colors[labels == label].mean(axis=0)
        for label in unique_labels
    ])

    label_map = {old: new for new, old in enumerate(unique_labels)}
    clean_labels = np.array([label_map[label] if label in label_map else -1 for label in labels])

    return clean_labels, centers


def recognize_maze(image, n_clusters=None):
    """
    Complete recognition pipeline for the Queens puzzle board.

    Steps:
        1. Detect grid size from visual lines.
        2. Extract average color from each grid cell.
        3. Cluster cells based on color similarity.

    Args:
        image (np.ndarray): BGR screenshot of the puzzle area.
        n_clusters (int, optional): Deprecated; DBSCAN determines clusters automatically.

    Returns:
        tuple:
            - dict: Mapping from (row, col) to cluster ID.
            - dict: Mapping from cluster ID to representative RGB color.
    """
    rows, cols = detect_grid_size_from_lines(image, debug=True)
    print(f"Detected grid size: {rows} rows × {cols} cols")

    positions, colors = extract_cell_colors(image, rows, cols)
    labels, centers = cluster_cell_colors(colors)

    maze_map = {positions[i]: int(labels[i]) for i in range(len(positions))}
    cluster_colors = {i: tuple(map(int, centers[i])) for i in range(len(centers))}

    print(f"Detected color clusters: {len(centers)}")

    return maze_map, cluster_colors