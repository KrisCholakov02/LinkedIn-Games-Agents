import numpy as np
import cv2
from sklearn.cluster import KMeans

from src.comp_vision.preprocess import crop_white_border


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

    return num_rows, num_cols

def extract_cell_colors(image, num_rows, num_cols):
    """
    Divides image into cells and extracts average color per cell.

    Args:
        image (np.ndarray): Cropped BGR image.
        grid_size (int): Grid dimension (e.g. 8 for 8x8).

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
            avg_color_bgr = cell_img.mean(axis=(0, 1))
            avg_color_rgb = avg_color_bgr[::-1]  # Convert BGR to RGB

            cell_positions.append((row, col))
            cell_colors.append(avg_color_rgb)

    return cell_positions, np.array(cell_colors)


def cluster_cell_colors(cell_colors, n_clusters=None):
    """
    Clusters cell colors into groups using KMeans.
    If n_clusters is None, it tries to guess optimal number using elbow method fallback.

    Args:
        cell_colors (np.ndarray): RGB vectors of all cells.
        n_clusters (int or None): Number of clusters to form.

    Returns:
        labels (np.ndarray): Cluster labels for each cell.
        centers (np.ndarray): Cluster center RGBs.
    """
    if n_clusters is None:
        # Estimate number of clusters using simple heuristic
        # You can implement full elbow method or silhouette later
        n_clusters = min(10, len(cell_colors) // 2)

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(cell_colors)
    centers = kmeans.cluster_centers_

    return labels, centers


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
    cropped = crop_white_border(image, debug=True)

    rows, cols = detect_grid_size_from_lines(cropped, debug=True)
    print(f"Detected grid size: {rows} rows × {cols} cols")

    positions, colors = extract_cell_colors(cropped, rows, cols)
    labels, centers = cluster_cell_colors(colors, n_clusters)

    maze_map = {positions[i]: int(labels[i]) for i in range(len(positions))}
    cluster_colors = {i: tuple(map(int, centers[i])) for i in range(len(centers))}

    return maze_map, cluster_colors