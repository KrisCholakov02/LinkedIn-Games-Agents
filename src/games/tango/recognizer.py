import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN, KMeans


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


def crop_cells_and_edges(image, h_lines, v_lines):
    """
    Crops each cell and edge area (right/bottom) from the grid image.

    Args:
        image (np.ndarray): Input BGR image.
        h_lines (list): Horizontal line positions.
        v_lines (list): Vertical line positions.

    Returns:
        tuple: (list of cell_images, list of right_edge_images, list of bottom_edge_images)
    """
    cell_images = []
    edge_images_right = []
    edge_images_bottom = []

    os.makedirs("img/debug/cells", exist_ok=True)
    os.makedirs("img/debug/edges/right", exist_ok=True)
    os.makedirs("img/debug/edges/bottom", exist_ok=True)

    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]

            # Crop 10 pixels inside each side of the cell
            margin = 10
            y1_inner = min(y1 + margin, y2)
            y2_inner = max(y2 - margin, y1_inner + 1)
            x1_inner = min(x1 + margin, x2)
            x2_inner = max(x2 - margin, x1_inner + 1)

            # Cell content
            cell_img = image[y1_inner:y2_inner, x1_inner:x2_inner]
            cell_images.append(((i, j), cell_img))
            cv2.imwrite(f"img/debug/cells/cell_{i}_{j}.png", cell_img)

            # Right edge
            if j < len(v_lines) - 2:
                edge_r = image[y1:y2, x2:x2 + 6]
                edge_images_right.append(((i, j), edge_r))
                cv2.imwrite(f"img/debug/edges/right/edge_{i}_{j}.png", edge_r)

            # Bottom edge
            if i < len(h_lines) - 2:
                edge_b = image[y2:y2 + 6, x1:x2]
                edge_images_bottom.append(((i, j), edge_b))
                cv2.imwrite(f"img/debug/edges/bottom/edge_{i}_{j}.png", edge_b)

    return cell_images, edge_images_right, edge_images_bottom


def classify_images(image_tuples, n_clusters=3):
    """
    Clusters images based on their normalized grayscale appearance using KMeans.

    Args:
        image_tuples (list of tuple): (metadata, image) tuples.
        n_clusters (int): Fixed number of visual clusters (e.g., Sun, Moon, Empty).

    Returns:
        list: Cluster labels for each image.
    """
    features = []
    for (_, img) in image_tuples:
        resized = cv2.resize(img, (24, 24))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Preprocessing: histogram equalization + blur
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        norm = blurred / 255.0
        features.append(norm.flatten())

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels


def recognize_tango_board(image, debug=True):
    """
    Recognizes a Tango puzzle board:
    - Detects grid structure from faded lines.
    - Crops cell interiors and border regions.
    - Clusters visual elements to classify them.

    Args:
        image (np.ndarray): Input BGR puzzle screenshot.
        debug (bool): Whether to save debug crops/images.

    Returns:
        tuple:
            - cell_map: {(row, col) → cluster_id}
            - edge_map: {((r1, c1), (r2, c2)) → cluster_id}
            - rows: int
            - cols: int
    """
    h_lines, v_lines = detect_grid_size_from_faded_lines(image, debug=debug)
    cell_imgs, edge_imgs_r, edge_imgs_b = crop_cells_and_edges(image, h_lines, v_lines)

    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    cell_labels = classify_images(cell_imgs)
    edge_labels_r = classify_images(edge_imgs_r)
    edge_labels_b = classify_images(edge_imgs_b)

    # Construct final maps
    cell_map = {pos: label for (pos, _), label in zip(cell_imgs, cell_labels)}

    edge_map = {}
    for ((r, c), _), label in zip(edge_imgs_r, edge_labels_r):
        edge_map[((r, c), (r, c + 1))] = label
    for ((r, c), _), label in zip(edge_imgs_b, edge_labels_b):
        edge_map[((r, c), (r + 1, c))] = label

    # --- Save clustered images into directories ---
    if debug:
        for (pos, img), label in zip(cell_imgs, cell_labels):
            cluster_dir = f"img/debug/cells/cluster_{label}"
            os.makedirs(cluster_dir, exist_ok=True)
            r, c = pos
            cv2.imwrite(f"{cluster_dir}/cell_{r}_{c}.png", img)

        for (pos, img), label in zip(edge_imgs_r, edge_labels_r):
            cluster_dir = f"img/debug/edges/right/cluster_{label}"
            os.makedirs(cluster_dir, exist_ok=True)
            r, c = pos
            cv2.imwrite(f"{cluster_dir}/edge_r_{r}_{c}.png", img)

        for (pos, img), label in zip(edge_imgs_b, edge_labels_b):
            cluster_dir = f"img/debug/edges/bottom/cluster_{label}"
            os.makedirs(cluster_dir, exist_ok=True)
            r, c = pos
            cv2.imwrite(f"{cluster_dir}/edge_b_{r}_{c}.png", img)

    return cell_map, edge_map, rows, cols