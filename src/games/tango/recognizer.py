import os
import numpy as np
import cv2
from sklearn.cluster import KMeans


def detect_grid_size_from_faded_lines(image, debug=True):
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


def crop_inner_edges(image, h_lines, v_lines, pad=10, cut=5):
    edge_images = []
    os.makedirs("img/debug/edges", exist_ok=True)
    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    for i in range(rows):
        for j in range(cols):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]

            # Right edge
            if j < cols - 1:
                x_edge = v_lines[j + 1]
                y_top = y1 + pad
                y_bot = y2 - pad
                edge = image[y_top:y_bot, x_edge - 3:x_edge + 3]
                edge = edge[cut:-cut, :]
                edge_images.append((((i, j), (i, j + 1)), edge))

            # Bottom edge
            if i < rows - 1:
                y_edge = h_lines[i + 1]
                x_left = x1 + pad
                x_right = x2 - pad
                edge = image[y_edge - 3:y_edge + 3, x_left:x_right]
                edge = edge[:, cut:-cut]
                edge = cv2.rotate(edge, cv2.ROTATE_90_CLOCKWISE)
                edge = cv2.flip(edge, 1)  # Flip horizontally
                edge_images.append((((i, j), (i + 1, j)), edge))

    return edge_images


def determine_optimal_k(features, max_k=3):
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


def recognize_tango_board(image, debug=True):
    h_lines, v_lines = detect_grid_size_from_faded_lines(image, debug=debug)
    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    # --- Cells ---
    cell_images = crop_cell_inners(image, h_lines, v_lines, margin=10)
    cell_labels = classify_images(cell_images, max_clusters=3)
    cell_map = {pos: label for (pos, _), label in zip(cell_images, cell_labels)}

    if debug:
        for (pos, img), label in zip(cell_images, cell_labels):
            cluster_dir = f"img/debug/cells/cluster_{label}"
            os.makedirs(cluster_dir, exist_ok=True)
            r, c = pos
            cv2.imwrite(f"{cluster_dir}/cell_{r}_{c}.png", img)

    # --- Edges ---
    edge_images = crop_inner_edges(image, h_lines, v_lines, pad=10, cut=5)
    edge_labels = classify_images(edge_images, max_clusters=3)
    edge_map = {pos: label for (pos, _), label in zip(edge_images, edge_labels)}

    if debug:
        for (pos, img), label in zip(edge_images, edge_labels):
            cluster_dir = f"img/debug/edges/cluster_{label}"
            os.makedirs(cluster_dir, exist_ok=True)
            r1, c1 = pos[0]
            r2, c2 = pos[1]
            cv2.imwrite(f"{cluster_dir}/edge_{r1}_{c1}__{r2}_{c2}.png", img)

    return cell_map, edge_map, rows, cols