import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN


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


def crop_cells_and_edges(image, h_lines, v_lines):
    cell_images = []
    edge_images_right = []
    edge_images_bottom = []

    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i + 1]
            x1, x2 = v_lines[j], v_lines[j + 1]
            cell_img = image[y1:y2, x1:x2]
            cell_images.append(((i, j), cell_img))

            if j < len(v_lines) - 2:
                edge = image[y1:y2, x2:x2 + 6]
                edge_images_right.append(((i, j), edge))
            if i < len(h_lines) - 2:
                edge = image[y2:y2 + 6, x1:x2]
                edge_images_bottom.append(((i, j), edge))

    return cell_images, edge_images_right, edge_images_bottom


def classify_images(image_tuples, eps=10):
    features = []
    for (_, img) in image_tuples:
        resized = cv2.resize(img, (20, 20))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm = gray / 255.0
        features.append(norm.flatten())

    db = DBSCAN(eps=eps / 255.0, min_samples=1, metric='euclidean')
    labels = db.fit_predict(features)
    return labels


def recognize_tango_board(image, debug=True):
    h_lines, v_lines = detect_grid_size_from_faded_lines(image, debug=debug)
    cell_imgs, edge_imgs_r, edge_imgs_b = crop_cells_and_edges(image, h_lines, v_lines)

    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    cell_labels = classify_images(cell_imgs)
    edge_labels_r = classify_images(edge_imgs_r)
    edge_labels_b = classify_images(edge_imgs_b)

    cell_map = {pos: label for (pos, _), label in zip(cell_imgs, cell_labels)}

    edge_map = {}
    for ((r, c), _), label in zip(edge_imgs_r, edge_labels_r):
        edge_map[((r, c), (r, c + 1))] = label
    for ((r, c), _), label in zip(edge_imgs_b, edge_labels_b):
        edge_map[((r, c), (r + 1, c))] = label

    return cell_map, edge_map, rows, cols
