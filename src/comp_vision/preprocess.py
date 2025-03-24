import cv2
import numpy as np


def crop_puzzle_area(image):
    """
    Crops and returns only the upper puzzle region from the given screenshot.

    Args:
        image (numpy.ndarray): The original screenshot as a BGR OpenCV image.

    Returns:
        numpy.ndarray: Cropped image containing only the puzzle region.
    """

    # The coordinates below (y_start:y_end, x_start:x_end) are placeholders.
    # Adjust them based on your puzzle's position in the screenshot.
    # For example, if your puzzle is near the top-left corner and
    # extends to about half the image, you might do something like:

    y_start, y_end = 140, 550   # vertical range (top to bottom)
    x_start, x_end = 390, 800   # horizontal range (left to right)

    # Perform the crop
    cropped = image[y_start:y_end, x_start:x_end]

    return cropped

def crop_white_border(image, threshold=240, debug=False):
    """
    Crops white borders from an image based on a brightness threshold.

    Args:
        image (np.ndarray): Original BGR image.
        threshold (int): Pixel intensity threshold to consider as "white".
        debug (bool): If True, saves debug mask and cropped preview.

    Returns:
        np.ndarray: Cropped image without white borders.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask of non-white pixels
    mask = gray < threshold

    # Find bounding box of non-white area
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        raise ValueError("No non-white region detected.")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # Add 1 to include end pixel

    cropped = image[y0:y1, x0:x1]

    if debug:
        debug_mask = (mask * 255).astype(np.uint8)
        cv2.imwrite("img/debug_white_mask.png", debug_mask)
        cv2.imwrite("img/debug_cropped_maze_area.png", cropped)
        print("Saved debug mask and cropped maze area.")

    return cropped