import cv2
from test_screenshot import test_take_screenshot
from src.comp_vision.preprocess import crop_puzzle_area


def test_preprocessing():
    """
    Initializes the LinkedInGameAgent, navigates to the game URL,
    captures a screenshot, crops the puzzle area, and saves the processed image.
    """
    agent, full_image = test_take_screenshot()

    # Crop only the puzzle area
    cropped = crop_puzzle_area(full_image)

    # Save the cropped image
    cv2.imwrite("img/processed_image.png", cropped)
    print("Cropped puzzle area saved as 'img/processed_image.png'.")

    cropped_img = cv2.imread("img/processed_image.png")

    return agent, cropped_img


if __name__ == '__main__':
    agent, _ = test_preprocessing()
    agent.close()