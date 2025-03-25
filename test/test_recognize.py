from src.comp_vision.recognize import recognize_maze
from test_screenshot import test_take_screenshot

def test_dynamic_recognition():
    agent, board_img = test_take_screenshot()

    # Recognize the puzzle
    maze_map, cluster_colors = recognize_maze(board_img)

    for pos, label in maze_map.items():
        print(f"{pos}: Cluster {label}, Color {cluster_colors[label]}")

    return agent, maze_map, cluster_colors


if __name__ == '__main__':
    agent, _, _ = test_dynamic_recognition()
    agent.close()
