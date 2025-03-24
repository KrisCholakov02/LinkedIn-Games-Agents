from test_preprocessing import test_preprocessing
from src.comp_vision.recognize import recognize_maze


def test_dynamic_recognition():
    agent, cropped_img = test_preprocessing()

    # Recognize the puzzle
    maze_map, cluster_colors = recognize_maze(cropped_img)

    for pos, label in maze_map.items():
        print(f"{pos}: Cluster {label}, Color {cluster_colors[label]}")

    return agent, maze_map, cluster_colors


if __name__ == '__main__':
    agent, _, _ = test_dynamic_recognition()
    agent.close()
