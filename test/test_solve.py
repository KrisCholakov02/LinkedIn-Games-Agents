from test_recognize import test_dynamic_recognition
from src.web_automation.solve import solve_queens

def test_solve():
    agent, maze_map, cluster_colors = test_dynamic_recognition()

    # Solve the puzzle
    queen_positions = solve_queens(maze_map, num_clusters=len(cluster_colors))

    print("Solved queen positions:")
    for pos in queen_positions:
        print(pos)

    return agent, queen_positions

if __name__ == '__main__':
    agent, _ = test_solve()
    agent.close()
