# 🧠 LinkedIn Game Agents: Queens & Tango

A fully automated puzzle-solving agent for the **LinkedIn Games** 🧩:  
- 👑 **Queens**
- 🔄 **Tango**

Built using Python, OpenCV, machine learning, and Selenium, this project demonstrates how computer vision and algorithmic reasoning can be combined to solve visual logic puzzles automatically.

---

## 📁 Project Structure

```
src/
├── core/
│   ├── base_game_agent.py     # Abstract interface for game agents
│   └── game_registry.py       # Game loader & registry
│
├── games/
│   ├── queens/
│   │   ├── agent.py
│   │   ├── main.py
│   │   ├── placer.py
│   │   ├── recognizer.py
│   │   └── solver.py
│   │
│   └── tango/
│       ├── agent.py
│       ├── main.py
│       ├── placer.py
│       ├── recognizer.py
│       ├── solver.py
│       ├── cross.png
│       └── equal.png
│
├── utils/
│   ├── screenshot.py
│   └── main.py                # Entry point for running games
│
├── config.py                  # Game and Selenium config
├── .gitignore
└── README.md
```

---

## 🔍 Computer Vision Techniques

The agents rely on multiple computer vision strategies:

### 👑 Queens
- Grid line detection using edge filters and Hough Transform.
- **DBSCAN clustering** is applied to group cells of the same color (segment), ensuring each queen is placed in a separate color zone.
- Puzzle constraints:
  - One queen per color cluster.
  - No two queens in the same row, column, or adjacent square (diagonals included).

### 🔄 Tango
- Faded grid detection using **morphological operations** and **adaptive thresholding**.
- Cell contents are extracted using precise inner-region cropping.
- **KMeans clustering** (1–3 clusters) is used to identify:
  - Empty cells
  - Icon 1
  - Icon 2
- Sign symbols (crosses, equals) are recognized using **multi-scale template matching** + non-max suppression.
- The solver performs **backtracking with constraint propagation**:
  - Row/column quotas (equal number of icons)
  - No three consecutive identical icons
  - Sign constraints (equal/cross logic)

---

## 🚀 How to Run

Ensure Python 3.8+ and `pip install -r requirements.txt`.

To run the agent for a given game, use:

```bash
PYTHONPATH=. python src/main.py --game <game_name> --username <email> --password <password>
```
