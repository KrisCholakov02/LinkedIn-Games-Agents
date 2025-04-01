# 🧠 LinkedIn Game Agents: Queens, Tango & Zip

A fully automated puzzle-solving agent for the **LinkedIn Games** 🧩:  
- 👑 **Queens**
- ☀️🌙 **Tango**
- 🗺️ **Zip**

Built using Python, OpenCV, machine learning, and Selenium, this project demonstrates how computer vision and algorithmic reasoning can be combined to solve visual logic puzzles automatically.

---

## 📁 Project Structure
```plaintext
src/
├── core/
│   ├── base_game_agent.py     
│   └── game_registry.py       
│
├── games/
│   ├── queens/
│   │   ├── agent.py
│   │   ├── main.py
│   │   ├── placer.py
│   │   ├── recognizer.py
│   │   └── solver.py
│   │
│   ├── tango/
│   │   ├── agent.py
│   │   ├── main.py
│   │   ├── placer.py
│   │   ├── recognizer.py
│   │   ├── solver.py
│   │   ├── cross.png
│   │   └── equal.png
│   │
│   └── zip/
│       ├── agent.py         
│       ├── main.py
│       ├── placer.py        
│       ├── recognizer.py    
│       └── solver.py        
│
├── utils/
│   ├── screenshot.py
│   └── main.py                
│
├── config.py                  
├── .gitignore
└── README.md
```
---

## 🔍 Computer Vision Techniques

The agents leverage several state-of-the-art computer vision strategies:

### 👑 Queens
- **Grid Detection:** Uses edge filters and Hough Transform.
- **Color Clustering:** Uses **DBSCAN** to group cells of the same color, ensuring that each queen is placed in its own color zone.
- **Constraints:** One queen per color cluster, with rules to prevent queens from being in the same row, column, or adjacent (diagonal) positions.

### ☀️🌙 Tango
- **Faded Grid Detection:** Employs morphological operations and adaptive thresholding to extract partially faded grid lines.
- **Cell Extraction & Clustering:** Precisely crops cell interiors and applies **KMeans clustering** to determine cell types (empty, icon1, icon2).
- **Sign Detection:** Uses multi-scale template matching combined with non-maximum suppression to detect signs (such as equals and cross), which impose constraints between adjacent cells.
- **Backtracking Solver:** Enforces row/column quotas and prevents three consecutive identical icons, while respecting sign constraints.

### 🗺️ Zip
- **Adaptive Grid Detection:** Uses adaptive thresholding and morphological operations to reliably detect greyish grid lines even under non-uniform lighting.
- **Dynamic OCR:** Converts each cell image to a strict binary (0 or 255) image and uses **pytesseract** to dynamically recognize digits. Cells are marked as "empty" if no digit is detected.
- **Wall Detection:** Samples the inner border regions between cells; if a border region has a very low average intensity, it is marked as a wall. These walls restrict movement between cells.
- **Hamiltonian Path Solver:** Finds a path that visits every cell exactly once, starting at the cell labeled "1" and ending at the cell with the highest number. The path must respect the order of numbers in pre-filled cells and cannot cross through walls.

---

## 🚀 How to Run

Ensure you have Python 3.8+ installed and run:

```bash
PYTHONPATH=. python src/main.py --game <game_name> --username <email> --password <password>
```

Replace `<game_name>` with one of the following: `queens`, `tango`, or `zip`.

Replace `<email>` and `<password>` with your LinkedIn credentials.

---

Happy Puzzle Solving! 🚀🎉