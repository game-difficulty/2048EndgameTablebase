# 2048EndgameTablebase: The Ultimate 2048 AI & Tablebase Solution

This project offers the fastest and most space-efficient tablebase generator available, enabling players and developers to explore the theoretical limits of the game. Our AI achieves unprecedented success rates, reaching the **32,768 tile at 86.1%** and the **65,536 tile at 8.4%** under no-undo conditions.


success rate of getting a 2048 in 12 spaces (32k endgame)

![Practise](<img width="1275" height="747" alt="image" src="https://github.com/user-attachments/assets/ed5312e5-5000-452f-a583-ccefc4712ec5" />
)

---

## 🚀 Core Highlights

* **Lightning-Fast Computation**: Utilizes highly optimized bitboard logic and advanced pruning. Compute large-scale tables up to 10x faster than traditional methods.
* **Storage Efficiency**: Multiple data compression and pruning techniques significantly reduce the disk footprint of your endgame tables.
* **State-of-the-Art AI**: Outperforms all contemporary 2048 AIs by utilizing pre-calculated global optimal solutions to navigate the most difficult endgame stages.
* **Complete Training Toolkit**: A comprehensive GUI for real-time practice, "Mistake Notebook" tracking, and deep replay analysis (supporting *2048verse.com* imports).

---

## 🛠️ System Requirements

* **OS**: Windows 7 or later (64-bit).
* **Memory**: 
    * **Basic**: 8GB RAM.
    * **Mega-Tables**: 32GB–128GB+ RAM (Significant virtual memory/paging file on SSD is recommended).
* **Disk Space**: 10GB for basic use; 1TB+ SSD recommended for computing advanced tables like `free12`.
* **CPU**: Any 64-bit CPU (AVX-512 support provides a significant performance boost).
* **GPU**: Not required; the engine is pure CPU-based.

---

## 📖 Quick Start Guide

### 1. Calculate Your First Table
To use the AI or Practice modes, you must first generate a **Table**.
1. Navigate to **Settings**.
2. Select a **Table Name** (e.g., `L3` or `442`) and a **Target Number** (e.g., `256`).
3. Specify a **Save Path** on your SSD.
4. Click **BUILD** to initiate the generation process.

### 2. Learn the Strategy
1. Go to **Practice**.
2. Select your calculated table and ensure the file path is correct.
3. Use the **Auto-Demo** or **Step** features to observe the optimal move sequences.
4. Analyze success rates displayed for all four directions to understand move priority.

### 3. Real-Time Training
1. Go to **Test**.
2. Select your table and execute moves (WASD or Arrow keys).
3. Review the **Real-Time Analytics** on the right side to see your "Match Rate" against the optimal solution.
4. Lower-accuracy moves are automatically logged in the **Mistake Notebook** for later review.

---

## 📈 AI Performance Statistics

Tested over 1,200 games (Adaptive Search Depth 1~9, no undo):

| Milestone | Success Rate | Comparable Score |
| :--- | :--- | :--- |
| **8k** | 100% | 97,000 |
| **16k** | 99.9% | 210,000 |
| **32k** | 86.1% | 453,000 |
| **65k** | 8.4% | 971,000 |

### Table Usage Distribution
The AI dynamically dispatches between search and pre-calculated tables:

| Search Mode | free12-2k | free11-2k | 4442f-2k | free11-512 |
| :--- | :--- | :--- | :--- | :--- |
| 26.49% | 13.38% | 1.41% | 51.50% | 7.22% |

---

## 🤝 Support & Feedback

* **Community**: 
    * **QQ Group**: 94064339
    * **Discord**: 2048 Runs

