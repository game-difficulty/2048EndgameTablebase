# 2048EndgameTablebase: The Ultimate 2048 AI & Tablebase Solution

**2048EndgameTablebase** is the world's fastest and most space-efficient tablebase solution, designed to push the theoretical boundaries of 2048. By integrating massive pre-computed tables, our AI achieves groundbreaking success rates under **no-undo** conditions:

* **32,768 Tile**: 86.1% success rate
* **65,536 Tile**: 8.4% success rate

Even in its **Standalone Configuration** (with a footprint within 3MB), the engine achieves the **32,768 tile with an 80% success rate**. Due to its extreme efficiency and compact size, this version is even deployable on static web pages ([Experience it here](https://game-difficulty.github.io/2048EndgameTablebase/)). This sets a new benchmark for 2048 AI engines.

---

## Demo

success rate of getting a 4096 in 12 spaces (65k endgame)

https://github.com/user-attachments/assets/c105d4e2-8696-423c-9a76-0a130a2d6960


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

average score 772,353

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

## 🧠 Standalone Agent: Engineering & Algorithm

### 1. Transposition Table (Cache)
* **64-bit Double-Hash Compression:** We abandoned the standard 128-bit cache entries. By compressing the signature, score, depth, and sub-tree effort into a single 64-bit integer, we effectively double L1/L2 cache efficiency and shatter the memory bandwidth bottleneck during deep searches.
* **Effort-Based Replacement Strategy:** Instead of a naive constant replacement policy, we introduce hash buckets and adopt a replacement strategy based on node sub-tree workload. This preserves the most computationally expensive nodes, drastically reducing redundant calculations in massive branches.
* **Dynamic Bucket Allocation:** The number of active hash buckets scales dynamically with search depth, significantly reducing the overhead of clearing the cache between moves in shallow searches.

### 2. Advanced Pruning & Search Dynamics
* **Max-Layer over Probability Pruning:** Traditional 2048 engines prune based on the diminishing probability of spawning '4's, which creates a logical contradiction with Transposition Table hit conditions (where high-depth entries inherently have lower cumulative probabilities). We pioneered a `max_layer` constraint. By searching the rare '4' branches first and restricting them via max-layer, we aggressively populate the cache, allowing the subsequent '2' branches to achieve massive cache hit rates.
* **Context-Aware Depth Reduction:** If the board is deemed "absolutely safe" (many empty slots and locked large tiles), the effective search depth is dramatically reduced, saving immense computational power.
* **Catastrophic Displacement Pruning:** Leveraging deep game knowledge, the engine strictly pre-evaluates at the root whether structural disruption is acceptable. If not, branches leading to large tile displacement or chaotic formations are immediately hard-pruned.

### 3. Deep-Knowledge Heuristic Evaluation
Our evaluation function fundamentally breaks the "rules" taught to novice players:
* **Embracing Non-Monotonicity (T-Formations):** Strict monotonicity often traps traditional AIs. Our engine dynamically permits non-monotonic formations and temporary disorder, granting it the flexibility needed to fix broken layouts. It also recognizes that having *too many* empty spaces can be detrimental to controlling where new tiles spawn.
* **The "0xF" Abstraction for Large Tiles:** Above 512/1024, the exact value of a large tile matters far less than its relative position. We universally mask massive tiles as unmergeable entities (`0xF`). This aligns perfectly with endgame tablebase logic.
* **Phase-Dynamic Weights:** The AI's preference for certain formations and its penalty for "death states" shift dynamically based on the current game phase and board characteristics, utilizing bitwise masks to evaluate massive tile structures instantly.

### 4. Endgame Tablebase Distillation
To guide the standalone AI through difficult endgame states, we distilled L3f tables into a hyper-compressed 2.5MB footprint:
* **Monte Carlo Extraction:** We simulated over ten million games to identify the hundreds of thousands of most frequently encountered endgame states.
* **Perfect Hashing Compression:** By utilizing layered perfect hashing, each state retains only a `uint16` success rate and a `uint8` double-hash verification. The final cost is 3-4 bytes per state.
* **Verification Search:** Upon entering the endgame, the AI consults this 2.5MB table. If a match occurs, it performs a rapid search validation. If verified, the AI executes the move; otherwise, it falls back to iterative deepening.

---

## ⚡ Tablebase Generation: Algorithmic Innovations

Computing endgame tables like `free12` involves state spaces reaching $10^{13}$ to $10^{14}$ nodes. To conquer this combinatorial explosion, our engine transcends standard Breadth-First Search (BFS) and Dynamic Programming (DP), introducing an "Advanced Algorithm" that calculates massive tables 10x faster while reducing peak memory by an order of magnitude.

### 1. State Space Compression & Abstraction
* **Bitboard representation & Symmetry Canonicalization** .
* **Large Number Masking:** Large tiles ($\ge 64$) not immediately involved in merges are abstracted as `0xF` placeholders. A single "masked position" acts as a vector representing thousands of actual board states, compressing the BFS layers by $10\times$–$100\times$ and delivering a 10x boost in throughput. Unmasking is performed **lazily** and **partially**. A masked board is only "exposed" (restored to a specific state vector) when a newly merged tile triggers a potential collision with a large number

### 2. Heuristic Bounding & Pruning
* **Small Tile Sum Limit (STSL):** The engine aggressively prunes branches where the sum of "small tiles" ($\le 32$) exceeds a strict threshold. It recognizes that overly cluttered, fragmented boards are seldom optimal and are unworthy of computation.
* **Large Tile Combination Constraints:** The engine restricts the types and maximum counts of co-existing large tiles (e.g., pruning branches with three `128` tiles). This filters out strategically divergent configurations that rarely emerge under optimal play.

### 3. Vectorized Backward DP Solving (Batch Solving)
* **Eliminating the Binary Search Bottleneck:** In standard DP, calculating move expectations for a state expanding into `n` unmasked positions requires `n` independent binary searches against massive arrays (which is still better than querying a giant hash table). Batch Solving radically changes this. The engine resolves all $n$ unmasked success rates for a masked position using only **~1.x binary searches on average**.
* **Transformation Encoding & Permutation Caching:** When a move and canonicalization are applied to a masked position, the internal sequence of its `n` unmasked states gets scrambled. The engine tracks this by assigning temporary labels to the mask bits, tracing how they are displaced to generate a unique permutation signature (`ind`).  Using this `ind` signature, the engine queries a hash permutation table to retrieve a pre-calculated index mapping. This allows the engine to instantly map the scrambled success rate array back to its correct sorted order in one sweep. The time complexity per batch is slashed from a heavy O(n log n) down to O(n) with a small constant factor.

> **Note:** For detailed technical specifications of the table generation logic, please refer to the `help.md` file included in the repository.

---

## 🤝 Support & Feedback

* **Community**: 
    * **QQ Group**: 94064339
    * **Discord**: 2048 Runs

---

## Build From Source

The repository now separates backend Python code, the `engine_core/` runtime layer, the Vue frontend, and the native `native_core/` workspace.

### Linux

The shortest path on Linux is the root helper script:

```bash
chmod +x build_linux.sh
./build_linux.sh
```

To build and launch immediately:

```bash
./build_linux.sh --run
```

Linux prerequisites:

- Python 3.8+
- Node.js + npm
- `cmake`
- `meson`
- `g++`
- OpenMP support in the toolchain/runtime
- `7z` when `native_core/egtb_data.7z` is present
- A `pywebview` system backend such as GTK/WebKit2 or Qt
- Python packages from `requirements.txt`
- `nanobind`

Recommended setup:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install nanobind
```

What `build_linux.sh` does:

1. Builds the Vue frontend into `frontend/dist`
2. Runs `native_core/make.sh` to:
   - extract `native_core/egtb_data.7z` into `native_core/src/`
   - build `ai_core`
   - build `mover_core`
   - build `formation_core`
   - build `bookgen_native.so`
3. Verifies the Python imports required to launch the app

Notes for manual native builds:

- `native_core` test executables are disabled by default.
- The normal app build does not require `_test` sources.
- If you intentionally enable `-DNATIVE_BUILD_TESTS=ON`, make sure the optional
  test sources under `native_core/tests/` are present in your source tree.

After the build finishes, you can run the desktop app manually:

```bash
python3 backend_server.py
```

If you only need part of the build:

```bash
./build_linux.sh --skip-frontend
./build_linux.sh --skip-native
```

### Windows

The repository also contains the full Windows source layout. Build the frontend with:

```powershell
cd frontend
npm install
npm run build
```

Then build the native modules from `native_core/` and run:

```powershell
python backend_server.py
```





