# User Manual

**Author**: game_difficulty  
**Version**: 9.0  
**Date**: 2026.1.19  

---

# 1. Overview

This is a comprehensive 2048 game training software offering numerous features including table calculation, endgame training, game analysis, AI assistance, modified mini-games, and more.

## 1.1 Key Highlights
- **Lightning-Fast Computation**: Advanced optimization algorithms and sophisticated pruning techniques deliver superior calculation speeds, with large table computations accelerated over 10x.
- **Storage Efficiency**: Multiple pruning and data compression techniques significantly reduce table file disk space.
- **Strongest AI**: Built-in optimal 2048 AI achieves 8.4% (±1.6%) success rate for 65536 and 86.1% (±2.0%) for 32768, far outperforming other AIs.
- **Complete Toolkit**: Comprehensive training and analysis tools including real-time feedback, mistake notebook, and replay analysis to help users master game strategies.

## 1.2 Main Function Modules
- **Game**: Standard 2048 gameplay with built-in strongest AI assistance.
- **Practice**: View tables and learn optimal moves, supporting recording and playback.
- **Test**: Assess your mastery, automatically track mistakes, support replay analysis and Verse replay import.
- **MiniGame**: Diverse modified games offering additional challenges.
- **Settings**: Global configuration, table calculation, advanced algorithm setup.
- **Help**: Documentation guide.

## 1.3 Target Users
- **Professional Players**: Serious players who need powerful tools to study techniques and strategies.
- **AI Developers**: Developers seeking to use game data for AI training and optimization.
- **Endgame Enthusiasts**: Players aiming to deeply understand 2048 endgames and tables.
- **Casual Players**: Players wanting to experience modified games or improve scores with AI help.

---

# 2. Installation and Startup

## 2.1 System Requirements
- **Operating System**: Windows 7 or later.
- **Memory**:
  - Basic usage: Minimum 8GB
  - Computing large tables: 32GB+ recommended; configure virtual memory (swap) as needed
- **Disk Space**:
  - Basic usage: 10GB+ free space
  - Computing large tables: Depending on table type, may need 1TB+ free space (SSD recommended)
- **CPU**:
  - Any 64-bit CPU is compatible
  - CPUs supporting AVX-512 instruction set will perform significantly better
  - Supports multi-core parallel computing
- **GPU**: Not required; CPU-based computation only.

## 2.2 Installation Steps
- **Extract**: Download and extract the software package.
- **Run**: Locate `main.exe` in the extracted folder and double-click to launch the software.
- **First Launch**: The software performs JIT compilation and initialization, which may take 2-10 seconds; please be patient.

## 2.3 Startup Instructions
- **Initialization Time**: JIT compilation and resource loading are normal on first use of various features.
- **Error Troubleshooting**: If the software fails to start or encounters errors, check the `logger.txt` file in the root directory for detailed error information.

---

# 3. Core Concepts

## 3.1 Understanding Endgames

An endgame represents a game state where several large numbers have already been placed in certain tiles, and the player must use the remaining empty spaces to merge and reach a target number. **An endgame is the complete process from a specific board state to achieving the target number.**

### Example Understanding
- **10-space 16K Endgame**: The board already contains large numbers (8K, 4K, 2K, 1K, 512, 256 - 6 tiles total). The remaining 10 empty spaces are used to create a new 256, which merges with existing numbers to form 16K.
- **12-space 32K Endgame**: The board contains (16K, 8K, 4K, 2K - 4 tiles). The remaining 12 spaces must create 2048, which eventually merges into 32K.

## 3.2 Endgame Classification

Endgame classification is determined by the **layout of remaining empty spaces** and the **target number**, regardless of the specific values of existing large tiles. The algorithm treats endgames as equivalent if their empty space configurations are identical. For instance, whether the existing numbers are (16K, 8K, 4K, 2K) or (32K, 16K, 8K, 2K), the computational difficulty and strategy remain the same as long as the remaining board topology and the target tile are identical.

### Thought Question
**What is the target number for a 9-space 65K endgame?**

**Hint**: Think backwards—what large numbers should already exist if the final goal is 65K (65536)?

## 3.3 Table Concept

**A table is an endgame with added constraints.**

In actual gameplay, to reliably achieve goals and maintain formation stability, players typically need to:
- Keep large numbers in relatively fixed positions
- Create target numbers at specified locations

These constraints are called **table constraints**.

### Table Type Comparison

| Type | Constraint Strength | Human Difficulty | Success Rate | Characteristics & Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Table** | **Medium-High**: Large numbers locked in place (e.g., Snake pattern) | **Low**: Clear patterns, easy to summarize formation techniques | **Medium**: Limited rescue capability if extreme situations occur consecutively | **Mainstream for real gameplay**. Trades some flexibility for high certainty. |
| **Free Table** | **Minimal**: Large numbers can move freely as needed | **Extreme**: Extremely complex branching; human analysis depth typically insufficient | **Maximum**: Greatest error tolerance, handles most extreme board combinations | **Theoretical optimum**. Used for AI scoring, extreme research, recovering large numbers after chaos, learning transformation techniques. |
| **Variant Table** | **Minimal**: Essentially free tables on different board sizes | **Medium-Low**: Limited state space, reasoning depth constrained | **High** | **High-precision training**. Small state count allows humans near-perfect accuracy; ideal for targeted practice. |

### Free Table Examples
- `free9-128`: 9 empty spaces to create 128, 16K difficulty
- `free10-512`: 10 empty spaces to create 512, 32K difficulty

### Position Table Examples
- `L3-512`: 6 large numbers locked in L-shaped corner region, target 512, 32K difficulty
- `442-256`: Layered formation (Snake pattern) with 6 large numbers, target 256, 16K difficulty

## 3.4 Table Parameters Explained

| Parameter | Name | Description | Example |
|-----------|------|-------------|---------|
| **pattern** | Table name | Describes table constraints and remaining empty space quantity/position | `L3`, `442`, `free9` |
| **target** | Target number | The number to be created from remaining spaces | `256`, `512`, `2048` |

### Understanding Table Naming

- **Numeric Tables** (e.g., `442`, `4441`, `4432`): Empty spaces per row
  - `442` = Row 1: 4 spaces, Row 2: 4 spaces, Row 3: 2 spaces = 10 total
  - `4431` = Row 1: 4, Row 2: 4, Row 3: 3, Row 4: 1 = 12 total
  
- **Letter Tables** (e.g., `L3`, `t`, `LL`): Empty space distribution shape
  - `L3` = Outer L-shape + 3 additional spaces, equivalent to `4411`
  - `t` = 10 spaces in T-shape, equivalent to `2422`
  - `LL` = Outer L + Inner L, equivalent to `4422`
  
- **Free Tables** (e.g., `free9`): Completely unrestricted
  - `free9` = 7 completely free large numbers and 9 empty spaces

- **Suffix Markers**
  - `f (Free)`: Free large number marker. Indicates unconstrained movable large numbers beyond base table constraints. Example: `4432f` = 4432 base + 1 free large number (4 total). `4442ff` = 2 free large numbers.
  - `t (Transport)`: Intra-column repositioning marker. Allows certain large numbers to move up/down within their column. Used for optimization or emergency recovery. In `t` tables, 1x2 sections can move vertically.

### Success Rate Meaning

The **success rate** computed for a table means:
- Starting from that table's current state
- Without using any undo operations
- Following optimal strategy
- Probability of successfully creating the target number

---

# 4. Feature Overview

The software's main menu (**MainMenu**) contains multiple functional interfaces with different modules. Quick access buttons on the main interface let you reach these modules. Here's a brief introduction to each interface:

## 4.1 Settings Interface

The settings interface is divided into two parts: **Table Calculation** and **Global Settings**.

### 4.1.1 Table Calculation Section

**Core Parameters**:
- **Table Name**: Select target table from secondary menu (e.g., `L3`, `442`, `free9`)
- **Target Number**: Choose the number to create (128, 256, 512, 1024, 2048...)
- **Save Path**: Specify table data location; local SSD recommended for better I/O speed

**Calculation Options**:
- **Compress Temp Files**: Reduce disk usage during computation but increase computation time
- **Compress** (Recommended): Compress final table data, significantly reducing disk space
- **Keep Only Optimal Branches**: Remove non-optimal information, reducing data size; requires additional computation
- **Prune Low Success Rate** (Recommended): Remove low success rate states after computation without sacrificing accuracy

**Advanced Options**:
- **Advanced Algorithm**: Enable advanced algorithm for massive tables, dramatically improving speed and memory usage (only for large tables)
- **Small Tile Sum Limit (STSL)**: Controls pruning strength in advanced algorithm; balance between accuracy and speed
- **Chunked Recalculation**: Reduce memory threshold through chunked I/O, ideal for memory-constrained systems computing huge tables; recommend using with SSD
- **Success Rate Precision**: Customize storage precision and format for success rate data

Click **BUILD** to start computation. Real-time progress displays; supports resume from breakpoint.

**Breakpoint Resume Notes:**
While resumption is supported, crashes or full disk errors may corrupt recently written files. Direct resume could produce incorrect results. Safe operation: Before resuming, check the latest generated file sizes. Manually delete the last 2-3 files/folders to ensure data integrity.

### 4.1.2 Global Settings Section

**Colors & Themes**:
- **Block Color Scheme**: Select number block color scheme (2-32768 in different colors)
- **Color Theme**: 40+ preset themes (Classic, Royal, Supernova, etc.)
- **Custom Color Scheme**: Dark mode will take effect after restarting. Edit `color_schemes.txt` for advanced customization

**Game Parameters**:
- **4-Spawn Rate**: Probability of spawning 4 (range 0-1, default 0.1 = 10%)
- **Mistake Notebook Threshold**: Records board states from tests where single-step match rate falls below this value
- **Font Size**: Adjust UI font size for different resolutions
- **Animation**: Disable block movement/merge animations to reduce stuttering

## 4.2 Game Interface

Provides basic 2048 gameplay with AI assistance options.

### 4.2.1 Game Controls

**Basic Controls**:
- **Move**: Arrow keys or WASD
- **Undo**: Reverse previous action (only available with AI OFF)
- **New Game**: Start fresh game
- **AI**: Click "AI: ON" button to toggle AI assistance

### 4.2.2 Difficulty Settings

**Difficulty Slider**:
- Located at bottom of game interface; drag right to increase difficulty
- New tile generation controlled by algorithm, biased toward unfavorable positions rather than random
- AI calculates based on random generation logic; not recommended to use AI in hard mode.

## 4.3 MiniGame Interface

Offers diverse modified 2048 games with novel mechanics and higher difficulty for challenge-seeking players.

| Game Name | Board Size | Core Mechanic | Difficulty |
|-----------|-----------|---------------|-----------|
| **Blitzkrieg** | 4x4 | 3-minute countdown; time bonuses for higher numbers | Medium-High |
| **Design Master 1-4** | 4x4 | Arrange blocks in specified patterns; merge targets at exact positions | Medium |
| **Column Chaos** | 4x4 | Timed column swap; two columns randomly exchange every 40 moves | Medium |
| **Mystery Merge 1-2** | 4x4 | Hidden blocks shown as "?"; only revealed upon merge; requires deduction | High |
| **Gravity Twist 1-2** | 4x4 | Auto-gravity; automatic movement in random direction after each move | Medium |
| **Ferris Wheel** | 4x4 | Timed rotation; outer 12 blocks rotate clockwise every 40 moves | Medium-High |
| **Ice Age** | 4x4 | Freeze mechanic; blocks freeze into immobile ice if unmoved 80+ steps | High |
| **Isolated Island** | 4x4 | Special tiles; can only merge with themselves, not with other numbers | Medium-High |
| **Shape Shifter** | Variable | Variable board; random 12x12 irregular shapes; different each game | Medium-High |
| **Tricky Tiles** | 4x4 | Adversarial AI generation; new tiles appear in worse positions | High |
| **Endless Factorization** | 4x4 | Special blocks factor touched numbers; endless gameplay | Low |
| **Endless Explosions** | 4x4 | Bombs eliminate touched blocks; endless gameplay | Low |
| **Endless Giftbox** | 4x4 | Gift boxes spawn random new numbers; endless gameplay | Medium |
| **Endless Hybrid** | 4x4 | Mixed mechanics; bombs/pits/gifts with different effects; endless gameplay | High |
| **Endless AirRaid** | 4x4 | Air raid targets randomly appear; marked blocks eliminated when filled | Medium |

**HardMode Button**: Located at bottom-left. Enabling increases difficulty in game-specific ways.

## 4.4 Practice Interface

Displays table data and optimal moves; core tool for study and research.

### 4.4.1 Interface Layout

**Control Panel**:
- Menu bar to select table
- Select path containing table files
- Show/hide success rates for current position's four directions
- Click number buttons bottom-right to edit position

**Board Area**:
- Displays current position
- Shows position encoding

### 4.4.2 Function Buttons

**Position Operations**:
- **Set Board**: Load encoding from input box as current position
- **Default**: Show random initial position for this table
- **Flip Board**: Perform flip/rotation operations

**Demo Features**:
- **Auto Demo**: Continuously execute optimal moves, showing complete solution
- **Step**: Execute one optimal move, convenient for gradual learning
- **Undo**: Reverse previous operation

**Position Editing**:
  - **Color Number Buttons** (0-32K): Click to enter "board arrangement mode"
  - **Left-click**: In arrangement mode, set clicked position to selected number
  - **Right-click**: Increment clicked position by one level (2→4→8...)
  - **Other keys**: Decrement clicked position by one level
  - Click selected button again to exit arrangement mode

**Recording Features**:
- **Record**: Save demo moves and success rate sequences
- **Load Demo**: Load previously saved demonstrations
- **Play Demo**: Play loaded demo content

**Manual Mode**:
- Board stops auto-generating new tiles
- Left-click empty space: Place 2
- Right-click empty space: Place 4

### 4.4.3 Keyboard Shortcuts

- **Arrow Keys / WASD**: Move operations.
- **Enter**:
    Stop auto-demo if running. Load position encoding if focus on encoding box. Otherwise execute one step of optimal move.
- **Backspace / Delete**: Undo previous operation.
- **Q**: Toggle manual mode.
- **Z**: Screenshot current board and copy to clipboard.
- **E**: Equivalent to clicking number button 0.

## 4.5 Test Interface

Evaluates player endgame skill level, providing real-time feedback and performance analysis.

### 4.5.1 Basic Testing Process

**1. Select Table**:
- Choose desired table from top menu bar
- If table not loaded, access it from practice interface

**2. Select Initial Position**:
- System randomly generates initial position
- Can manually set position in practice interface, copy encoding and paste here

**3. Execute Moves**:
- Use arrow keys or WASD to move
- System auto-generates new tiles after each move

**4. View Feedback**:
- Real-time display of optimal moves and match rate
- Cumulative match rate and combo counter
- Positions automatically recorded to "Mistake Notebook"

### 4.5.2 Advanced Features

**Verse Replay Analysis**:
- Click "Analyze Verse Replay" button
- Import game replay from *2048verse.com* (`.txt` format)
- Automatically extracts relevant endgame segments and scores
- Generates detailed analysis report and replay files for each segment

**Replay Review**:
- Click "Replay Review" button
- Supports reviewing current test game or Verse analysis-generated `.rpl` files
- Fast-forward, rewind, precisely locate mistakes
- Step through historical games

**Mistake Notebook Function**:
- Located at bottom of test interface
- Automatically records low match-rate positions
- Filter by table, importance, etc.
- Jump to practice interface to view optimal moves for current position

### 4.5.3 Keyboard Shortcuts

- **Arrow Keys / WASD**: Move operations; real-time table comparison triggered.
- **R**: Quick reset and save `.rpl` replay to default directory.
- **F**: Show/hide right-side real-time analysis text.

---

# 5. Quick Start

Table computation is a prerequisite for AI, practice, and test modules. You need to first calculate and save a **table**.

## 5.1 Calculate a Table (Settings Interface)

Follow these steps to calculate your first table:

1. Enter main menu, access settings interface
2. Select `L3` or `442` for table name; select `256` for target number
3. Specify save path (e.g., `F:/L3_256`)
4. Check **compress**; keep other options default
5. Click **BUILD** button to start computation

## 5.2 Learn the Table (Practice Interface)

After computation completes, enter **Practice** to learn optimal moves:

1. Enter main menu, access practice interface
2. Select the just-calculated table from top-left menu bar
3. If completed, program auto-shows save path and random initial position
4. Display success rates for four directions
5. Click **Default** to switch random initial positions
6. Continuously demo optimal moves, observe how to progress from initial to target

## 5.3 Master the Table (Test Interface)

Verify your table mastery through testing:

1. Enter main menu, access test interface
2. Select desired table from top-left menu bar
3. Input custom position encoding
4. Execute your best moves
5. Observe real-time analysis panel on right side

---

# 6. Table Calculation Algorithm

## 6.1 Fundamental Design

### 6.1.1 Board Bitboard Representation

The software uses a single **64-bit unsigned integer (uint64)** to represent 4x4 board state. The 16 board squares require 4 bits each, totaling 64 bits.

### 6.1.2 Lookup Tables & Move Functions

To avoid loops and branches in merge logic, **lookup table (LUT)** acceleration is employed:

1. Pre-compute all possible 16-bit row states during initialization (65,536 combinations total)
2. Lookup tables don't store move results directly; instead store **XOR difference between post-move and original state**
3. Rotation transforms convert up/down moves into row operations
4. Extract rows/columns from position, lookup corresponding XOR difference, apply back to position

### 6.1.3 Symmetry & Canonical Forms

Many positions are logically equivalent through flipping or rotation. **Canonicalization** reduces redundant computation:

Implemented transformations:
* **Mirror flips**: Horizontal (`ReverseLR`), vertical (`ReverseUD`)
* **Rotations**: 90° CW/CCW (`RotateL/RotateR`), 180° (`Rotate180`)
* **Diagonal flips**: Main diagonal (`ReverseUL`), anti-diagonal (`ReverseUR`)

Before storage, positions and symmetries are compared; the **numerically smallest form** is stored. For specific tables, only diagonal or horizontal canonicalization can be selected.

## 6.2 Generation & Solution Logic

Table computation can be summarized as: **forward-layer BFS generation** + **backward DP solving**. By splitting the enormous position space into different "layers," computation completes within resource constraints.

### 6.2.1 Forward Position Generation (Forward BFS)

Starting from initial seed positions, breadth-first search explores all reachable positions (generation phase).

#### 1. Layered Search Logic
Positions divided into "layers" for memory efficiency, typically by sum of log values of all tiles.
- **State Transition**: Each layer's positions transition to next layer through "spawn new tile" and "move" actions
- **Operation Loop**: For each position in layer, try spawning 2/4 at each empty location, then move in four directions. If result fits pattern and changed, record it

#### 2. Sorting and Deduplication
- **Pre-dedup**: Use hash map during generation for immediate caching/filtering, reducing memory overhead. Handles collisions only lightly for duplicate-rate reduction
- **Sort Dedup**: After each layer completes, sort and remove duplicates. Sorting serves both dedup and enables efficient lookup of sorted positions

#### 3. Memory Management & Prediction
- **Length Prediction**: Predict next layer's scale growth, dynamically pre-allocate memory
- **Segmentation**: When single layer exceeds available memory threshold, auto-split into segments, process and merge individually

### 6.2.2 Backward Success Rate Recalculation

After all layers generated and persisted to disk, solving phase begins.

#### 1. Boundary Conditions
- Position containing target number at required location = 100% success
- Position with no valid moves = 0% success

#### 2. Success Rate Calculation
For each position $B$, success rate $P(B)$ depends on maximum success expectation across all possible moves:

$$P(B) = \max_{d \in \{U, D, L, R\}} \left( 0.9 \times \sum_{s_2 \in S_2(d)} \frac{P(s_2)}{N_{empty}} + 0.1 \times \sum_{s_4 \in S_4(d)} \frac{P(s_4)}{N_{empty}} \right)$$

Where:
- $d$ is move direction
- $S_2(d)$ and $S_4(d)$ are post-move positions from spawning 2 and 4 respectively
- $N_{empty}$ is total empty squares after move

#### 3. Indexing & Lookup Mechanism

- **Prefix Indexing**: Create index table using first 24 bits of uint64 (header) as key, record offset of first occurrence in sorted array
    
- **Search Strategy**: 1. Compute header of target position 2. Get binary search range from table 3. Binary search within narrow range

### 6.2.3 Breakpoint Resume & Error Recovery

Since huge table computation may take hours or days, robust error handling is built-in:
- **Layer Persistence**: Each completed layer written to disk immediately
- **Breakpoint Resume**: On startup, program auto-detects generated files. If interrupted, restart will resume from last complete layer, avoiding duplication

## 6.3 Advanced Algorithm

The advanced algorithm is this program's breakthrough technology for solving `free10`, `4442ff` and larger mega-tables. In these tables, single-layer position counts reach 10^10 to 10^11 magnitude (`free12`), making standard algorithms nearly impossible on personal computers.

### 6.3.1 Design Philosophy: Large Number Masking & Equivalence Classes

Advanced algorithm uses "masking" to differentially handle two number classes:
- **Large Numbers**: [64, 32768] range
- **Small Numbers**: ≤32 range

#### 1. Position Masking
During generation, large numbers not involved in merges are unified as special placeholders (Masks):
- **Equivalence Class**: One masked position can "represent" thousands of positions differing only in large number arrangements but sharing local logic
- **State Compression**: This classification compresses BFS state space 10-100x+, dramatically reducing sorting/dedup and binary search pressure

#### 2. Dynamic Expansion & Unmasking (Derive)
When new tiles might trigger large number merges (e.g., existing 128, newly created 128), "expose" operation unmasking:
- **Full-Position Expose**: Since masked positions contain multiple 32768 placeholders and mask represents position set, we can't predict specific value locations. Generate positions with value in all possible mask locations
- **Overhead Analysis**: Most unmasked positions are reachable game states, so unmasking adds minimal invalid computation

### 6.3.2 Heuristic Pruning Rules

Advanced algorithm introduces pruning logic to maximize efficiency:

#### 1. Large Number Combination Limits
Restrictions on concurrent large number types and quantities:
- Except smallest large number, others exist at most once
- If smallest > 64, may coexist at most 2 times
- If smallest = 64 with no 128, may coexist at most 3 times
- Valid: `(4K, 2K, 512, 128, 128)`, `(512, 64, 64, 64)`
- Pruned: `(4K, 2K, 512, 128, 128, 128)` or `(4K, 2K, 512, 128, 128, 64)`

#### 2. Small Number Sum Limit (STSL)
Hyperparameter **SmallTileSumLimit (STSL)** precisely controls small number complexity:
- Sum of all small numbers ≤ STSL + 64
- If position has two identical large numbers, require small sum ≤ STSL
- If three 64s exist, require small sum ≤ STSL - 64
- Per-move limit: Each move generates at most one new large number

Actually, only need small-number-sum pruning every 32 layers.

### 6.3.3 Batch Solving Mechanism

Core of advanced solving: treat "masked positions" as vectors for bulk evolution, use transformation encoding to solve ordering consistency.

#### 1. Full Unmasking & Cardinal:
   - Full unmasking: Expose all mask bits according to large number combinations
   - Ordering: Guarantees strict ascending position sequence
   - Unmasked cardinal: Resulting position count determined by large number combination
   - Data structure: Each masked position associates with Success Rate Array; array length equals its "unmasked cardinal," corresponding 1-to-1 with position sequence
   - Note: Full unmasking costly; core algorithm avoids it via masking

#### 2. Transformation Encoding & Cache Tables:
   - Problem: Masked positions' position sequence scrambles after "move + canonicalize" transform
   - Causes success rate array mismatch with current board, preventing expectation calculation
   - Labeling & Tracing: Introduce transformation encoding. Before transform, traverse all mask bits (`0xf`) in masked position, assign descending "labels" (`0xf, 0xe, 0xd...`) sequentially. After transform, observe label final positions, compress arrangement state into unique feature value `ind`
   - Pattern: Identical `ind` masked positions have identical sequence perturbation patterns
   - Mapping Repair: Use feature `ind` retrieve ranked_array from cache table. Array records index mapping from "perturbed sequence" back to "original ascending sequence"
   - Effect: Success Rate Array[ranked_array] achieves O(n) alignment, avoiding repeated full unmasking/sorting in solve loop

#### 3. Dynamic Classification:
   - Cardinal Change: Generating new large number changes unmasked cardinal. Must re-evaluate current large number combination, filter/multi-query success rate arrays
   - Cardinal = 1: Revert to standard algorithm logic

#### 4. Advantage Summary:
   - Batch solving dramatically reduces binary search count
   - Bulk success rate array processing with high parallelization potential
   - Supports block-by-block solving to further reduce peak memory on massive computation

### 6.3.4 Algorithm Advantages

Advanced algorithm overwhelmingly outperforms standard on mega-tables:

| Dimension | Standard | Advanced |
| :--- | :--- | :--- |
| **Computation Time** | Super-linear slowdown with table growth | Faster as tables grow (10x+ speedup possible) |
| **Memory Usage** | Minimum 2x single-layer memory | Peak memory reduced 1 order of magnitude |
| **Parallelization** | Parallelizable | Even higher parallelization potential |
| **Storage Efficiency** | Large files; needs compression | Refined intermediate files, smaller final tables |

Using this algorithm, `free12-4096` successfully computed on 9950X 128GB system in ~24 days.

### 6.3.5 Usage Limitations & Considerations

- **Applicable Scope**: Designed for `free10`, `free12`, `4442ff` mega-tables. For small tables like `L3`, `t`, advanced algorithm may hurt performance
- **Accuracy Loss**: Pruning causes minor success rate deviation; barely affects optimal move selection
- **Feature Conflict**: Unsupported: "keep only optimal branches" option, variant tables, tables with 't' suffix (or other limited-movement large numbers)
- **Compression Characteristic**: Standard algorithm high compression ratio, due to data redundancy. Advanced algorithm lower ratio; masked mechanism stores mostly success rate data with high entropy (near random), traditional compression ineffective on this


---

# 7. About AI

## 7.1 Prerequisites

To fully utilize the AI's performance, the following tables need to be computed:
   - **free12w-2048**
   - **free11w-2048**
   - **4442f-2048**
   - **free11w-512**

Ensure these tables can be properly loaded in the **Practice** before running the AI. As these tables require significant disk space, to minimize storage usage, it is recommended to:
   - Set a higher **success rate threshold (Remove boards with a success rate below)** for the first three tables. The program allows modifying this parameter during computation to apply different thresholds for different endgame stages.
   - Delete files numbered after half of the target value. For example, delete files from `free11w_2048_1024b` to `free11w_2048_1049b`.
   - Enable advanced algorithm.

## 7.2 AI Testing

Basic programming knowledge is required. Run AItest.py separately, where the `run_test` function writes single-test gameplay records to a specified path.
Record reading method:
```
import numpy as np

rec=np.fromfile(r"C:\Users\Administrator\Desktop\record\0", dtype='uint64,uint32,uint8')

def decode(encoded_board: np.uint64) -> np.ndarray:
    encoded_board = np.uint64(encoded_board)
    board = np.zeros((4, 4), dtype=np.int32)
    for i in range(3, -1, -1):
        for j in range(3, -1, -1):
            encoded_num = (encoded_board >> np.uint64(4 * ((3 - i) * 4 + (3 - j)))) & np.uint64(0xF)
            if encoded_num > 0:
                board[i, j] = 2 ** encoded_num
            else:
                board[i, j] = 0
    return board

for board,score,operator in rec[:]:
    print()
    print(hex(board))
    print(decode(board))
    print({0:'AI',1:'free12',2:'free11_2k',3:'4442f',4:'free11_512'}[operator],'  ',score)
```

## 7.3 AI Performance
-  This AI reaches the 65536 tile 8.4% (±1.6%) of the time and the 32768 tile 86.1% (±2.0%) of the time without undos.

   
| search depth    | games | avg score  | median score | moves/s |
|-----------------|-------|------------|--------------|---------|
| 1~9 adaptive    | 1200  | 772353     | 819808       | 11      |

-  The following table gives the AI performance statistics and success rates by stage

| milestone     | success rate | comparable score |
|---------------|--------------|------------------|
| 8k            | 100%         | 97000            |
| 16k           | 99.9%        | 210000           |
| 32k           | 86.1%        | 453000           |
| 32k+16k       | 72.8%        | 663000           |
| 32k+16k+8k    | 61.8%        | 760000           |
| 32k+16k+8k+4k | 53.3%        | 804000           |
| final 2k      | 46.3%        | 824000           |
| 65k           | 8.4%         | 971000           |


![survival rate](https://github.com/user-attachments/assets/8d708f06-4994-4878-8e27-2aa831db1a2b)


-  The following table shows how often each table is used.

 | Search | free12w-2k     | free11w-2k | 4442f-2k | free11w-512  |
 |--------|----------------|------------|----------|--------------|
 | 26.49% | 13.38%         | 1.41%      | 51.50%   | 7.22%        |



## 7.4 General AI

After version 8.0, the AI can utilize any calculated tables.
However, it currently does not adjust its strategy based on available table combinations, 
so it can hardly leverage t tables & snake tables.
Recommended tables include free10w-512 (or L3-512), 4431-512, etc.


---

# 8. Frequently Asked Questions

### Q1: Why does the software take a long time to start?
**A1**: It requires JIT (Just-In-Time) compilation. JIT compilation improves program performance, but it results in longer initialization times when using various features for the first time.

### Q2: How to optimize the disk space occupied by the table file?
**A2**: Consider enabling the following options to reduce disk usage:
1. **Compress Temp Files**: Compress all intermediate files to reduce disk usage during calculation.
2. **Keep Only Optimal Branches**: Delete all non-optimal moves and keep only the best branches to further reduce disk space.

### Q3: How to resolve high memory usage during computation?
**A3**: The high memory usage during computation is due to storing a large amount of endgame data. It is recommended to increase your system memory or configure virtual memory.

### Q4: Does the computation support resuming from a breakpoint?
**A4**: Yes, table computation supports resuming from a breakpoint. For less experienced users, it is recommended to keep all options and parameters the same when reconnecting.
For experienced users, some parameters can be adjusted based on the calculation stage to achieve more flexible goals, including compressing table files that were not compressed earlier.

### Q5: How to handle errors during computation?
**A5**: If you encounter an error during computation, follow these steps:
1. **Try reconnecting**: Close all windows and restart the program to resume computation.
2. **Check file size trends**: If the issue persists, check the file size trends to identify any anomalies.
3. **Delete recent files**: Delete a few recent files and try reconnecting.
4. **Check error logs**: If the issue still exists, check the `logger.txt` file for error logs.
5. **Contact the Author**: If the above methods do not resolve the issue, it is recommended to contact the author, providing the formation you calculated and the specific situation you encountered. The author will assist you in further troubleshooting and resolving the issue.

### Q6: How to handle disk I/O hang-ups during computation?
**A6**: If there is a hang-up during computation, particularly when using external hard drives, try the following:
1. Manually copy a small file to the hard drive to activate the write operation.
2. If the issue persists, check the external hard drive connection or try a different connection method.

### Q7: Why is the AI's performance lower than expected?
**A7**: If the AI's performance is not as expected, it may be due to the following reasons:
1. **No table computed**: The AI relies on pre-computed tables. Make sure the related tables are available.
2. **Table path changed**: If the table file was moved, reset its path in the Practice interface.
3. **Insufficient computer performance**: If the computer's performance is low, the AI's search speed may be slow, so please be patient.

### Q8: Why do the numbers on the board appear too large, causing distortion?
**A8**: The font size is affected by the display resolution, which in turn affects the board layout. You can adjust the font size in the settings.

---

# 9. Support and Feedback

## 9.1 Getting Support

If you encounter any issues while using the software, you can get technical support through the following channels:

**GitHub Issue Submission**  

   - Please visit the [GitHub project page](https://github.com/game-difficulty/2048EndgameTablebase) and submit an issue. Provide detailed information to help us reproduce the problem (e.g., formation, error logs, steps).

**Join the Community**  

   - Join the 2048 online community to interact with other users, share experiences, and discuss solutions.
   - **QQ Group**: 94064339  
   - **Discord Channel**: 2048 Runs

## 9.2 Submit Feedback

**Star the Project on GitHub**  

   - If you like the project or find it helpful, feel free to star the project on our [GitHub project page](https://github.com/game-difficulty/2048EndgameTablebase).


---

# 10. Appendix

## 10.1 Common Formation Info

| Formation | Number of Empty Spaces | Description                                                                            | Final Formation | Large Number Movement Restrictions                                            | Special Notes                                                                                                                                                                                       |
|-----------|------------------------|----------------------------------------------------------------------------------------|-----------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 442       | 10-space               | The empty spaces in each row are 4420                                                  | Snake           | Cannot move                                                                   |                                                                                                                                                                                                     |
| 442t      | 10-space               | The empty spaces in each row are 4420                                                  | Snake           | 3 columns in both sides shift                                                 |                                                                                                                                                                                                     |
| L3        | 10-space               | The empty spaces form an outer L shape + 3 spaces                                      | Gap PDF         | Cannot move                                                                   | Supports position parameters [0,1,2]                                                                                                                                                                |
| L3t       | 10-space               | "t" indicates more allowed shifts                                                      | Gap PDF         | Single column shift                                                           | Does not support position parameters                                                                                                                                                                |
| t         | 10-space               | Empty spaces form a T shape                                                            | T               | 1x2 part shift, <br/>2x2 part cannot move                                     |                                                                                                                                                                                                     |
| 444       | 12-space               | The empty spaces in each row are 4440                                                  | Snake           | Cannot move                                                                   | Supports position parameters [0,1]                                                                                                                                                                  |
| LL        | 12-space               | Empty spaces form two L shapes                                                         | T               | Cannot move                                                                   | Supports position parameters [0,1]                                                                                                                                                                  |
| 4431      | 12-space               | The empty spaces in each row are 4431                                                  | Gap PDF         | Cannot move                                                                   |                                                                                                                                                                                                     |
| 4432f     | 12-space               | "f" indicates a freely movable large number                                            | T               | Corner large numbers cannot move, <br/>the free large number are unrestricted | Same for 4442f                                                                                                                                                                                      |
| 4432      | 13-space               | The empty spaces in each row are 4432                                                  | T               | Cannot move                                                                   | Other unnamed numeric-name formation can follow this naming pattern                                                                                                                                 |
| 4441      | 13-space               | The empty spaces in each row are 4441                                                  | Gap PDF         | Cannot move                                                                   |                                                                                                                                                                                                     |
| free9w    | 9-space                | Full 9-space free endgame                                                              |                 | Unrestricted                                                                  | Other full free tables follow this pattern                                                                                                                                                          |
| free9     | 9-space                | Half 10-space free endgame.<br/>free9-n is equivalent to the second half of free10w-2n |                 | Unrestricted                                                                  | Example: free9-256:<br/>Initial setup includes 6 large numbers and one 256,<br/>the remaining 9 spaces form another 256 and merge with the existing 256. Other half free tables follow this pattern |
| 3x3       | 9-space (Variant)      | 3x3 variant, initial setup only has a 2                                                |                 | Unrestricted                                                                  | Same for 2x4 and 3x4 variants                                                                                                                                                                       |


| Formation | Example Initial Setup                                                  |
|-----------|------------------------------------------------------------------------|
| 442/442t  | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ x x ```<br/>```x x x x``` |
| L3/L3t    | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ x x x ```<br/>```_ x x x``` |
| t         | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```x _ x x ```<br/>```x _ x x``` |
| 444       | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```x x x x``` |
| LL        | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ x x ```<br/>```_ _ x x``` |
| 4431      | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ x ```<br/>```_ x x x``` |
| 4432f     | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ x ```<br/>```x _ x x``` |
| 4432      | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ x ```<br/>```_ _ x x``` |
| 4441      | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ x x x``` |
| free9w    | Arbitrary arrangement of large numbers                                 |
| free9     | Arbitrary arrangement of large numbers                                 |
| 3x3       | ```_ _ _ ```<br/>```_ _ 2 ```<br/>```_ _ _```                          |

## 10.2 Table Size Reference

| Formation    | ~Size  | ~Compressed |
|--------------|--------|-------------|
| LL_4096_0    | 1.1 TB | 450 GB      |
| 4431_2048    | 650 GB | 250 GB      |
| 444_2048     | 130 GB | 60 GB       |
| t_512        | 2.5 GB | 1 GB        |
| L3_512_0     | 2 GB   | 1 GB        |
| 442_512      | 2 GB   | 1 GB        |
| 4432_512     | 250 GB | 100 GB      |
| 4441_512     | 800 GB | 330 GB      |
| 4432f_2048   | 3.5 TB | 1.33 TB     |
| free9_256    | 460 GB | 170 GB      |
| free10w_1024 | 2 TB   | 700 GB      |
