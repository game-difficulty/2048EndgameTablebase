
# User Manual

**Author**: game_difficulty  
**Version**: 9.0  
**Date**: 2026.1.20 

---

# 1. Overview

This is a comprehensive 2048 training software suite offering table calculation, endgame training, game analysis, AI assistance, variant mini-games, and more.

## 1.1 Key Highlights

- **Lightning-Fast Computation**: Highly optimized algorithms and advanced pruning deliver industry-leading calculation speeds, accelerating large table computations by over 10x.
- **Storage Efficiency**: Multiple pruning and data compression techniques significantly reduce the disk footprint of table files.
- **State-of-the-Art AI**: Features the world’s most powerful 2048 AI, achieving an 8.4% (±1.6%) success rate for 65536 and 86.1% (±2.0%) for 32768, far surpassing other AI engines.
- **Complete Toolkit**: Comprehensive training and analysis tools—including real-time feedback, an error log (mistake book), and replay analysis—help users master advanced strategies.

## 1.2 Main Function Modules

- **Game**: Standard 2048 gameplay with built-in professional AI assistance.
- **Practice**: Explore tables and learn optimal moves, with support for recording and playback.
- **Test**: Assess your mastery with automated mistake tracking, replay analysis, and 2048verse.com replay import.
- **MiniGame**: Diverse variant games offering unique challenges.
- **Settings**: Global configuration, table calculation, and advanced algorithm setup.
- **Help**: Documentation and guides.

## 1.3 Target Users

- **Professional Players**: Those seeking powerful tools to research advanced techniques and strategies.
- **AI Developers**: Developers looking to utilize game data for AI training and optimization.
- **Endgame Enthusiasts**: Players dedicated to a deep understanding of 2048 endgames and tablebases.
- **Casual Players**: Those looking to experience new variants or improve their scores with AI help.

---

# 2. Installation and Startup

## 2.1 System Requirements
- **Operating System**: Windows 7 or later.
- **Memory**:
    - Basic usage: Minimum 8GB.
    - Large tablebase calculation: 32GB+ recommended; configure significant virtual memory (paging file) if necessary.
- **Disk Space**:
    - Basic usage: 10GB+ free space.
    - Large tablebase calculation: Up to 1TB+ depending on the pattern (SSD strongly recommended).
- **CPU**:
    - Any 64-bit CPU is compatible.
    - CPUs supporting the AVX-512 instruction set offer superior performance.
    - Full support for multi-core parallel processing.
- **GPU**: No GPU required; the engine is pure CPU-based.

## 2.2 Installation Steps

- **Extract**: Download and extract the software package.
- **Run**: Locate and double-click `main.exe` in the extracted folder to launch.
- **Initial Launch**: The software will perform JIT compilation and initialization, taking approximately 2-10 seconds.

## 2.3 Startup Instructions

- **Initialization**: JIT compilation and resource loading occur during the first use of each feature; this is normal.
- **Troubleshooting**: If you encounter errors, please check the `logger.txt` file for detailed diagnostic information.

# 3. Core Concepts

## 3.1 Endgames

An endgame represents a game state where several large numbers have already been placed on specific tiles, and the player must use the remaining empty spaces to merge and reach a target tile. **An endgame is the complete process from a specific board state to achieving the target tile.**

### Example

- **10-space 16K Endgame**: The board already contains large numbers (8K, 4K, 2K, 1K, 512, 256 - 6 tiles total). The remaining 10 empty spaces are used to create a new 256, which merges with existing numbers to form 16K.
- **12-space 32K Endgame**: The board contains (16K, 8K, 4K, 2K - 4 tiles). The remaining 12 spaces must create 2048, which eventually merges into 32K.

## 3.2 Endgame Classification

Endgame classification is determined by the **layout of remaining empty spaces** and the **target tile**, regardless of the specific values of existing large tiles. The algorithm treats endgames as equivalent if their empty space configurations are identical. For instance, whether the existing numbers are (16K, 8K, 4K, 2K) or (32K, 16K, 8K, 2K), the computational difficulty and strategy remain the same as long as the remaining board topology and the target tile are identical.

### Thought Question
**What is the target tile for a 9-space 65K endgame?**

**Hint**: Think backwards—what large numbers should already exist if the final goal is 65K (65536)?

## 3.3 Table Concept

**A table is an endgame with added constraints.**

In actual gameplay, to reliably achieve goals and maintain formation stability, players typically need to:

- Keep large numbers in relatively fixed positions
- Create target tiles at specified locations

These constraints are called **table constraints**.

### Table Type Comparison

| Type | Constraint Strength | Cognitive Complexity | Success Rate | Characteristics & Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Table** | **Medium-High**: Large numbers locked in place (e.g., Snake pattern). | **Low**: Clear patterns, easy to summarize formation techniques. | **Medium**: Limited rescue capability if extreme situations occur consecutively. | **Mainstream for real gameplay**. Trades some flexibility for high certainty. |
| **Free Table** | **Minimal**: Large numbers can move freely as needed. | **Extreme**: Extremely complex branching; human analysis depth typically insufficient. | **Maximum**: Greatest error tolerance, handles most extreme board combinations. | **Theoretical optimum**. Used for AI benchmarking, extreme research, recovering large numbers after chaos, and learning transformation techniques. |
| **Variant Table** | **Minimal**: Essentially free tables on different board layouts. | **Medium-Low**: Limited state space, reasoning depth constrained. | **High** | **High-precision training**. Small state count allows humans near-perfect accuracy; ideal for targeted practice. |

### Free Table Examples

- `free9-128`: 9 empty spaces to create 128 (16K endgame).
- `free10-512`: 10 empty spaces to create 512 (32K endgame).

### Fixed Table Examples

- `L3-512`: 6 large numbers locked in L-shaped corner region, target 512 (32K endgame).
- `442-256`: Layered formation (Snake pattern) with 6 large numbers, target 256 (16K endgame).

## 3.4 Table Parameters

| Parameter | Name | Description | Example |
|-----------|------|-------------|---------|
| **pattern** | Table name | Describes table constraints and remaining empty space quantity/position. | `L3`, `442`, `free9` |
| **target** | Target number | The number to be created from remaining spaces. | `256`, `512`, `2048` |

### Table Naming

- **Numeric Tables** (e.g., `442`, `4441`, `4432`): Empty spaces per row.
    - `442` = Row 1: 4 spaces, Row 2: 4 spaces, Row 3: 2 spaces = 10 total.
    - `4431` = Row 1: 4, Row 2: 4, Row 3: 3, Row 4: 1 = 12 total.
  
- **Letter Tables** (e.g., `L3`, `t`, `LL`): Empty space distribution shape.
    - `L3` = Outer L-shape + 3 additional spaces, equivalent to `4411`.
    - `t` = 10 spaces in T-shape, equivalent to `2422`.
    - `LL` = Outer L + Inner L, equivalent to `4422`.
  
- **Free Tables** (e.g., `free9`): Completely unrestricted.
    - `free9` = 7 completely free large numbers and 9 empty spaces.

- **Suffix Markers**
    - `f (Free)`: Free large number marker. Indicates unconstrained movable large numbers beyond base table constraints. Example: `4432f` = 4432 base + 1 free large number (4 total). `4442ff` = 2 free large numbers.
    - `t (Transport)`: Intra-column repositioning marker. Allows certain large numbers to move up/down within their column. Used for optimization or emergency recovery. In `t` tables, 1x2 sections can move vertically.

### Success Rate

The **success rate** computed for a table means:

- Starting from that table's current state;
- Without using any undo operations;
- Following the optimal strategy;
- The probability of successfully creating the target tile.

---

# 4. Feature Overview

The software's main menu (**MainMenu**) serves as the central hub, containing several functional interfaces. Quick access buttons on the main screen allow you to navigate between these modules. Here is a brief introduction to each interface:

## 4.1 Settings Interface

The Settings interface is split into two primary sections: **Table Calculation** and **Global Settings**.

### 4.1.1 Table Calculation Section

**Core Parameters**:

- **Table Name**: Select the target table pattern from the dropdown menu (e.g., `L3`, `442`, `free9`).
- **Target Number**: Choose the tiles to be achieved (128, 256, 512, 1024, 2048...).
- **Save Path**: Specify the storage location for table data; using a local SSD is strongly recommended for better I/O performance.

**Calculation Options**:

- **Compress Temp Files**: Reduces disk footprint during the generation phase but extends total computation time.
- **Compress** (Recommended): Compresses the final table data, significantly saving long-term disk space.
- **Keep Only Optimal Branches**: Strips away sub-optimal move data to minimize file size; requires additional processing time.
- **Prune Low Success Rate** (Recommended): Post-calculation pruning that removes states with negligible winning chances without compromising overall accuracy.

**Advanced Options**:

- **Advanced Algorithm**: Utilizes specialized logic for massive tables, drastically reducing computation time and memory overhead (recommended for large tables only).
- **Small Tile Sum Limit (STSL)**: Controls the pruning intensity of the advanced algorithm—a trade-off between success rate precision and calculation speed.
- **Chunked Backward Solving**: Lowers the memory ceiling via chunked I/O. Ideal for memory-constrained systems computing massive endgames; best paired with an SSD.
- **Success Rate Precision**: Allows customization of the storage format and bit-depth for success rate data.


Click **BUILD** to initiate the process. The interface displays real-time progress and supports resuming from interruptions.

**Notes on Resuming Calculation:**
While the program supports resuming from breakpoints, interruptions caused by a full disk or unexpected crashes may corrupt the most recently written files. Resuming directly could lead to data inconsistencies. 
**Recommended Recovery Procedure**: Before resuming, inspect the size of the latest generated files. Manually delete the last 2-3 potentially corrupted files or folders to ensure a clean data chain.

### 4.1.2 Global Settings Section

**Appearance & Themes**:

- **Tile Color Scheme**: Choose the background color mapping for tiles (2 through 32,768).
- **Theme Presets**: 40+ pre-configured styles (Classic, Royal, Supernova, etc.).
- **Color Mode**: Dark mode applies upon restart. Advanced users can edit `color_schemes.txt` for deep UI customization.

**Game Parameters**:

- **4-Spawn Rate**: Probability of spawning a 4 (Range 0-1, Default 0.1 = 10%).
- **Mistake Tracker Threshold**: Automatically logs board states from tests where the move optimality (match rate) falls below this specified value.
- **Font Size**: Scale the UI text to match different screen resolutions.
- **Animations**: Toggle movement and merge animations to improve UI responsiveness on lower-end hardware.

## 4.2 Game Interface

Provides a standard 2048 gameplay with professional AI assistance.

### 4.2.1 Gameplay & Controls

- **Move**: Use Arrow keys or WASD.
- **Undo**: Revert the last move (Available only when AI is OFF).
- **New Game**: Discard the current game and start fresh.
- **AI**: Click the **"AI: ON/OFF"** button to enable or disable real-time assistance.

### 4.2.2 Difficulty Settings

**Difficulty Slider**:

- Located at the bottom; drag right to increase the challenge.
- Higher settings use an adversarial algorithm that spawns tiles in unfavorable positions rather than randomly.
- Note: The built-in AI is optimized for random spawn logic; it is not recommended for use in adversarial modes.

## 4.3 MiniGame Interface

Features a collection of 2048 variants with unique mechanics, designed for players seeking unconventional challenges.



| Game Name | Board Size | Core Mechanic                                                               | Difficulty |
|-----------|-----------|-----------------------------------------------------------------------------|-----------|
| **Blitzkrieg** | 4x4 | 3-minute blitz; reach milestones to earn time bonuses                       | Med-High |
| **Design Master 1-4** | 4x4 | Pattern matching; reach specific milestones at exact coordinates            | Medium |
| **Column Chaos** | 4x4 | Dynamic columns; two columns randomly swap every 40 moves                   | Medium |
| **Mystery Merge 1-2** | 4x4 | Blind 2048; tiles are "?" until merged; requires memory and deduction       | High |
| **Gravity Twist 1-2** | 4x4 | Auto-tilt; board automatically shifts in a random direction after each move | Medium |
| **Ferris Wheel** | 4x4 | Perpetual motion; the outer 12 tiles rotate clockwise every 40 moves        | Med-High |
| **Ice Age** | 4x4 | Permafrost; tiles unmoved for 80+ steps freeze into immobile obstacles      | High |
| **Isolated Island** | 4x4 | Exotic tiles; special blocks that can only merge with their own kind        | Med-High |
| **Shape Shifter** | Variable | Morphing grid; irregular 12x12 layouts that change every session            | Med-High |
| **Tricky Tiles** | 4x4 | Adversarial Spawning; tiles are placed by an AI trying to end your game     | High |
| **Endless Factorization** | 4x4 | Math merge; special blocks factorize tiles they collide with                | Low |
| **Endless Explosions** | 4x4 | Demolition; bombs eliminate the first tile they touch                       | Low |
| **Endless Giftbox** | 4x4 | RNG Boxes; boxes transform into random tiles                                | Medium |
| **Endless Hybrid Mode** | 4x4 | Chaos; combines bombs, pits, and gift boxes in one board                    | High |
| **Endless AirRaid** | 4x4 | Danger zones; marked grids are destroyed after moves        | Medium |

**HardMode Button**: Found at the bottom-left. Enabling this adds game-specific modifiers to further increase the challenge.

## 4.4 Practice Interface

A core tool for study and research, the Practice interface displays table data and optimal moves.

### 4.4.1 Interface Layout

**Control Panel**:

- Select the table to view from the menu bar.
- Choose the directory path where table files are located.
- Show/Hide success rates for the four move directions.
- Click the numeric buttons at the bottom right to enter board editing mode.

**Board Area**:

- Displays the current board state.
- Displays the position encoding.

### 4.4.2 Function Buttons

**Board Operations**:

- **Set Board**: Loads the encoding from the input box as the current board state.
- **Default**: Displays a random initial position for the selected table.
- **Flip/Rotate**: Performs board symmetry operations (flips and rotations).

**Demo Functions**:

- **Demo**: Automatically executes optimal moves continuously to demonstrate the full solution process.
- **Step**: Executes a single optimal move, ideal for step-by-step learning.
- **Undo**: Reverts the last move.

**Board Editing**:

- **Colored Tile Buttons (0-32k)**: Click to enter "Board Editing Mode."
- **Left-click**: Sets the target tile on the board to the selected value.
- **Right-click**: Increases the target tile value by one level (e.g., 2 → 4 → 8...).
- **Other keys**: Decreases the target tile value by one level.
- Click the currently selected tile button again to exit Board Editing Mode.

**Recording Features**:

- **Record**: Captures the sequence of moves and success rate data from an auto-demo.
- **Load Recording**: Loads a previously saved recording file.
- **Play Recording**: Plays the loaded recording content.

**Manual Mode**:

- Stops the system from automatically spawning new tiles.
- Left-click empty space: Place a '2' tile.
- Right-click empty space: Place a '4' tile.

### 4.4.3 Keyboard Shortcuts

- **Arrow Keys / WASD**: Move operations.
- **Enter**: Stops the demo if auto-playing; loads the encoding if the focus is on the input box; otherwise, executes a single optimal move.
- **Backspace / Delete**: Undo the previous move.
- **Q**: Toggle Manual Mode.
- **Z**: Take a screenshot of the current board and copy it to the clipboard.
- **E**: Acts as clicking the '0' tile button.

## 4.5 Test Interface

The Test interface evaluates a player's endgame skill level, providing real-time feedback and performance analysis.

### 4.5.1 Basic Testing Process

**1. Select Table**:

- Choose the desired table from the top menu bar.
- If table data is not loaded, specify the path in the Practice interface first.

**2. Select Initial Position**:

- The system randomly generates an initial position.
- You can manually set a position in the Practice interface, then copy the encoding and paste it here.

**3. Execute Moves**:

- Use the arrow keys or WASD to perform moves.
- The system automatically spawns a new tile after each move.

**4. View Feedback**:

- Real-time display of the optimal move and your move's match rate.
- Displays cumulative match rate and current combo counter.
- Games are automatically recorded in the "Mistake Notebook."

### 4.5.2 Advanced Features

**Verse Replay Analysis**:

- Click the "Analyze Verse Replay" button.
- Import replay files (`.txt` format) from *2048verse.com*.
- Automatically extracts relevant endgame segments from the full replay and scores them.
- Generates detailed analysis reports and replay files for each segment.

**Replay Review**:

- Click the "Replay Review" button.
- Supports reviewing the current test session or `.rpl` files generated by Verse analysis.
- Supports fast-forward, rewind, and precise "Mistake" (blunder) positioning.
- Step through historical games for deep review.

**Mistake Notebook**:

- Located at the bottom of the Test interface.
- Automatically records positions with low match rates.
- Filter by table type, importance, and other conditions.
- Jump to the Practice interface to view the optimal solution for a recorded position.

### 4.5.3 Keyboard Shortcuts

- **Arrow Keys / WASD**: Move operations; triggers real-time table comparison.
- **R**: Quick reset and save the `.rpl` replay to the default directory.
- **F**: Show/Hide the real-time analysis text on the right.

---

# 5. Quick Start

Table computation is a prerequisite for using key features like the AI, Practice, and Test modules. You must first calculate and save a **table**.

## 5.1 Calculate a Table (Settings Interface)

Follow these steps to calculate your first table:

1. Open the Main Menu and enter the **Settings** interface.
2. For **Formation**, select `L3` or `442`; for **Target Tile**, select `256`.
3. Specify a **Save Path** (e.g., `F:/L3_256`).
4. Check **Compress** and keep other options at their defaults.
5. Click the **BUILD** button to start the calculation.

## 5.2 Learn the Table (Practice Interface)

Once the calculation is complete, enter the **Practice** interface to study the optimal strategy:

1. Open the Main Menu and enter the **Practice** interface.
2. Select the newly calculated table from the top-left menu bar.
3. If the calculation is finished, the program will automatically load the save path and a random initial position.
4. The success rates for all four directions will be displayed.
5. Click the **Initial Position** button to cycle through different random starting boards.
6. Use the auto-demo to observe how the strategy progresses from the initial state to the target tiles.

## 5.3 Master the Table (Test Interface)

Verify your mastery of the table through active testing:

1. Open the Main Menu and enter the **Test** interface.
2. Select the desired table from the top-left menu bar.
3. Input a custom position encoding (or use a system-generated random one).
4. Execute the moves you believe are best.
5. Monitor the real-time analysis panel on the right side to check your accuracy.

# 6. Table Calculation Algorithm

## 6.1 Fundamental Design

### 6.1.1 Bitboard Representation

The software employs a **64-bit unsigned integer (uint64)** to represent the 4x4 board state. The board consists of 16 tiles, with each tile occupying 4 bits, totaling 64 bits.

### 6.1.2 Lookup Tables & Move Functions (LUT)

To bypass the loops and conditional branches in the merge logic, a **Lookup Table (LUT)** based acceleration scheme is used:

1. Pre-calculate all 65,536 possible 16-bit row configurations during the initialization phase.
2. The lookup table does not store move results directly; instead, it stores the **XOR difference between the post-move state and the original state**.
3. Rotation transforms are used to convert up/down moves into row-wise operations.
4. The algorithm extracts rows/columns from the board, retrieves the corresponding XOR difference from the table, and applies it back to the original board state.

### 6.1.3 Symmetry & Canonical Forms

In table calculation, many positions are logically equivalent through flipping or rotation. A **Canonicalization** mechanism is introduced to reduce redundant computations.

Implemented symmetry transformations:

* **Mirror Flips**: Horizontal (`ReverseLR`) and Vertical (`ReverseUD`).
* **Rotations**: 90° Clockwise/Counter-clockwise (`RotateL/RotateR`) and 180° (`Rotate180`).
* **Diagonal Flips**: Main diagonal (`ReverseUL`) and Anti-diagonal (`ReverseUR`).

Before a position is stored, the program compares it with all its symmetrical forms and stores only the **numerically smallest form**. For specific tables, the user can opt for only diagonal or only horizontal canonicalization.

## 6.2 Generation & Solution Logic

The core table algorithm is divided into two phases: **Forward Layer-by-Layer BFS Generation** and **Backward Dynamic Programming Solving (DP)**. By partitioning the vast position space into different "Layers," the program can compute massive tables within finite resource constraints.

### 6.2.1 Forward Position Generation (Forward BFS)

Starting from one or more initial seed boards, a Breadth-First Search (BFS) algorithm explores all reachable position spaces. This is known as the **Generation Phase**.

#### 1. Layered Search Logic

Positions are divided into "Layers" for memory efficiency, typically categorized by the sum of the tiles' exponents (logarithms).

- **State Transition**: Positions in each layer transition to the next through two actions: "spawn new tile" and "execute move."
- **Operation Loop**: For every position in a layer, the program iterates through all empty spaces, attempts to spawn a 2 or 4 tile, and then executes moves in four directions. If the resulting position fits the table pattern and the board has changed, it is recorded.

#### 2. Sorting and Deduplication

- **Pre-deduplication**: During generation, a Hash Map is used for immediate caching and filtering to reduce memory overhead. This process does not resolve hash collisions but serves to lower the duplication rate.
- **Sort Dedup**: Once a layer is complete, the data is sorted and duplicates are removed. Sorting serves both deduplication needs and allows for efficient sorted position lookups.

#### 3. Memory Management & Prediction

- **Length Prediction**: The program predicts the growth scale of the next layer to pre-allocate memory dynamically.
- **Segmentation**: When a single layer exceeds the available memory threshold, it is automatically split into segments, processed, and merged individually.

### 6.2.2 Backward Success Rate Recalculation

Once all layers are generated and persisted to disk, the program enters the **Solving Phase**. During this process, you can adjust the "Success Rate Deletion Threshold" to precisely control the file size of each layer.

#### 1. Boundary Conditions

- If the target tile appears at the required location, the position is marked as 100% success.
- If a position has no valid moves, its success rate is marked as 0%.

#### 2. Success Rate Calculation
For any position $B$, the success rate $P(B)$ is determined by the maximum expected success rate across all possible move directions:

$$P(B) = \max_{d \in \{U, D, L, R\}} \left( 0.9 \times \sum_{s_2 \in S_2(d)} \frac{P(s_2)}{N_{empty}} + 0.1 \times \sum_{s_4 \in S_4(d)} \frac{P(s_4)}{N_{empty}} \right)$$

Where:

- $d$ is the move direction.
- $S_2(d)$ and $S_4(d)$ are the sets of subsequent positions generated by spawning a 2 or 4 tile after the move.
- $N_{empty}$ is the total number of empty tiles after the move.

#### 3. Indexing & Lookup Mechanism

- **Prefix Indexing**: An index table is established using the first 24 bits of the uint64 board (the header) as the key. It records the offset of the first occurrence of each 24-bit header in the sorted array.
- **Search Strategy**: 1. Compute the header of the target position. 2. Retrieve the binary search range from the index table. 3. Perform a binary search within that narrow range.

### 6.2.3 Breakpoint Resume

Since computing massive tables can take hours or even days, the program includes a robust error-tolerance mechanism:

- **Layer Persistence**: Data is written to disk immediately after each layer completes.
- **Breakpoint Resume**: On startup, the program automatically detects generated files. If interrupted, it resumes from the last complete layer, avoiding redundant calculations.

## 6.3 Advanced Algorithm

The Advanced Algorithm is a critical technical breakthrough designed for calculating large-scale tables like `free10`, `4442ff`, and beyond. In these scenarios, the number of positions per layer can reach $10^{10}$ to $10^{11}$ (e.g., `free12`), making standard algorithms computationally prohibitive on personal computers.

### 6.3.1 Design Philosophy: Large Number Masking & Equivalence Classes

The algorithm utilizes "Masking" technology to treat two classes of numbers differently:

- **Large Numbers**: Tiles in the range $[64, 32768]$.
- **Small Numbers**: Tiles with values $\le 32$.

#### 1. Position Masking
During the generation phase, large numbers not currently involved in merges are treated as special placeholders (Masks).

- **Equivalence Classes**: A single masked position can represent thousands of actual positions that differ only in the arrangement of large numbers but share the same local logic.
- **State Compression**: By generating only masked positions, the BFS (Breadth-First Search) state space is compressed by $10$–$100\times$, significantly reducing the pressure on sorting, deduplication, and binary searches.

#### 2. Dynamic Expansion & Unmasking (Derivation)
When a newly spawned tile might trigger a merge with a large number (e.g., an existing 128 tile colliding with a newly spawned 128), the algorithm performs an "Expose" (or "Derive") operation to restore the mask to a specific value.

- **Exhaustive Position Exposure**: Since a masked position contains multiple 32768 placeholders and represents a set of possible states, the specific location of a value cannot be predicted. Therefore, the algorithm generates positions for that value in all possible mask locations.
- **Overhead Analysis**: Most exposed positions correspond to reachable game states; thus, unmasking introduces negligible computational overhead.

---

### 6.3.2 Heuristic Pruning Rules

To maximize efficiency, the Advanced Algorithm introduces specific pruning logic:

#### 1. Large Number Combination Limits
Restrictions are placed on the types and quantities of co-existing large numbers:

- Aside from the smallest among the large numbers, all other large tiles may exist at most once.
- If the smallest large tile $> 64$, it may co-exist at most twice.
- If the smallest large tile $= 64$ and no 128 tile is present, it may co-exist at most three times.
- **Valid**: `(4K, 2K, 512, 128, 128)`, `(512, 64, 64, 64)`
- **Pruned**: `(4K, 2K, 512, 128, 128, 128)` or `(4K, 2K, 512, 128, 128, 64)`

#### 2. Small Number Sum Limit (STSL)
The hyperparameter **SmallTileSumLimit (STSL)** precisely controls the complexity of small tiles:

- The sum of all small tiles $\le STSL + 64$.
- If a position contains two identical large numbers, the small sum must be $\le STSL$.
- If three 64s are present, the small sum must be $\le STSL - 64$.
- **Single-step limit**: Each move may generate at most one new large tile.

In practice, small-number-sum pruning is only required every 32 layers.

### 6.3.3 Batch Solving Mechanism

The core of the advanced solving phase is treating "masked positions" as vectors for bulk evolution, utilizing transformation encoding to maintain ordering consistency.

#### 1. Full Unmasking & Cardinality

- **Full Unmasking**: Restoring all mask bits based on the large number combinations.
- **Ordering**: Guarantees a strictly ascending sequence of positions.
- **Unmasked Cardinality**: The total count of positions after unmasking, determined by the large number combinations.
- **Data Structure**: Each masked position is associated with a **Success Rate Array**. The array length equals its "Unmasked Cardinality," mapping 1-to-1 with the position sequence.
- **Note**: Full unmasking is computationally expensive; the algorithm avoids repeated unmasking via the masking mechanism.

#### 2. Transformation Encoding & Cache Tables

- **Problem**: The sequence order of unmasked positions is scrambled after a "Move + Canonicalize" transformation. This causes the retrieved success rate array to misalign with the current board state, preventing expectation calculations.
- **Labeling & Tracing**: Before a transformation, the algorithm traverses all mask bits (`0xf`) and assigns unique descending "labels" (`0xf, 0xe, 0xd...`). After the transformation, it observes the final positions of these labels and compresses the arrangement into a unique feature value `ind`.
- **Pattern**: Masked positions with the same `ind` share identical sequence perturbation patterns.
- **Mapping Repair**: The feature `ind` is used to retrieve a `ranked_array` from the cache table, which records the index mapping from the "perturbed sequence" back to the "original ascending sequence."
- **Effect**: Using `Success Rate Array[ranked_array]` achieves $O(n)$ alignment, avoiding redundant full unmasking and sorting within the solving loop.

#### 3. Dynamic Classification

- **Cardinality Change**: Spawning a new large tile changes the Unmasked Cardinality. The algorithm must re-evaluate the large number combination and filter or merge success rate arrays accordingly.
- **Cardinality = 1**: Reverts to standard algorithm logic.

#### 4. Advantage Summary

- Batch solving dramatically reduces the frequency of binary searches.
- Bulk processing of success rate arrays offers high parallelization potential.
- Supports block-by-block solving to further reduce peak memory usage during massive computations.

### 6.3.4 Algorithm Advantages

The Advanced Algorithm overwhelmingly outperforms the standard algorithm on large-scale tables:

| Dimension | Standard Algorithm | Advanced Algorithm |
| :--- | :--- | :--- |
| **Computation Time** | Super-linear slowdown as tables grow | Faster as tables grow (10x+ speedup) |
| **Memory Usage** | Min 2x single-layer memory | Peak memory reduced by 1 order of magnitude |
| **Parallelization** | Parallelizable | Significantly higher parallelization potential |
| **Storage Efficiency** | Large files; requires heavy compression | Refined intermediate files; smaller final tables |

Using this algorithm, a `free12-4096` table was successfully computed on a 9950X system with 128GB RAM in approximately 24 days.

### 6.3.5 Usage Limitations & Considerations

- **Applicable Scope**: Designed specifically for large-scale tables like `free10`, `free12`, and `4442ff`. For small tables like `L3` or `t`, the advanced algorithm may actually decrease performance.
- **Accuracy Loss**: Pruning causes minor deviations in success rates, but these rarely affect the selection of the optimal move.
- **Feature Conflicts**: Does not support the "keep only optimal branches" option, variant tables, or tables with the 't' suffix (and other patterns with restricted-movement large tiles).
- **Compression Characteristics**: Standard algorithms have high compression ratios because position data is highly redundant. The Advanced Algorithm has a lower ratio because it stores mostly success rate data. High-precision success rate data has high entropy (approaching a random sequence), making traditional lossless compression less effective.


# 7. About the AI

## 7.1 AI Performance
The strongest 2048 AI available.

- Tested over 1200 games without undo: **65536 success rate: 8.4% (±1.6%), 32768 success rate: 86.1% (±2.0%)**.

| Search Depth      | Games | Avg Score  | Median Score | Moves/s |
|-------------------|-------|------------|--------------|---------|
| 1~9 Adaptive      | 1200  | 772,353    | 819,808      | 11      |

- Segmented Success Rates:

| Milestone         | Success Rate | Comparable Score |
|-------------------|--------------|------------------|
| 8k                | 100%         | 97,000           |
| 16k               | 99.9%        | 210,000          |
| 32k               | 86.1%        | 453,000          |
| 32k + 16k         | 72.8%        | 663,000          |
| 32k + 16k + 8k    | 61.8%        | 760,000          |
| 32k + 16k + 8k + 4k| 53.3%       | 804,000          |
| Final 2k          | 46.3%        | 824,000          |
| 65k               | 8.4%         | 971,000          |

- Table Usage Distribution:

| Search | free12-2k | free11-2k | 4442f-2k | free11-512 |
|--------|-----------|-----------|----------|------------|
| 26.49% | 13.38%    | 1.41%     | 51.50%   | 7.22%      |

## 7.2 AI Testing
Testing requires basic programming knowledge. Run `AItest.py` independently; the `run_test` function will write individual game records to the specified directory.

## 7.3 General AI
The AI is compatible with any calculated table. 
However, it currently does not adjust its overall strategy based on available table combinations, making it difficult to utilize `t` or `444` tables effectively on their own. 
To maximize AI performance, it is recommended to calculate tables in increasing order of size: `L3 \ LL \ 4431 \ 4432f \ free10 \ 4442ff \ free11 \ 4442f \ free12`. The more and larger the tables calculated, the stronger the AI becomes.

## 7.4 AI Engine & Dispatcher System

The endgame is the most challenging phase of 2048. This AI relies on search-based decision-making in the early game and switches to the Table Dispatcher system during specific endgame stages, utilizing pre-calculated global optimal solutions to ensure success.


### 7.4.1 Expectimax Search

- **Adaptive Depth**: Search depth automatically adjusts based on the number of empty tiles, typically ranging from 2 to 5 layers.
- **Evaluation Dimensions**: Evaluates the rationality of tile arrangements using row/column lookup tables.
- **Caching Mechanism**: Features a built-in hash-based cache table to eliminate redundant computations.


### 7.4.2 Table Dispatcher
The dispatcher's logic follows a specific pipeline: Identify Current Endgame State -> Match Corresponding Table -> Extract Optimal Solution.

- **Table Priority**: Tables are categorized by "Large Tile Count + Target Exponent." The system prioritizes tables that match the current endgame level with fewer constraints (e.g., `free` tables). 
- **Fail-safe**: If no loaded table provides a valid success rate for the current state, the system automatically reverts to Search Mode.

---

# 8. FAQ

Tip: Press Ctrl + F to search the entire manual.

### Q1: Why does the software take a long time to start or switch features?
**A1**: The software uses **Numba JIT (Just-In-Time)** compilation to optimize calculation modules. Upon the first call, code is compiled into CPU-optimized machine code, resulting in an initialization delay of 2–10 seconds. This is normal behavior.

### Q2: How do I migrate my settings and mistake records when upgrading versions?
**A2**: Steps to preserve your configuration and Mistake Notebook:
1. Copy `mistakes_book.pkl` from the root directory and the `config` file from the `_internal` folder of the old version.
2. Paste and overwrite them in the corresponding paths of the new version.
3. Apply the same process to any other custom configuration files.

### Q4: Why is memory usage extremely high during calculation?
**A4**: Solutions:
1. **Segmentation**: The program has a built-in segmentation algorithm, but memory may still be a bottleneck if physical RAM is very low.
2. **Virtual Memory**: Set your SSD virtual memory (paging file) to 64GB or higher.
3. **Enable Chunked Backward Solving**: When using the Advanced Algorithm, check this option during the solving phase to trade disk I/O for reduced memory consumption.

### Q5: How do I safely resume calculation after an unexpected interruption?
**A5**: Unexpected interruptions can corrupt files. Recovery steps:
1. **Safe Rollback**: Check the last 2–3 modified files in the output folder. If the file size is abnormal (e.g., 0KB or significantly smaller than preceding files), delete them manually.
2. **Resume**: Restart the program, keep the parameters unchanged, and click **BUILD**. The program will automatically resume from the last complete layer.
3. **Log Check**: If errors persist, consult `logger.txt` in the root directory to locate the issue.

### Q6: Calculation is frozen (low CPU usage, no progress, disk I/O hung)?
**A6**: This often occurs when using external mobile hard drives if the drive enters sleep mode or suffers from high latency.
1. **Manual Wake-up**: Attempt to manually copy a small file into that drive to force the system to activate disk writing.
2. **Hardware Optimization**: We recommend disabling power-saving modes for external drives or setting the table calculation path to a local SSD.

### Q7: Why is the AI performing below expectations despite having calculated tables?
**A7**: Troubleshooting steps:
1. **Path Verification**: Switch table names in the **Practice** interface and confirm the path displays the correct folder.
2. **Dispatcher Logic**: If the sum of tiles on the board is very low, the Dispatcher will switch to AI search. Wait for the board to evolve into the specific formation required by the table.

### Q8: UI text is cut off or the board tiles are distorted?
**A8**: The UI adapts to different resolutions but is affected by system scaling (DPI):
1. **Manual Adjustment**: Adjust the "Font Size" parameter in settings.
2. **System Settings**: Right-click `main.exe` > Properties > Compatibility > Change high DPI settings > Override high DPI scaling behavior performed by Application.

# 9. Support and Feedback

## 9.1 Getting Support
If you encounter issues, you can obtain technical support through:

- **GitHub Issues**: Visit the [GitHub project page](https://github.com/game-difficulty/2048EndgameTablebase) to submit an issue. Please describe your problem in detail and provide error logs or steps to reproduce it.
- **Join the Community**:
    - **QQ Group**: 94064339
    - **Discord Channel**: 2048 Runs

## 9.2 Submitting Feedback
- **Star the Project on GitHub**: If you find this project helpful, please give us a Star on the [GitHub project page](https://github.com/game-difficulty/2048EndgameTablebase).

---

# 10. Appendix

## 10.1 Table Size Reference

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
| free8_128    | 3.5 GB | 1 GB        |
| free10_1024  | 2 TB   | 700 GB      |

*Note: Table sizes based on standard algorithms with 32-bit precision and no pruning of low success rate positions.*

**General rules:** For the same endgame level, each additional empty space increases the volume by one order of magnitude. For the same number of empty spaces, `free` tables are approximately three orders of magnitude larger than tables with restricted large tile movement. Volume doubles as the target number doubles.

Setting a high deletion threshold can significantly reduce the final size. Recommended upper limits for thresholds: 16k endgame ~80%, 32k endgame ~40%, 65k endgame ~5%.

## 10.2 Pro-Tips

These tips are intended for players with a deep understanding of table algorithms.

**1. Calculation & Compression Decoupling**
To speed up generation and solving, do not check the compression option during initial calculation. Once finished, keep the parameters identical, check "Compress," and run again. The program will recognize the existing table and perform incremental compression.

**2. Storage Management for Large Tables**
When SSD capacity is insufficient for mega-tables, perform "Hot/Cold Separation":

- **Hot Data**: Keep the current Layer and 20–30 adjacent files on the SSD to ensure I/O performance.
- **Cold Data**: Move completed or currently unnecessary files to an HDD.
- **Virtual Pathing**: Since the program only verifies file existence and not content during breakpoints, you can create empty files/folders in the original SSD path to "trick" the resume check.
- **Multi-path Loading**: Tables can be distributed across multiple drives. You don't need to merge them; simply enter all disk paths in the Practice interface to enable cross-drive recognition.

**3. Custom Table Definitions**
Modify `patterns_config.json` to customize table parameters:

- **valid pattern**: Bitmask for the legal space of large tiles with restricted movement.
- **target pos**: Legal output locations for the target tiles.
- **canonical mode**: Symmetry rules and canonicalization functions.
- **seed boards**: Initial position encodings (must have an identical sum of tiles).
- **extra steps**: Additional steps. Total steps = (Target Tile / 2) + extra steps. (Can be negative, provided the target tile can be merged within that step count).


Examples:
```json
{
  "L3f":{
    "category": "10 space",
    "valid pattern": ["0x000000000fff0000","0x0000000f0fff0ff0","0x000f000f0ff00ff0",
                    "0x000000f00fff0f0f","0x000000ff0fff0f00",
                    "0x00000f000fff00ff","0x00000f0f0fff00f0",
                    "0x00000ff00fff000f",
                    "0x000000f00fff0ff0","0x00000f000fff0ff0",
                    "0x00000ff00fff0f00",
                    "0x00000f000fff0f0f",
                    "0x00000f000fff00ff","0x00000fff0fff0000"],
    "target pos": "0xffffffffffffffff",
    "canonical mode": "identity",
    "seed boards": ["0x100000001fff2fff","0x000000012fff1fff"],
    "extra steps": 48
  },
  "3x3from512to1k": {
    "category": "variant",
    "valid pattern": [],
    "target pos": "0xfff0fff0fff00000",
    "canonical mode": "min33",
    "seed boards": ["0x000f000f009fffff"],
    "extra steps": -200
  },
  "3x3from1kto512": {
    "category": "variant",
    "valid pattern": [],
    "target pos": "0xfff0fff0fff00000",
    "canonical mode": "min33",
    "seed boards": ["0x000f000f00afffff"],
    "extra steps": 30
  }
}
```


Avoid using `_` in pattern names.

Restart the software after modifications to load and calculate your custom table.

## 10.3 Formation Info


| Formation | Example Initial Setup                                                  |
|-----------|------------------------------------------------------------------------|
| 442/442t  | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ x x ```<br/>```x x x x``` |
| L3/L3t    | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ x x x ```<br/>```_ x x x``` |
| t         | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```x _ x x ```<br/>```x _ x x``` |
| 444       | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```x x x x``` |
| LL        | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ x x ```<br/>```_ _ x x``` |
| 4431      | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ x ```<br/>```_ x x x``` |
| 4432f     | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ x ```<br/>```f _ x x``` |
| 4432      | ```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ _ x ```<br/>```_ _ x x``` |
| 4442ff    | ```_ _ f f ```<br/>```_ _ _ _ ```<br/>```_ _ _ _ ```<br/>```_ _ x x``` |
| 3x3       | ```_ _ _ ```<br/>```_ _ 2 ```<br/>```_ _ _```                          |


