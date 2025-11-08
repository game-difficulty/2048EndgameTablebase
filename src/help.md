# User Manual

**Author**: game_difficulty  
**Version**: 7.8  
**Date**: 2025.11.3  

---

# 1. Overview

This is a multifunctional 2048 game training software, offering a variety of features including endgame calculation, endgame training, game analysis, AI, modified mini-games, and more.

## 1.1 Highlights
- **Fast Calculation**: Uses highly optimized algorithms to provide extremely fast computation speeds.
- **Low Space Occupation**: Employs multiple data compression techniques to significantly reduce disk space usage.
- **The Strongest AI**: Features the strongest 2048 AI, with a 65536 success rate far exceeding current AI.
- **Comprehensive Features**: Provides complete training and analysis tools to help users deeply learn game strategies.

## 1.2 Target Users
- **Professional Players**: Players who need more powerful tools to study game techniques and strategies.
- **AI Developers**: Developers who want to use game data to train and optimize AI.
- **Casual Players**: Players who want to experience modified mini-games or improve their scores with AI assistance.

---

# 2. Installation and Startup

## 2.1 System Requirements
- **Operating System**: Windows 7 and above.
- **Memory**: At least 4GB, the more, the better, virtual memory is recommended.
- **Disk Space**: The larger the disk space, the better. An SSD with at least 10GB of free space is recommended.
- **CPU**: CPUs supporting AVX-512 instruction set will perform better.
- **GPU**: The software does not rely on GPU.

## 2.2 Installation Steps
- **Extract**: Download and extract the software package.
- **Run**: Find the `main.exe` file in the extracted folder and double-click to run the software.

## 2.3 Startup Instructions
- **Startup**: The first time you use some features, initialization may take a while, please be patient.
- **Error Troubleshooting**: If the software fails to start or shows an error, first check the `logger.txt` file, which may contain detailed information or error logs related to the startup failure.

---

# 3. Concepts

## 3.1 What is an Endgame?

An endgame starts from a specific setup where some large numbers are already placed in certain tiles, and the player must use the remaining spaces to merge numbers and achieve a target number, thus advancing the game. **An endgame refers to the entire process of reaching the target number from a specific setup.**

## 3.2 Classification of Endgames

**Endgames are classified based on the number of remaining tiles and the difficulty of the endgame.**

For example, a 10-space 16k endgame (16384 endgame) refers to a setup where large numbers such as 8k, 4k, 2k, 1k, 512, and 256 are already placed in 6 tiles. The remaining 10 spaces need to form a 256, which is considered a "16k difficulty".

Similarly, a 12-space 32k endgame (32768 endgame) means that 16k, 8k, 4k, and 2k are already placed, and the remaining 12 spaces must form 2048 to eventually reach 32768.

If the setup already contains 32k, 16k, 4k, and 2k, and the player needs to use 12 empty spaces to form 2048, it is still considered a 12-space 32k endgame because this process is equivalent to forming a 32768.

Therefore, the classification of an endgame only depends on how many large numbers are in the setup, not which specific large numbers are present.

Question: What is the target number for a 9-space 65k endgame?

## 3.3 What is a Formation?

**A formation is an endgame with additional constraints.**

In the game, it is generally better to keep the large numbers stationary, while the target number must be formed at specific positions to maintain the formation. Therefore, the constraints usually include:
- The large numbers in the initial setup must stay in place or can only move in limited ways.
- The target number must be formed at specified positions.

If no constraints are added to the endgame, it is called a **free formation**.

## 3.4 Classification of Formations

Formations are classified based on the initial setup, constraints, and target number (endgame difficulty). Detailed classifications can be found in the appendix.

---

# 4. Feature Overview

The software's main menu contains multiple functional interfaces, each with different feature modules. You can quickly access these modules via buttons in the main interface. Below is a brief introduction to the software's various functional interfaces:

## 4.1 MainMenu
After launching the software, you will enter the Main Menu, which contains the following 6 functional interfaces:

- **Game**: Provides the basic game functionality and AI.
- **MiniGame**: Offers various modified mini-games with higher difficulty levels.
- **Practise**: Shows the optimal moves to help you learn game strategies.
- **Test**: Tests your mastery of endgames.
- **Settings**: The global settings and table calculation.
- **Help**: Displays the help documentation to guide you on how to use the software.

## 4.2 Settings
Includes table calculation and global setting changes.

- **Adjust Settings**: Customize the game's tile colors, 4-spawn rate, animation effects, etc.
- **Table Calculation**: Select the table you want to calculate, specify the save path, and set advanced parameters. Click the "BUILD" button to begin the calculation. The calculation process may take some time, depending on the complexity of the endgame and your computer's performance.

## 4.3 Game
Provides the basic 2048 game functionality.

- **AI Module**: Click the "AI: ON" button on the game interface to let the AI play the game automatically. Note that the AI function depends on previously computed tables. Therefore, you must calculate and save the related tables before enabling AI.
- **Hard Mode**: At the bottom of the game interface, you can find a "Difficulty" slider. Drag it all the way to the right to enable Hard Mode, which significantly increases the game difficulty.

## 4.4 MiniGame
Offers a variety of modified mini-games with higher difficulty levels, ideal for players who enjoy a challenge.

- **Game Rules**: Each mini-game has specific gameplay instructions, which can be tried according to the prompts on the interface.
- **Difficulty**: In the bottom left corner, there is a "HardMode" button. Click it to further increase the game difficulty and challenge your limits.

## 4.5 Practise
Displays the optimal moves to learn game strategies.

- **Practice**: In this interface, you can view the calculated tables, showing the four possible moves for each position and their success rates. By learning the optimal moves, you can improve your game skills.
- **Recording**: Record the optimal moves for an endgame or play back the moves recorded by others.

## 4.6 Test
The Test interface helps you evaluate your mastery of endgames:

- **Setup**: You can select the formation and initial setup to test and practice.
- **Replay Analysis**: Supports importing game replays from 2048verse.com for analysis.

---

# 5. Quick Start

Table calculation is a prerequisite for using important features such as the AI module, practice module, and test module. You need to first calculate and save a **table**.

## 5.1 Calculate a Formation
   - Open the software and enter the main interface.
   - Click **Settings**.
   - Choose the formation you want to calculate (e.g., L3-512-0).
   - Specify the save path for the table (e.g., F:/L3_512_0).
   - Click **Build** to start the calculation.

## 5.2 Learn the Formation
   - Enter **Practise**.
   - In the top left menu (pattern\target\position), select the formation you just calculated.
   - Check the **Show** checkbox to view the success rates.
   - Click **Default** to show a random initial setup.
   - Click **Demo** to view the demo.

## 5.3 Master the Formation
   - Enter **Test**.
   - In the top left menu, select the formation you just calculated.
   - Execute your moves (using WASD or arrow keys for controls).
   - View real-time analysis on the right side.
   - Click **Save Logs** to save the game logs.

---

# 6. Advanced Usage

## 6.1 Formation Calculation Parameters and Options

### Formation Calculation Parameters

| Parameter      | Description                                                                                                            |
|----------------|------------------------------------------------------------------------------------------------------------------------|
| **Pattern**    | Formation name                                                                                                         |
| **Target**     | Target number                                                                                                          |
| **Position**   | The position where the target number should be formed. <br/> Only a few pattern support this option, usually set to 0. |

### Advanced Options

| Button                                      | Function                                          | Purpose                              |
|---------------------------------------------|---------------------------------------------------|--------------------------------------|
| **Compress Temp Files**                     | Compress all intermediate files                   | Reduce disk usage during calculation |
| **Compress**                                | Compress the final .book files                    | Reduce final disk usage              |
| **Opt Only**                                | Keep only the optimal branches                    | Reduce final disk usage              |
| **Remove Boards with a Success Rate Below** | Remove boards with success rate below a threshold | Reduce final disk usage              |

**Note**:
- All of the above options **do not** affect the accuracy of the success rates.
- The **"Remove Boards with a Success Rate Below"** and **"Opt Only"** options will delete some boards (usually unreasonable ones) and display a success rate of 0, but **will not** affect other data.

### Advanced Algorithm Option  
When **Advanced Algo** is checked, the program will adopt an advanced algorithm. Tiles ≥64 and <32k are defined as *large numbers*, and those ≤32 as *small numbers*. 
The algorithm will prune positions that do not meet the following conditions:   

1. **Large number combinations**:  
   a. Except for the smallest large number, all other large numbers may exist at most once.  
   b. If the smallest large number is >64, it may exist at most twice.  
   c. If the smallest large number is 64, it may exist at most three times.  
   Examples: Valid combinations (4k 2k 512 128 128), (1k 512 256), (512 64 64 64); Invalid combinations (4k 2k 512 128 128 128), (4k 2k 512 128 128 64).  

2. **Sum of small numbers**:  
  Controlled by the hyperparameter **SmallTileSumLimit (stsl)**:  
   a. Sum of small numbers ≤ stsl + 64  
   b. If the board contains two identical large numbers: sum of small numbers ≤ stsl  
   c. If the board contains three 64s: sum of small numbers ≤ stsl - 64  

3. **Single-step merge limit**:  
  Each operation produce at most one new large number.  

**Advantages compared to the standard algorithm**:  
- **Faster computation**: Larger formations (e.g., free10w, 4442ff and above) achieve over 10× speedup.  
- **Memory optimization**: Peak memory usage reduced by 1-2 orders of magnitude. Up to 100× speedup under memory constraints.  
- **Storage efficiency**: Smaller intermediate files (space savings increase with table size); book files require no compression.  

**Usage limitations**:  
- Designed specifically for extremely large formations, not suitable for small formations like L3 or T (may perform worse).  
- Pruning reduces success rate accuracy. Accuracy loss increases with deviations from optimal paths, higher endgame difficulty, and stricter formations constraints. Typical loss is within 0.01%, with no impact on optimal paths.  
- **opt only** and **compress** options are unsupported.  

Using this algorithm, free12w-2048 was successfully computed on a 9950x 128GB RAM computer in ~15 days.   

## 6.2 Practise Interface Button Functions

### Practise Interface Button Functions

| Button                  | Function                                                                                                                                                                                                                                                                                                                                                                             |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Set**                 | Reset the board to the value entered in the left box.                                                                                                                                                                                                                                                                                                                                |
| **0-32k Color Buttons** | Click a number button to enter the set-board mode. During this mode, moves are not executed, and the success rate will not refresh. <br/> In this mode, you can use different keys to set the numbers on the board: <br/>1. Left-click - Change to the number on the button  <br/>2. Right-click - Increase the number by one level <br/>3. Other - Decrease the number by one level |
| **Undo**                | Undo the previous action                                                                                                                                                                                                                                                                                                                                                             |
| **Demo**                | Execute the optimal move continuously                                                                                                                                                                                                                                                                                                                                                |
| **OneStep**             | Execute one optimal move at a time                                                                                                                                                                                                                                                                                                                                                   |
| **Default**             | Show a random initial setup                                                                                                                                                                                                                                                                                                                                                          |
| **UD/RL/R90/L90**       | Rotate and flip the board                                                                                                                                                                                                                                                                                                                                                            |
| **Record Demo**         | Record and save the demo moves and success rates                                                                                                                                                                                                                                                                                                                                     |
| **Load Demo/Play Demo** | Load and play the recorded demo file                                                                                                                                                                                                                                                                                                                                                 |
| **Manual**              | Manual mode, no automatic number spawn. Left-click to place 2, right-click to place 4.                                                                                                                                                                                                                                                                                               |
| **Set...**              | Choose a path to load. **If the calculated table file has been moved, it must be reset.**                                                                                                                                                                                                                                                                                            |

## 6.3 Testing Interface Advanced Features

You can **set the initial setup**. If you are unsure about the encoding method, you can arrange the initial setup in the Practise interface and then copy the encoding.

You can use the **Analyze Verse Log** button to analyze game replays from 2048verse.com and generate a detailed analysis report.

   - Write the game replay to a txt file and upload it.
   - Choose a formation to analyze. You must have calculated this table.
   - **If the book files have been moved, they must be reset in the Practise interface.**
   - The program will analyze all endgames in the replay that meet the formation constraints and generate a txt report.

Use the **Review Replay** button to reexamine your gameplay on Testing screen or Verse.

   - Entering the interface will allow you to reexamine the test you have just taken.
   - You can select the replay file you want to review from the left side of the menu bar in the interface.
   - Replay files will be generated when you analyze a Verse replay.

The **Mistakes Notebook** is at the bottom of the test interface. It automatically records mistakes made during tests, so you can practice them repeatedly.

To retain your settings and Mistakes Notebook after updating:
Find the *config* file (without an extension) and the mistakes_book.pkl file in your previous installation folder, then copy them to the same locations in the new version's folder.

---

# 7. About AI

## 7.1 Prerequisites

To fully utilize the AI's performance, the following tables need to be computed:
   - **free12w-2048**
   - **free11w-2048**
   - **4442f-2048**
   - **free11w-512**

Ensure these tables can be properly loaded in the **Practise** before running the AI. As these tables require significant disk space, to minimize storage usage, it is recommended to:
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
2. **Table path changed**: If the table file was moved, reset its path in the Practise interface.
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
