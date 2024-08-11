# 2048EndgameTablebase: Overview
Experience the pinnacle of AI and endgame generation with our 2048 tablebase generator, the fastest and most space-efficient solution available. 
Outperforming its contemporaries, this AI boasts the highest success rates for 32k and 65k challenges, approaching theoretical limits.

![Practise](https://github.com/game-difficulty/2048EndgameTablebase/assets/169589278/723510f5-a434-4640-bc82-9eadde41a0ed)
![Minigames](https://github.com/user-attachments/assets/c08e4d0b-2767-4b99-8a66-dc8160bef9a6)



## Features
-  **Highly Optimized Generator**: A robust endgame generator that significantly outperforms other available tools in terms of speed and efficiency.

-  **Effective Training Interface**: Includes a user-friendly GUI that aids players in training and enhancing their skills.

-  **Advanced Data Compression**: Supports data storage in a compressed format, enabling efficient searching without compromising performance.

-  **Resource Requirements**: Ensure 32-64GB of virtual memory is available for processing large tables such as 4431_2048, free9_256.

-  **Patience Required**: All functions are highly optimized, yet require patience during initial use.


## User Guide
1. Get started
   - Open **main.exe** or run **main.py** (Make sure to use `numpy 2.0`) 
   - Wait for the program to start
   - Take note of the **help** documentation
2. Calculate your first table:
   - Go to **Settings**
   - Select a pattern, *e.g. L3-512-0*
   - Specify a save path, *e.g. F:/L3_512_0*
   - Choose whether to compress the data
   - Click **Build** to initiate the calculation
3. View your first table:
   - Go to **Practise**
   - From the upper left menu bar, select a pattern, *e.g. L3-512-0*
   - Enter a file path where the table is saved, *e.g. Q:/L3_512_0*
   - Check the **Show** checkbox to view success rates
   - Click **Default** to display the initial positions
   - Wait for the success rates to show
4. Find the optimal moves:
   - Try rotating the board with **UD RL R90 L90**
   - Try to perform the optimal move with **ONESTEP**
   - Try **Demo** for continuous moves
   - Try to record a game demo (do not make your moves during this recording)
   - Try setting up the board using the provided tiles
5. Real-time Training:
   - Go to **Test**
   - Select a pattern from the upper left menu bar, *e.g. L3-512-0*
   - Execute what you consider the best move within the formation (wasd or ↑↓←→)
   - Utilize the real-time analytics on the right-hand side to imporve your skills
   - Click **Save Logs** to keep a record of your game analysis
6. Enjoy the AI：
   - Calculate *LL_4096_0* & *4432_512* in **Settings**
   - Go to **Game**
   - Click **New Game** to start a new game
   - Enable the AI by clicking the **AI: ON** button and watch the AI play

##  AI Performance
-  This AI reaches the 65536 tile 5.8% (±1.3%) of the time and the 32768 tile 78.5% (±2.5%) of the time without undos.
  -  The following table gives the AI performance statistics and success rates by stage
   
     | depth    | games | score/game | moves/s | seconds/game |
     |----------|-------|------------|---------|--------------|
     | adaptive | 1200  | 720087     | 26      | 978          |


     | milestone     | success rate | comparable score |
     |---------------|--------------|------------------|
     | 8k            | 100%         | 97000            |
     | 16k           | 99.3%        | 210000           |
     | 32k           | 78.5%        | 453000           |
     | 32k+16k       | 65.6%        | 663000           |
     | 32k+16k+8k    | 53.4%        | 760000           |
     | 32k+16k+8k+4k | 43.0%        | 804000           |
     | final 1k      | 28.5%        | 833000           |
     | 65k           | 5.8%         | 971000           |


![survival rate](https://github.com/game-difficulty/2048EndgameTablebase/assets/169589278/4b1a4bd8-3f3c-4fcb-9740-afde4f19889f)

-  The tables used by the AI are LLs (mainly LL-4096), 4431-2048, 4441-512, 4432-512, free9w-256, free10w-1024.
-  If you want to run the AI, it is recommended to just calculate LL-4096 and 4432-512. This will take 1-2 days without compression.
-  The following table shows how often each table is used.
   
     | Search | LL    | 4431 | 4441 | 4432  | free9 | free10 |
     |--------|-------|------|------|-------|-------|--------|
     | 12.5%  | 40.4% | 1.6% | 3.1% | 40.9% | 1.1%  | 0.4%   |

### Role of AI components
-  **Search**: Utilizes an Expectimax search with a specially designed evaluation function to handle simple situations and bridge different formations effectively.
-  **LL**: The heart of the AI, enhancing T formations which are superior to DPDF and snake in 32k and 65k endgames.
-  **4431**: Operates during the transitional stages where the four large tiles are not arranged in LL formations.
-  **4432(4441)**: Preventing simple mistakes that could lead to unnecessary terminations.
-  **free**: Provide optimal strategies for 32k and 65k endgames. However, their actual boost to LL is minimal.


## Project Structure

![Project Structure](https://github.com/game-difficulty/2048EndgameTablebase/assets/169589278/b273c818-8985-4844-adef-350d00f6525c)

