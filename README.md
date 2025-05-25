# 2048EndgameTablebase: Overview
Experience the pinnacle of AI and endgame generation with our 2048 tablebase generator, the fastest and most space-efficient solution available. 
Outperforming its contemporaries, this AI boasts the highest success rates for 32k and 65k challenges, approaching theoretical limits.

success rate of getting a 2048 in 12 space (32k endgame)
![Practise](https://github.com/user-attachments/assets/9bd553b8-6156-45b9-ad63-1e7157b641de)



## Features
-  **Highly Optimized Generator**: A robust endgame generator that significantly outperforms other available tools in terms of speed and efficiency.

-  **Effective Training Interface**: Includes a user-friendly GUI that aids players in training and enhancing their skills.

-  **Advanced Algorithms**: Advanced algorithms and compressed file formats are built-in to support the computation of extremely large tables.

-  **Resource Requirements**: Ensure 32-128GB of virtual memory is available for processing large tables such as free11w_1024.

-  **Patience Required**: All functions are highly optimized, yet require patience during initial use.


## User Guide
1. Get started
   - Open **main.exe** or run **main.py**
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
-  This AI reaches the 65536 tile 8.4% (±1.6%) of the time and the 32768 tile 86.1% (±2.0%) of the time without undos.
  -  The following table gives the AI performance statistics and success rates by stage
   
     | search depth    | games | avg score  | median score | moves/s |
     |-----------------|-------|------------|--------------|---------|
     | 1~9 adaptive    | 1200  | 772353     | 819808       | 11      |


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


-  The tables used by the AI are
   - **free12w-2048**
   - **free11w-2048**
   - **4442f-2048**
   - **free11w-512**

-  The following table shows how often each table is used.
     | Search | free12w-2k     | free11w-2k | 4442f-2k | free11w-512  |
     |--------|----------------|------------|----------|--------------|
     | 26.49% | 13.38%         | 1.41%      | 51.50%   | 7.22%        |



