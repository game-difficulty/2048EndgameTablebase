# Instructions

## A. User Guide

### Get started
   - Open **main.exe** or run **main.py**
   - Wait for the program to start
   - Take note of the **help** documentation

### Calculate your first table:
   - Go to **Settings**
   - Select a pattern, *e.g. L3-512-0*
   - Specify a save path, *e.g. F:/L3_512_0*
   - Choose whether to compress the data
   - Click **Build** to initiate the calculation

### View your first table:
   - Go to **Practise**
   - From the upper left menu bar, select a pattern, *e.g. L3-512-0*
   - Enter a file path where the table is saved, *e.g. Q:/L3_512_0*
   - Check the **Show** checkbox to view success rates
   - Click **Default** to display the initial positions
   - Wait for the success rates to show

### Find the optimal moves:
   - Try rotating the board with **UD RL R90 L90**
   - Try to perform the optimal move with **ONESTEP**
   - Try **Demo** for continuous moves
   - Try to record a game demo (do not make your moves during this recording)
   - Try setting up the board using the provided tiles

### Real-time Training:
   - Go to **Test**
   - Select a pattern from the upper left menu bar, *e.g. L3-512-0*
   - Execute what you consider the best move within the formation (wasd or ↑↓←→)
   - Utilize the real-time analytics on the right-hand side to imporve your skills
   - Click **Save Logs** to keep a record of your game analysis

### Enjoy the AI：
   - Calculate *LL_4096_0* & *4432_512* in **Settings**
   - Go to **Game**
   - Click **New Game** to start a new game
   - Enable the AI by clicking the **AI: ON** button and watch the AI play


## B. About the Formations

* **Parameters Defined**: Each formation consists of three parameters: pattern, target, and position.

* **Positional Parameters**: Only the 444, LL, and L3 patterns support positional parameters. For patterns without positional support, any position can be selected arbitrarily.

* **Grid Symbols**:
    - ■ represents a large number grid.
    - □ represents a small number grid.
    - ☒ represents the target number grid.

### 444

* Positional parameter 0:  

        □□□□  
        □□□□  
        □□□□  
        ☒■■■  

* Positional parameter 1:  

        □□□□  
        □□□□  
        □□□□  
        ■☒■■  

### 4431

        □□□□  
        □□□□  
        □□□☒  
        □■■■  

### LL

* Positional parameter 0:  

        □□□□  
        □□□□  
        □□☒■  
        □□■■

* Positional parameter 1:  

        □□□□  
        □□□□  
        □□■■  
        □□☒■

### L3

* Positional parameter 0:  

        □□□□  
        □□□□  
        □☒■■  
        □■■■  

* Positional parameter 1:  

        □□□□  
        □□□□  
        □■☒■  
        □■■■  

* Positional parameter 2:  

        □□□□  
        □□□□  
        □■■☒  
        □■■■  

### T

        □□□□  
        □□□□  
        ☒□■■  
        ■□■■  

### 442

        □□□□  
        □□□□  
        □□☒■  
        ■■■■  

### 4441

        □□□□  
        □□□□  
        □□□□  
        □☒■■  

### 4432

        □□□□  
        □□□□  
        □□□■   
        □□☒■  

### Free & Free w

All tiles can move freely in these formations.

* Examples:
    - free9-256: With 6 unmergeable large number tiles and a 256 tile, the remaining 9 grids are used to create another 256 tile to merge with the existing one.
    - free9w-256: With 7 unmergeable large number tiles, the remaining 9 grids are used to build a 256 tile at any position.

## C. Typical size reference

 - The table below shows the approximate sizes of each formation, their sizes after compression, and the computation time required (with SSD).
    
    | Formation    | ~Size  | ~Compressed | ~Time Taken |
    |--------------|--------|-------------|-------------|
    | LL_4096_0    | 1.1 TB | 450 GB      | 12 h        |
    | 4431_2048    | 650 GB | 250 GB      | 7.5 h       |
    | 444_2048     | 130 GB | 60 GB       | 1.5 h       |
    | t_512        | 2.5 GB | 1 GB        | 1.5 min     |
    | L3_512_0     | 2 GB   | 1 GB        | 1.5 min     |
    | 442_512      | 2 GB   | 1 GB        | 1.5 min     |
    | 4432_512     | 250 GB | 100 GB      | 3 h         |
    | 4441_512     | 800 GB | 330 GB      | 9 h         |
    | free9_256    | 460 GB | 170 GB      | 20 h        | - 
    | free10w_1024 | 2 TB   | 700 GB      | 80 h        |

### Additional Notes

* Supports breakpoints during calculations.
* Compression is slow. For faster results, calculate without compression first, then select the "Compress" option later to compress separately.
