


# 说明


## A.   Highlights


l 
一个大量优化过的2048游戏残局库生成器和训练器


l 
支持以压缩格式储存数据，且能够高效查找


l 
所有功能在第一次使用时需要耐心等待加载


l 
为保障计算速度，大型定式计算需要大量RAM，建议提供32-64G虚拟内存


## B.   关于定式


l 
定式生成工具在设置中


l 
定式三个参数分别是样式、目标和位置


l 
支持自定义保存路径


l 
■表示大数字格子，□表示小数字格子，☒表示目标数字格子


l 
对不支持位置参数的定式，可随意选择一个位置参数


l 
444


支持位置参数


位置参数0:


□□□□


□□□□


□□□□


☒■■■


位置参数1:


□□□□


□□□□


□□□□


■☒■■


l 
4431


不支持位置参数


□□□□


□□□□


□□□☒


□■■■


l 
LL


支持位置参数


位置参数0:


□□□□


□□□□


□□☒■


□□■■


位置参数1:


□□□□


□□□□


□□■■


□□☒■


l 
L3


支持位置参数


位置参数0:


□□□□


□□□□


□☒■■


□■■■


位置参数1:


□□□□


□□□□


□■☒■


□■■■


位置参数2:


□□□□


□□□□


□■■☒


□■■■


l 
t


不支持位置参数


□□□□


□□□□


☒□■■


■□■■


l 
442


不支持位置参数


□□□□


□□□□


□□☒■


■■■■


l 
4441


不支持位置参数


□□□□


□□□□


□□□□


□☒■■


l 
4432


不支持位置参数


□□□□


□□□□


□□□■


□□☒■


l 
Free & Free w


不支持位置参数


所有位置均可自由移动


例：free9-256是指存在6个大数格子和一个256格子的情况下，使用剩下9个格子合成一个256并与已有256合并

例：free9w-256是指存在7个大数格子的情况下，使用剩下9个格子在任意位置合成一个256

l 
典型定式大小参考（未压缩）：


444-2048:130G


4431-2048:650G


LL-2048:520G


L3-512:2G


Free8-128:45G


Free9-256:460G


Free10-512:未知


4441-4096:未知


4432-4096:未知


l 
压缩后可节约60%左右的空间，但是在计算过程中仍需要未压缩大小的磁盘空间。压缩过程较慢，
追求计算速度可选择先不压缩，等全部计算完成再勾选压缩选项单独进行压缩过程。L3定式可不压缩


l 
支持断点续接。使用者可自行发挥，通过更改已有文件后缀、
创建虚假空文件等方式，将一个大定式分散储存在不同位置


## C.   其他


l 
内置一个可以使用LL-2048和4431-2048的AI，但需本地已下载过对应的定式


l 
训练页面功能包括修改棋盘、旋转翻转棋盘、自动演示等功能


l 
设置页面可修改格子颜色、演示速度等，不建议打开动画，会很卡


l 
目前不支持65536


# Instructions


## A.   Highlights


l 
A heavily optimized 2048 endgame tablebase
generator and trainer.


l 
Supports storing data in a compressed format
while enables efficient searching.


l 
All functions require patience during the first
use.


l 
For calculation speed, large pattern calculations
require a significant amount of RAM, it is recommended to provide 32-64G
virtual memory.


## B.   About the Pattern


l 
Generator is in the Settings.


l 
The three parameters are pattern, target and
position.


l 
Supports save to customized path.


l 
■ means large number grid, □ means small number grid, ☒ means
target number grid.


l 
For patterns that do not support the position
parameter, you can choose one at will.


l 
444


Position parameters are supported


Position parameter 0:


□□□□


□□□□


□□□□


☒■■■


Position parameter 1:


□□□□


□□□□


□□□□


■☒■■


l 
4431


Position parameters are not supported


□□□□


□□□□


□□□☒


□■■■


l 
LL


Position parameters are supported


Position parameter 0:


□□□□


□□□□


□□☒■


□□■■


Position parameter 1:


□□□□


□□□□


□□■■


□□☒■


l 
L3


Position parameters are supported


Position parameter 0:


□□□□


□□□□


□☒■■


□■■■


Position parameter 1:


□□□□


□□□□


□■☒■


□■■■


Position parameter 2:


□□□□


□□□□


□■■☒


□■■■


l 
t


Position parameters are not supported


□□□□


□□□□


☒□■■


■□■■


l 
442


Position parameters are not supported


□□□□


□□□□


□□☒■


■■■■


l 
4441


Position parameters are not supported


□□□□


□□□□


□□□□


□☒■■


l 
4432


Position parameters are not supported


□□□□


□□□□


□□□■


□□☒■


l 
Free & Free w


Position parameters are not supported


All tiles are free to move.


Example: free9-256 means that in the case where there are 6 large
number tiles and a 256 tile, the remaining 9 grids are used to build a 256 and
merge it with the existing 256.

Example: free9w-256 means that in the case where there are 7 large
number tiles, the remaining 9 grids are used to build a 256 at any position.

l 
Typical size reference (uncompressed):


444-2048:130G


4431-2048:650G


LL-2048:520G


L3-512:2G


Free8-128:45G


Free9-256:460G


Free10-512:Unknown


4441-4096:Unknown


4432-4096:Unknown


l 
Saves about 60% of space after compression, but
still requires disk space of uncompressed size during computation.
The compression process is slow. If you are looking for speed,
you can choose not to compress first, and then check the Compress option to do
the compression process separately when all calculations are completed. L3 pattern
can be kept uncompressed.


l 
Supports breakpoints. Users can also try to
spread a big pattern across different file addresses by changing the suffix of
an existing file, creating a fake empty file, etc.


## C.   Others


l 
Has a built-in AI that can use LL-2048 and
4431-2048, but only after the patterns have been downloaded locally.


l 
The practice page includes functions such as
modifying the board, rotating and flipping the board, and automatic
demonstration.


l 
Settings page can modify the tile color, demo
speed, etc. It is not recommended to open the animation, it will be very laggy.


l 
Does not support 65536 currently.


 


 


 





