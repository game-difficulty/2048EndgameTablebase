const boardEl = document.getElementById('board');
const scoreEl = document.getElementById('score');
const bestScoreEl = document.getElementById('best-score');
const inputEl = document.getElementById('board-input');
const btnSet = document.getElementById('btn-set');
const btnAuto = document.getElementById('btn-auto');
const btnStep = document.getElementById('btn-step');
const btnUndo = document.getElementById('btn-undo');
const speedSlider = document.getElementById('speed-slider');

let score = 0;
let bestScore = parseInt(localStorage.getItem('2048-best') || '0');
bestScoreEl.innerText = bestScore;
let isAutoRunning = false;

// 移动历史记录 (用于撤销)
let history = [];

// 游戏状态
let tiles = [];
let grid = Array.from({ length: 4 }, () => Array(4).fill(null));

// 动画调度
let animTimeout = null;
let dyingTiles = [];
let hiddenMergedTiles = [];

// 初始化底层固定格
boardEl.innerHTML = '';
for (let i = 0; i < 16; i++) {
    const gridCell = document.createElement('div');
    gridCell.className = 'grid-cell';
    boardEl.appendChild(gridCell);
}

// ==========================================
// 方块对象 (Tile)
// ==========================================
class Tile {
    constructor(x, y, value) {
        this.x = x;
        this.y = y;
        this.targetX = x;
        this.targetY = y;
        this.value = value;

        this.mergedThisTurn = false;
        this.isDying = false; // 标记：动画结束后将被销毁
        this.isJustAdded = false;

        this.element = document.createElement('div');
        this.element.className = 'tile';
        this.inner = document.createElement('div');
        this.inner.className = `tile-inner cell-${value.toString(16)}`;
        this.inner.innerText = value === 0 ? '' : Math.pow(2, value);
        this.element.appendChild(this.inner);

        this.snapPosition();
        boardEl.appendChild(this.element);
    }

    // 瞬间移动（无动画，用于打断上一个操作）
    snapPosition() {
        this.x = this.targetX;
        this.y = this.targetY;
        this.element.style.transition = 'none';
        this.element.style.setProperty('--x', this.targetX);
        this.element.style.setProperty('--y', this.targetY);
    }

    // 开启滑动动画
    animatePosition() {
        this.element.style.transition = '100ms ease-in-out transform';
        this.element.style.setProperty('--x', this.targetX);
        this.element.style.setProperty('--y', this.targetY);
    }
}

// ==========================================
// 核心：快进动画与逻辑渲染引擎
// ==========================================
const vectors = {
    1: { x: -1, y: 0 }, // Left
    2: { x: 1, y: 0 }, // Right
    3: { x: 0, y: -1 },// Up
    4: { x: 0, y: 1 }  // Down
};

// 清理上一帧残留的动画，强制同步 DOM 与逻辑
// isInterrupt 为 true 时表示由于新输入强制打断动画
function fastForwardAnimations(isInterrupt = false) {
    // 1. 强制所有活跃方块瞬间就位
    tiles.forEach(t => {
        t.snapPosition();
        
        // 如果是强行打断，则立刻杀掉所有缩放透明度动画，强制还原到 1:1 状态
        if (isInterrupt) {
            t.inner.classList.remove('anim-new');
            t.inner.classList.remove('anim-merged');
            t.element.style.opacity = '1';
        }
        
        t.mergedThisTurn = false;
        t.isJustAdded = false;
    });

    // 2. 清理需要被销毁的合并源方块
    dyingTiles.forEach(t => {
        if (t.element && t.element.parentNode) {
            t.element.remove();
        }
    });
    tiles = tiles.filter(t => !t.isDying);
    dyingTiles = [];

    // 3. 处理上一帧产生的隐藏块
    // 如果是自然结算 (isInterrupt=false)，开启 Pop 动画
    // 如果是由于新输入打断，我们为了视觉稳定，直接显示结果而不添加新的 Pop 脉冲
    hiddenMergedTiles.forEach(t => {
        t.element.style.opacity = '1';
        if (!isInterrupt) {
            t.inner.classList.add('anim-merged');
        } else {
            t.inner.classList.remove('anim-merged');
        }
    });
    hiddenMergedTiles = [];

    saveGameState();
}

// ==========================================
// 辅助工具：状态转化与持久化
// ==========================================
function getGridHex() {
    let arr = new Array(16).fill(0);
    for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
            if (grid[y][x]) {
                // 即使超过 32768 (val 15)，在 hex 中也只记为 f
                let val = grid[y][x].value;
                arr[y * 4 + x] = Math.min(val, 15);
            }
        }
    }
    let str = '';
    for (let i = 0; i < 16; i++) {
        str += arr[i].toString(16);
    }
    return str;
}

function saveGameState() {
    const hex = getGridHex();
    inputEl.value = hex; // 更新输入框
    localStorage.setItem('2048-board', hex);
    localStorage.setItem('2048-score', score.toString());
    localStorage.setItem('2048-best', bestScore.toString());
}

function loadGameState() {
    const savedBoard = localStorage.getItem('2048-board');
    const savedScore = parseInt(localStorage.getItem('2048-score') || '0');
    if (savedBoard && savedBoard.length === 16) {
        score = savedScore;
        loadFromHex(savedBoard, true); // true 表示保留 score
    }
}

function pushHistory() {
    // 限制历史记录长度，防止内存溢出 (可选, 这里暂不设限)
    history.push({
        board: getGridHex(),
        score: score
    });
}

function applyMove(direction) {
    // 【拦截器】：如果还在动画中又来了新输入，立刻打断并快进，绝不丢弃输入
    if (animTimeout) {
        clearTimeout(animTimeout);
        animTimeout = null;
    }

    // 只有在真正移动前记录历史
    const currentHex = getGridHex();
    const currentScore = score;

    // 强制同步上一帧逻辑并打断已有动画
    fastForwardAnimations(true);

    // 强制浏览器触发重排 (Reflow)，确保 snapPosition 的瞬间移动生效，从而杜绝斜向滑动
    void document.body.offsetHeight;

    const vector = vectors[direction];
    const traversalsX = vector.x === 1 ? [3, 2, 1, 0] : [0, 1, 2, 3];
    const traversalsY = vector.y === 1 ? [3, 2, 1, 0] : [0, 1, 2, 3];

    let moved = false;
    let newGrid = Array.from({ length: 4 }, () => Array(4).fill(null));

    // 纯粹的逻辑推演 (与 DOM 坐标彻底脱钩)
    traversalsY.forEach(y => {
        traversalsX.forEach(x => {
            const tile = grid[y][x];
            if (tile) {
                let currX = x;
                let currY = y;
                let nextX = currX + vector.x;
                let nextY = currY + vector.y;

                // 寻路
                while (nextX >= 0 && nextX < 4 && nextY >= 0 && nextY < 4) {
                    const nextTile = newGrid[nextY][nextX];
                    if (!nextTile) {
                        currX = nextX;
                        currY = nextY;
                    } else if (nextTile.value === tile.value && !nextTile.mergedThisTurn) {
                        currX = nextX;
                        currY = nextY;
                        break;
                    } else {
                        break;
                    }
                    nextX = currX + vector.x;
                    nextY = currY + vector.y;
                }

                if (currX !== x || currY !== y) moved = true;

                const targetTile = newGrid[currY][currX];
                if (targetTile && targetTile.value === tile.value && !targetTile.mergedThisTurn) {
                    // 发生合并
                    tile.targetX = currX;
                    tile.targetY = currY;
                    tile.isDying = true;
                    targetTile.isDying = true;

                    dyingTiles.push(tile);
                    if (!dyingTiles.includes(targetTile)) dyingTiles.push(targetTile);

                    // 创建合并后的新实体 (初始隐藏)
                    const mergedTile = new Tile(currX, currY, tile.value + 1);
                    mergedTile.mergedThisTurn = true;
                    mergedTile.element.style.opacity = '0';

                    tiles.push(mergedTile);
                    hiddenMergedTiles.push(mergedTile);
                    newGrid[currY][currX] = mergedTile;

                    score += Math.pow(2, mergedTile.value);
                } else {
                    // 普通平移
                    tile.targetX = currX;
                    tile.targetY = currY;
                    newGrid[currY][currX] = tile;
                }
            }
        });
    });

    if (moved) {
        // 先记录移动前的状态到历史中
        history.push({
            board: currentHex,
            score: currentScore
        });

        // 更新逻辑网络阵列
        grid = newGrid;

        // 生成新块
        addRandomTileToNewGrid();

        scoreEl.innerText = score;
        if (score > bestScore) {
            bestScore = score;
            bestScoreEl.innerText = bestScore;
        }

        // 触发物理渲染 (滑动)
        tiles.forEach(t => {
            if (!hiddenMergedTiles.includes(t) && !t.isJustAdded) {
                t.animatePosition();
            }
        });

        // 切换为并行模式：逻辑更新完立即请求下一次 AI 计算，不需要等待 100ms 动画
        if (isAutoRunning) triggerAI();

        // 100ms 后清理动画现场，但不在此触发下一次计算
        animTimeout = setTimeout(() => {
            fastForwardAnimations();
        }, 100);

    } else {
        if (isAutoRunning) {
            isAutoRunning = false;
            btnAuto.innerText = "Auto Run";
        }
    }
}

function addRandomTileToNewGrid() {
    const emptyCells = [];
    for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
            if (!grid[y][x]) emptyCells.push({ x, y });
        }
    }
    if (emptyCells.length > 0) {
        const rand = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        const val = Math.random() < 0.9 ? 1 : 2;
        const tile = new Tile(rand.x, rand.y, val);
        tile.isJustAdded = true;
        tile.inner.classList.add('anim-new');
        tiles.push(tile);
        grid[rand.y][rand.x] = tile;
    }
}

// ==========================================
// 外部交互与 AI Worker 通讯
// ==========================================
function loadFromHex(hexStr, keepScore = false) {
    if (animTimeout) {
        clearTimeout(animTimeout);
        animTimeout = null;
    }
    fastForwardAnimations(true);

    tiles.forEach(t => t.element.remove());
    tiles = [];
    grid = Array.from({ length: 4 }, () => Array(4).fill(null));

    if (!keepScore) {
        score = 0;
    }
    scoreEl.innerText = score;

    for (let i = 0; i < 16; i++) {
        const val = parseInt(hexStr[i] || '0', 16);
        if (val > 0) {
            const x = i % 4;
            const y = Math.floor(i / 4);
            const tile = new Tile(x, y, val);
            tiles.push(tile);
            grid[y][x] = tile;
        }
    }
    saveGameState();
}

function triggerAI() {
    if (!isAutoRunning && document.activeElement !== btnStep) return;
    let str = getGridHex();
    aiWorker.postMessage({ type: 'calculate', board_encoded: str });
}

function undo() {
    if (isAutoRunning || history.length === 0) return;

    if (animTimeout) {
        clearTimeout(animTimeout);
        animTimeout = null;
    }
    fastForwardAnimations(true);

    const lastState = history.pop();
    score = lastState.score;
    loadFromHex(lastState.board, true);
}

const aiWorker = new Worker('worker.js', { type: 'module' });

// 立即绑定监听，防止错过 ready 消息
aiWorker.onmessage = function (e) {
    if (e.data.type === 'ready') {
        console.log("AI Worker is ready.");
        // 兜底：如果加载完存档后棋盘仍然是空的，则初始化两个块
        const currentHex = getGridHex();
        if (currentHex === '0000000000000000') {
            console.log("Empty board, initializing new game tiles.");
            score = 0;
            scoreEl.innerText = '0';
            addRandomTileToNewGrid();
            addRandomTileToNewGrid();
            saveGameState();
        }
    } else if (e.data.type === 'move_result') {
        const moveCode = e.data.best_move;
        if (moveCode > 0) {
            applyMove(moveCode);
        } else {
            isAutoRunning = false;
            btnAuto.innerText = "Auto Run";
            btnAuto.classList.remove('active');
        }
    }
};

// 手动把玩拦截
document.addEventListener('keydown', (e) => {
    if (document.activeElement === inputEl) return;
    if (isAutoRunning) return;

    const keyMap = {
        'ArrowLeft': 1, 'a': 1, 'A': 1,
        'ArrowRight': 2, 'd': 2, 'D': 2,
        'ArrowUp': 3, 'w': 3, 'W': 3,
        'ArrowDown': 4, 's': 4, 'S': 4,
        'z': 'undo'
    };

    if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
        undo();
        return;
    }

    if (keyMap[e.key]) {
        if (keyMap[e.key] === 'undo') {
            undo();
        } else {
            e.preventDefault();
            applyMove(keyMap[e.key]);
        }
    }
});

// 速度控制 (通过 time_limit_ratio 映射)
speedSlider.addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    const exponent = (100 - value) / 100.0;
    const ratio = Math.pow(10, exponent);
    aiWorker.postMessage({ type: 'update_speed', ratio: ratio });
});
speedSlider.addEventListener('change', (e) => {
    e.target.blur();
});

btnSet.addEventListener('click', () => {
    const val = inputEl.value.toLowerCase().padStart(16, '0');
    if (/^[0-9a-f]{16}$/.test(val)) {
        loadFromHex(val);
    } else {
        alert("Invalid board hex format!");
    }
    btnSet.blur();
});

// 控制面板
btnAuto.addEventListener('click', () => {
    isAutoRunning = !isAutoRunning;
    if (isAutoRunning) {
        btnAuto.innerText = "Stop Auto";
        btnAuto.classList.add('active');
        triggerAI();
    } else {
        btnAuto.innerText = "Auto Run";
        btnAuto.classList.remove('active');
    }
    btnAuto.blur();
});

btnStep.addEventListener('click', () => {
    if (isAutoRunning) return;
    triggerAI();
    btnStep.blur();
});

btnUndo.addEventListener('click', () => {
    undo();
    btnUndo.blur();
});

boardEl.addEventListener('click', () => {
    if (document.activeElement instanceof HTMLElement) {
        document.activeElement.blur();
    }
});

// ==========================================
// 移动端手势逻辑 (Swipe Detection)
// ==========================================
let touchStartX = 0;
let touchStartY = 0;

boardEl.addEventListener('touchstart', (e) => {
    if (e.touches.length > 1) return; // 多指忽略
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
}, { passive: true });

boardEl.addEventListener('touchend', (e) => {
    if (isAutoRunning) return;
    if (e.changedTouches.length > 1) return;

    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;

    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    // 最小滑动阈值
    if (Math.max(absDx, absDy) > 30) {
        if (absDx > absDy) {
            // 水平滑动
            applyMove(dx > 0 ? 2 : 1); // 2:右, 1:左
        } else {
            // 垂直滑动
            applyMove(dy > 0 ? 4 : 3); // 4:下, 3:上
        }
    }
}, { passive: true });

// 初始化加载
loadGameState();