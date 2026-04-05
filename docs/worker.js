// worker.js
import createAICore from './ai_core.js';

let ai_core = null;
let ai_player = null;
let pendingTask = null; // 用于暂存初始化完成前的计算请求

// 2. 初始化 WASM 模块
createAICore().then((instance) => {
    console.log("WASM 核心加载成功！");
    ai_core = instance;

    // 如果初始化完成前有任务进来，现在执行它
    if (pendingTask) {
        handleCalculate(pendingTask);
        pendingTask = null;
    }

    // 通知主线程准备就绪
    postMessage({ type: 'ready' });
});


// ==========================================
// 64位盘面位运算辅助与对称变换
// ==========================================

function ReverseLR(board) {
    board = BigInt(board);
    board = ((board & 0xff00ff00ff00ff00n) >> 8n) | ((board & 0x00ff00ff00ff00ffn) << 8n);
    board = ((board & 0xf0f0f0f0f0f0f0f0n) >> 4n) | ((board & 0x0f0f0f0f0f0f0f0fn) << 4n);
    return board;
}

function ReverseUD(board) {
    board = BigInt(board);
    board = ((board & 0xffffffff00000000n) >> 32n) | ((board & 0x00000000ffffffffn) << 32n);
    board = ((board & 0xffff0000ffff0000n) >> 16n) | ((board & 0x0000ffff0000ffffn) << 16n);
    return board;
}

function ReverseUL(board) {
    board = BigInt(board);
    board = (board & 0xff00ff0000ff00ffn) |
        ((board & 0x00ff00ff00000000n) >> 24n) |
        ((board & 0x00000000ff00ff00n) << 24n);
    board = (board & 0xf0f00f0ff0f00f0fn) |
        ((board & 0x0f0f00000f0f0000n) >> 12n) |
        ((board & 0x0000f0f00000f0f0n) << 12n);
    return board;
}

function ReverseUR(board) {
    board = BigInt(board);
    board = (board & 0x0f0ff0f00f0ff0f0n) |
        ((board & 0xf0f00000f0f00000n) >> 20n) |
        ((board & 0x00000f0f00000f0fn) << 20n);
    board = (board & 0x00ff00ffff00ff00n) |
        ((board & 0xff00ff0000000000n) >> 40n) |
        ((board & 0x0000000000ff00ffn) << 40n);
    return board;
}

function Rotate180(board) {
    board = BigInt(board);
    board = ((board & 0xffffffff00000000n) >> 32n) | ((board & 0x00000000ffffffffn) << 32n);
    board = ((board & 0xffff0000ffff0000n) >> 16n) | ((board & 0x0000ffff0000ffffn) << 16n);
    board = ((board & 0xff00ff00ff00ff00n) >> 8n) | ((board & 0x00ff00ff00ff00ffn) << 8n);
    board = ((board & 0xf0f0f0f0f0f0f0f0n) >> 4n) | ((board & 0x0f0f0f0f0f0f0f0fn) << 4n);
    return board;
}

function RotateL(board) {
    board = BigInt(board);
    board = ((board & 0xff00ff0000000000n) >> 32n) |
        ((board & 0x00ff00ff00000000n) << 8n) |
        ((board & 0x00000000ff00ff00n) >> 8n) |
        ((board & 0x0000000000ff00ffn) << 32n);
    board = ((board & 0xf0f00000f0f00000n) >> 16n) |
        ((board & 0x0f0f00000f0f0000n) << 4n) |
        ((board & 0x0000f0f00000f0f0n) >> 4n) |
        ((board & 0x00000f0f00000f0fn) << 16n);
    return board;
}

function RotateR(board) {
    board = BigInt(board);
    board = ((board & 0xff00ff0000000000n) >> 8n) |
        ((board & 0x00ff00ff00000000n) >> 32n) |
        ((board & 0x00000000ff00ff00n) << 32n) |
        ((board & 0x0000000000ff00ffn) << 8n);
    board = ((board & 0xf0f00000f0f00000n) >> 4n) |
        ((board & 0x0f0f00000f0f0000n) >> 16n) |
        ((board & 0x0000f0f00000f0f0n) << 16n) |
        ((board & 0x00000f0f00000f0fn) << 4n);
    return board;
}

function encode_board(boardArray) {
    let res = 0n;
    for (let i = 0; i < 16; i++) {
        res |= (BigInt(boardArray[i]) << BigInt((15 - i) * 4));
    }
    return res;
}

function move_board(b, moveCode) {
    if (ai_core && ai_core.move_board) {
        return ai_core.move_board(b, moveCode);
    }
    return b;
}

function printBoard(boardArray) {
    let output = "\n";
    for (let i = 0; i < 4; i++) {
        let row = "";
        for (let j = 0; j < 4; j++) {
            let val = boardArray[i * 4 + j];
            let real = val === 0 ? 0 : Math.pow(2, val);
            row += real.toString().padStart(6, ' ');
        }
        output += row + "\n";
    }
    return output;
}

function argmax(arr) {
    if (arr.length === 0) return -1;
    let maxVal = arr[0], maxIdx = 0;
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
    }
    return maxIdx;
}

function arraySum(arr) { return arr.reduce((a, b) => a + b, 0); }
function arrayMax(arr) { return Math.max(...arr); }

// ==========================================
// L3Manager 逻辑完全重现
// ==========================================
class L3Manager {
    constructor() {
        this.map_move = [
            [0, 1, 2, 3, 4],
            [0, 3, 4, 2, 1],
            [0, 2, 1, 4, 3],
            [0, 4, 3, 1, 2],
            [0, 2, 1, 3, 4],
            [0, 1, 2, 4, 3],
            [0, 3, 4, 1, 2],
            [0, 4, 3, 2, 1],
        ];
    }

    probe(board, counts, board_sum) {
        let threshold = 0;
        let c10_15 = counts.slice(10, 15);
        let c9_15 = counts.slice(9, 15);
        let c8_15 = counts.slice(8, 15);
        let c7_15 = counts.slice(7, 15);

        if (arraySum(counts.slice(10)) === 6 && arrayMax(c10_15) === 1) {
            if (!((board_sum % 1024) < 480 && counts[9] === 1)) {
                if (board_sum % 1024 > 96) threshold = 10;
                else if (board_sum % 1024 > 60) threshold = 8;
            }
        } else if (arraySum(counts.slice(9)) === 6 && arrayMax(c9_15) === 1) {
            if (!((board_sum % 512) < 240 && counts[8] === 1)) threshold = 9;
        } else if (arraySum(counts.slice(8)) === 6 && arrayMax(c8_15) === 1 && (board_sum % 256) < 240) {
            if (!((board_sum % 256) < 120 && counts[7] === 1) && !(arraySum(counts.slice(10)) === 5 && counts[9] === 0 && (board_sum % 256) < 64)) {
                threshold = 8;
            }
        } else if (arraySum(counts.slice(7)) === 6 && arrayMax(c7_15) === 1 && 20 < (board_sum % 128) && (board_sum % 128) < 120 && counts[7] === 1) {
            if (!((board_sum % 128) < 60 && counts[6] === 1)) threshold = 7;
        }

        if (threshold === 0) return [0, false, 0, [0.0, 0.0, 0.0, 0.0], threshold];

        let boardBigInt = BigInt(board) & 0xFFFFFFFFFFFFFFFFn;
        let masked_board = this.mask_large_tiles(boardBigInt, threshold);
        let table_types;

        if (threshold === 10 && (board_sum % 1024) < 128) table_types = [256, 512];
        else if (threshold > 8 && ((board_sum % 256) > 128 || (board_sum % 512) < 72)) table_types = [512];
        else if (threshold > 8) table_types = [512, 256];
        else if ((threshold === 8 && (board_sum % 256) > 60) || (threshold === 7 && (board_sum % 128) > 60)) table_types = [256, 512];
        else table_types = [256];

        let [best_move, win_rates, table_type] = this.probe_L3(masked_board, table_types, board_sum);

        if (best_move === 0 && arraySum(counts.slice(8)) === 7 && arrayMax(c8_15) === 1 && (board_sum % 256) < 240) {
            threshold = 9;
            [best_move, win_rates] = this.probe_441(boardBigInt, threshold - 1, board_sum);
            table_type = 512;
        }
        if (best_move === 0 && arraySum(counts.slice(7)) === 7 && arrayMax(c7_15) === 1 && (board_sum % 128) < 120) {
            threshold = 8;
            [best_move, win_rates] = this.probe_441(boardBigInt, threshold - 1, board_sum);
            table_type = 512;
        }

        if (threshold <= 8 && 0 < arrayMax(win_rates) && arrayMax(win_rates) < 0.625) {
            return [0, true, 0, win_rates, threshold];
        }

        return [best_move, false, table_type, win_rates, threshold];
    }

    get_syms(board) {
        return [board, RotateL(board), Rotate180(board), RotateR(board), ReverseLR(board), ReverseUD(board), ReverseUL(board), ReverseUR(board)];
    }

    probe_L3(masked_board, table_types, board_sum) {
        if (65280 < board_sum && board_sum < 65436) {
            masked_board = this.mask_large_tiles(masked_board, 8);
            table_types = [1256];
        }
        if (65000 < board_sum && board_sum < 65280) {
            masked_board = this.mask_large_tiles(masked_board, 9, 0x8);
            table_types = [512];
        }

        let syms = this.get_syms(masked_board);

        for (let i = 0; i < syms.length; i++) {
            let b = syms[i] & 0xFFFFFFFFFFFFFFFFn;
            if ((b & 0xfff0fffn) !== 0xfff0fffn) continue;

            for (let table_type of table_types) {
                if (table_type === 512) {
                    let [best_move, original_win_rates] = this.probe_44_128(b, i, board_sum);
                    if (best_move !== 0) return [best_move, original_win_rates, table_type];
                }

                let win_rates = ai_core.find_best_egtb_move(b, table_type);

                if (arrayMax(win_rates) > 0) {
                    let [best_move, original_win_rates] = this.handle_result(win_rates, i);
                    return [best_move, original_win_rates, table_type];
                }
            }
        }
        return [0, [0.0, 0.0, 0.0, 0.0], 0];
    }

    probe_441(board, threshold, board_sum) {
        let masked = this.mask_large_tiles(board, threshold, threshold);
        let syms = this.get_syms(masked);
        let threshBig = BigInt(threshold);

        for (let i = 0; i < syms.length; i++) {
            let b = syms[i] & 0xFFFFFFFFFFFFFFFFn;
            if ((b & 0xfff0fffn) === (0x1110111n * threshBig)) {
                b |= 0xfff0fffn;
            } else {
                continue;
            }

            let [best_move, original_win_rates] = this.probe_44_128(b, i, board_sum);
            if (best_move !== 0) return [best_move, original_win_rates];

            let win_rates = ai_core.find_best_egtb_move(b, 512);

            if (arrayMax(win_rates) > 0) {
                return this.handle_result(win_rates, i);
            }
        }
        return [0, [0.0, 0.0, 0.0, 0.0]];
    }

    probe_44_128(masked_board, i, board_sum) {
        if (390 < (board_sum % 512) && (board_sum % 512) < 480 && board_sum > 63000) {
            let remask_44 = this.mask_large_tiles(masked_board, 7);
            if ((remask_44 & 0xffffffffn) !== 0xffffffffn) return [0, [0.0, 0.0, 0.0, 0.0]];

            remask_44 = (ReverseLR(remask_44) & 0xFFFFFFFF00000000n) + 0x7fff8fffn;
            let win_rates = ai_core.find_best_egtb_move(remask_44, 512);

            if (arrayMax(win_rates) > 0) {
                let temp = win_rates[0]; win_rates[0] = win_rates[1]; win_rates[1] = temp;
                return this.handle_result(win_rates, i);
            }
        }
        return [0, [0.0, 0.0, 0.0, 0.0]];
    }

    handle_result(win_rates, i) {
        let found_dir = argmax(win_rates) + 1;
        let best_move = this.map_move[i][found_dir];
        let original_win_rates = [0.0, 0.0, 0.0, 0.0];
        for (let d = 1; d < 5; d++) {
            let orig_dir = this.map_move[i][d];
            original_win_rates[orig_dir - 1] = win_rates[d - 1];
        }
        return [best_move, original_win_rates];
    }

    mask_large_tiles(board, threshold, mask = 0xF) {
        let res = 0n;
        let threshBig = BigInt(threshold);
        let maskBig = BigInt(mask);
        for (let i = 0n; i < 16n; i++) {
            let shift = 4n * i;
            let val = (board >> shift) & 0xFn;
            if (val >= threshBig) val = maskBig;
            res |= (val << shift);
        }
        return res;
    }

    probe_after_move(board, threshold, table_types, board_sum) {
        let masked_board = this.mask_large_tiles(board, threshold);
        let win_rate = 0.0;
        let empty_slots = 0;
        for (let i = 0n; i < 16n; i++) {
            if (((masked_board >> (4n * i)) & 0xFn) === 0n) {
                empty_slots++;
                let t1 = masked_board | (1n << (4n * i));
                let t2 = masked_board | (2n << (4n * i));

                let r1 = this.probe_L3(t1, table_types, board_sum + 2);
                win_rate += arrayMax(r1[1]) * 0.9;

                let r2 = this.probe_L3(t2, table_types, board_sum + 4);
                win_rate += arrayMax(r2[1]) * 0.1;
            }
        }
        return empty_slots === 0 ? 0 : win_rate / empty_slots;
    }
}

// ==========================================
// CoreAILogic 逻辑完全重现
// ==========================================
class CoreAILogic {
    constructor() {
        this.SCORE_CRITICAL = -5000;
        this.SCORE_HOPELESS = -30000;
        this.FALLBACK_DEPTH = 5;
        this.manager = new L3Manager();
        this.last_depth = 4;
        this.last_sum = 0;
        this.last_prune = 0;
        this.last_move = '';
        this.time_ratio = 4.0;
        this.time_limit_ratio = 1.0;
    }

    is_mess(boardArray) {
        // 使用的是真实数字（2,4,8...），而不是对数 0~15
        let rawBoard = boardArray.map(v => v === 0 ? 0 : Math.pow(2, v));
        let board_sum = arraySum(rawBoard);
        if (board_sum % 512 < 12) return false;

        let large_tiles = rawBoard.filter(v => v > 128).length;
        if (large_tiles < 3) return false;

        let indexed = rawBoard.map((val, idx) => ({ val, idx }));
        indexed.sort((a, b) => b.val - a.val);

        if (large_tiles === 6) {
            let top6 = indexed.slice(0, 6);
            let uniqueVals = new Set(top6.map(o => o.val));
            if (uniqueVals.size < 6) return false;
            let top6_pos = top6.map(o => o.idx).sort((a, b) => a - b).join(',');
            const allowed = new Set([
                "0,1,2,3,4,5", "0,1,2,3,6,7", "0,1,4,5,8,12", "0,4,8,9,12,13",
                "2,3,6,7,11,15", "3,7,10,11,14,15", "8,9,12,13,14,15", "10,11,12,13,14,15",
                "0,1,2,4,5,6", "1,2,3,5,6,7", "0,1,4,5,8,9", "4,5,8,9,12,13",
                "2,3,6,7,10,11", "6,7,10,11,14,15", "8,9,10,12,13,14", "9,10,11,13,14,15",
                "0,1,3,4,5,7", "0,2,3,4,6,7", "0,1,4,5,12,13", "0,1,8,9,12,13",
                "2,3,6,7,14,15", "2,3,10,11,14,15", "8,9,11,12,13,15", "8,10,11,12,14,15"
            ]);
            return !allowed.has(top6_pos);
        } else if (large_tiles === 4) {
            let top4 = indexed.slice(0, 4);
            if (new Set(top4.map(o => o.val)).size < 4) return false;
            let top4_pos = top4.map(o => o.idx).sort((a, b) => a - b).join(',');
            const allowed = new Set([
                "0,1,2,3", "0,4,8,12", "12,13,14,15", "3,7,11,15",
                "0,1,2,4", "4,8,12,13", "11,13,14,15", "2,3,7,11",
                "0,1,4,8", "8,12,13,14", "7,11,14,15", "1,2,3,7",
                "0,1,4,5", "8,9,12,13", "10,11,14,15", "2,3,6,7",
                "0,1,3,4", "0,1,4,12", "0,2,3,7", "2,3,7,15",
                "0,8,12,13", "8,12,13,15", "3,11,14,15", "11,12,14,15"
            ]);
            return !allowed.has(top4_pos);
        } else if (large_tiles === 3) {
            let top3 = indexed.slice(0, 3);
            if (new Set(top3.map(o => o.val)).size < 3) return false;
            let top3_pos = top3.map(o => o.idx).sort((a, b) => a - b).join(',');
            const allowed = new Set([
                "0,1,2", "1,2,3", "3,7,11", "7,11,15", "13,14,15", "12,13,14", "4,8,12", "0,4,8",
                "0,1,3", "0,2,3", "3,7,15", "3,11,15", "12,14,15", "12,13,15", "0,8,12", "0,4,12",
                "0,1,4", "2,3,7", "11,14,15", "8,12,13"
            ]);
            return !allowed.has(top3_pos);
        } else {
            let top_n_pos = new Set(indexed.slice(0, large_tiles).map(o => o.idx));
            const corners_l_shapes = [
                [0, 1, 4], [3, 2, 7], [12, 8, 13], [15, 11, 14]
            ];
            for (let corner of corners_l_shapes) {
                if (corner.every(val => top_n_pos.has(val))) return false;
            }
            return true;
        }
    }

    calculate_step(ai_player, boardArray, counts) {
        let empty_slots = counts[0];
        let board_sum = arraySum(boardArray.map(v => v === 0 ? 0 : Math.pow(2, v)));
        let big_nums = arraySum(counts.slice(8));

        let [move, is_evil, table_type, win_rates, threshold] = this.manager.probe(ai_player.board, counts, board_sum);

        if (move) {
            console.log(`[EGTB] Candidate Move: ${move}, Rates: [${win_rates.map(r => r.toFixed(4))}], Table: ${table_type}`);
            if (this.validate_egtb_move(boardArray, ai_player, move, table_type, win_rates, board_sum, threshold)) {
                console.log(`%c[EGTB Hit] Decision: ${move}`, "color: #4CAF50; font-weight: bold;");
                this.last_move = 'L3';
                return move;
            } else {
                console.log("[EGTB] Validation failed, falling back to search.");
            }
        }

        // 1. 基础判断：判定是否处于“不合并”状态
        const is_not_merging = (arrayMax(counts.slice(8, 15)) === 1) &&
            !(counts[7] > 1 && counts[8] === 1) &&
            !(counts[6] > 1 && counts[7] === 1 && counts[8] === 1 && board_sum % 1024 < 96);

        const is_mess = is_not_merging ? this.is_mess(boardArray) : false;

        // 2. 5-tiler 特殊残局判定（严格对应 Python 括号优先级）
        const is_5tiler = ((65520 > board_sum && board_sum > 62000) || (arraySum(counts.slice(11)) === 4 && counts[10] === 0)) &&
            ((board_sum % 1024 < 24) || (board_sum % 1024 > 996));

        // 3. AI 控制位初始化
        ai_player.do_check = (is_mess && [3, 4, 5, 6].includes(big_nums)) ? big_nums : 0;

        // 4. 计算 Prune
        let prune = 0;
        if (is_not_merging) {
            const cond1 = (!(40 < board_sum % 512 && board_sum % 512 < 500) && arrayMax(counts.slice(7, 9)) > 1 && big_nums > 2);
            const cond2 = (!(32 < board_sum % 256 && board_sum % 256 < 250) && arrayMax(counts.slice(6, 8)) > 1 && big_nums > 4);
            const cond3 = (!(24 < board_sum % 128 && board_sum % 128 < 126) && arrayMax(counts.slice(5, 7)) > 1 && big_nums > 4);

            if (!(cond1 || cond2 || cond3 || is_mess)) {
                prune = 1;
            }
        }
        ai_player.prune = prune;

        // 5. 高层残局
        const is_sparse_endgame = (arrayMax(counts.slice(6)) === 1 && arraySum(counts.slice(6)) >= 9);

        if ((is_mess || is_evil || this.tiles_all_set(counts) || is_sparse_endgame) && !is_5tiler) {
            ai_player.prune = 0;
        }

        // 6. 特殊补丁
        if (this.danbianhuichuan_patch(boardArray, board_sum)) {
            ai_player.prune = 1;
        }

        console.log(`[Status] Sum: ${board_sum}, BigNums: ${big_nums}, Mess: ${is_mess}, Prune: ${ai_player.prune}`);

        let initial_depth, max_depth, time_limit;
        let best_op, final_depth, scores;

        if (is_mess || is_5tiler) {
            let big_nums2 = arraySum(counts.slice(9));
            initial_depth = 5;
            max_depth = 24;
            time_limit = 1.2 * Math.pow(big_nums2, 0.25);
            [best_op, final_depth, scores] = this.perform_iterative_search(ai_player, initial_depth, max_depth, time_limit);
        } else if (empty_slots > 4 && big_nums < 2 && is_not_merging) {
            initial_depth = 3;
            [best_op, final_depth, scores] = this.perform_iterative_search(ai_player, initial_depth, initial_depth, 0.1);
        } else if (((big_nums <= 3 && 32 < board_sum % 256 && board_sum % 256 < 248 && is_not_merging) || big_nums < 3) && !(board_sum % 256 < 72 && counts[6] > 0)) {
            initial_depth = counts[7] === 0 ? 4 : 5;
            [best_op, final_depth, scores] = this.perform_iterative_search(ai_player, initial_depth, initial_depth, 0.1);
        } else {
            if (65380 < board_sum && board_sum <= 65500) {
                initial_depth = Math.min(33, Math.floor((65540 - board_sum) / 2)); max_depth = 60; time_limit = 0.8;
            } else if (65260 < board_sum && board_sum <= 65380) {
                initial_depth = 20; max_depth = 60; time_limit = 1.0;
            } else if (counts[7] > 1 || (board_sum % 512 < 20 && arraySum(counts.slice(8)) > 4)) {
                initial_depth = 4; max_depth = 32; time_limit = 0.32 * Math.pow(big_nums, 0.4);
            } else if (is_not_merging && arraySum(counts.slice(7)) > 5) {
                initial_depth = 4; max_depth = 48; time_limit = 0.32 * Math.pow(big_nums, 0.25);
            } else {
                initial_depth = 4; max_depth = 24; time_limit = 0.16 * Math.pow(big_nums, 0.25);
            }

            initial_depth += ai_player.prune;
            if (!is_mess && arraySum(counts.slice(9)) <= 3) max_depth = 10;

            console.log(`[Debug] initial_depth pre-pers: ${initial_depth}, last_sum: ${this.last_sum}, last_depth: ${this.last_depth}`);

            if (ai_player.prune && Math.abs(board_sum - this.last_sum) < 6) {
                let min_initial = Math.min(this.last_depth - 1, Math.round(this.last_depth * 0.9));
                initial_depth = Math.max(initial_depth, min_initial);
                console.log(`[Debug] Scaled depth to ${initial_depth} based on last_depth ${this.last_depth}`);
            }

            [best_op, final_depth, scores] = this.perform_iterative_search(ai_player, initial_depth, max_depth, time_limit);
        }

        this.last_sum = board_sum;
        this.last_depth = final_depth;
        this.last_prune = ai_player.prune;
        this.last_move = 'search';

        return best_op;
    }

    perform_iterative_search(ai_player, initial_depth, max_depth, time_limit) {
        let best_op_so_far = -1;
        let final_depth = 0;
        let valid_scores = [];

        let start_time = performance.now() / 1000.0;
        let local_limit = time_limit * this.time_limit_ratio;
        let last_depth_time = null;
        let depth = initial_depth;
        let fallback_attempts = 0;

        while ((depth <= max_depth && best_op_so_far !== 0) && !(fallback_attempts > 0 && valid_scores.length > 0)) {
            let elapsed = (performance.now() / 1000.0) - start_time;
            let remaining_time = local_limit - elapsed;

            if (fallback_attempts) {
                remaining_time = Math.max(remaining_time, local_limit / 2 + 0.005);
            }

            if (this._should_stop_search(best_op_so_far, last_depth_time, remaining_time)) break;

            let timeout_ms = Math.max(5, Math.floor(remaining_time * 1000) - 1);
            if (best_op_so_far === -1) {
                timeout_ms = Math.max(timeout_ms, Math.floor((Math.max(remaining_time * 2.5, time_limit * 0.2)) * 1000));
            }

            let depth_start = performance.now() / 1000.0;
            ai_player.stop_search = false;
            ai_player.start_search(depth, timeout_ms);
            let depth_elapsed = (performance.now() / 1000.0) - depth_start;

            if (ai_player.stop_search) {
                let res = this._handle_timeout(best_op_so_far, depth, fallback_attempts);
                depth = res[0]; fallback_attempts = res[1];
                if (depth === null) break;
                continue;
            }

            best_op_so_far = ai_player.best_operation;
            final_depth = depth;

            valid_scores = ai_player.scores;

            let elapsed_total = (performance.now() / 1000.0) - start_time;
            console.log(`[Depth ${depth}] Best: ${best_op_so_far}, Scores: [${valid_scores.map(s => Math.round(s))}], Time: ${depth_elapsed.toFixed(3)}s (Total: ${elapsed_total.toFixed(3)}s)`);

            let strat = this._update_search_strategy(ai_player, best_op_so_far, valid_scores, depth, initial_depth, local_limit, time_limit, max_depth);
            local_limit = strat[0]; max_depth = strat[1];

            this._update_time_ratio(last_depth_time, depth_elapsed);
            last_depth_time = depth_elapsed;
            depth++;
        }

        if (best_op_so_far === -1 || valid_scores.length === 0) {
            let fallback = this._force_fallback_search(ai_player);
            best_op_so_far = fallback[0]; final_depth = fallback[1];
        }

        ai_player.stop_search = false;
        return [best_op_so_far, final_depth, valid_scores];
    }

    _should_stop_search(best_op, last_time, remaining_time) {
        if (best_op === -1) return false;
        if (remaining_time <= 0.0001) return true;
        if (last_time !== null && (last_time * this.time_ratio) > remaining_time * 0.9) return true;
        return false;
    }


    _handle_timeout(best_op, depth, attempts) {
        this.time_ratio *= 1.03;
        if (best_op === -1 && attempts < 3) {
            let new_depth = Math.max(1, Math.floor(Math.min(Math.max(depth - 2, 1), depth * 0.8)));
            return [new_depth, attempts + 1];
        }
        return [null, attempts];
    }

    _update_search_strategy(ai_player, best_op, scores, depth, init_depth, local_limit, time_limit, max_depth) {
        if (scores.length === 0) return [local_limit, max_depth];
        let best_score = (best_op > 0 && best_op <= scores.length) ? scores[best_op - 1] : 0;

        if (best_score < this.SCORE_CRITICAL) {
            local_limit = time_limit * this.time_limit_ratio + 1 + 0.1 * (depth - init_depth);
        }
        if (arrayMax(scores) < this.SCORE_HOPELESS) {
            max_depth = Math.min(max_depth, 16);
        }
        return [local_limit, max_depth];
    }

    _update_time_ratio(last_time, current_time) {
        if (last_time !== null && last_time > 0.001) {
            let current_ratio = Math.max(1.2, Math.min(12.0, current_time / last_time));
            this.time_ratio = Math.pow(2, 0.75 * Math.log2(this.time_ratio) + 0.25 * Math.log2(current_ratio));
        }
    }

    _force_fallback_search(ai_player) {
        let fallback_start = performance.now() / 1000.0;
        ai_player.stop_search = false;
        ai_player.prune = 0;
        ai_player.clear_cache();
        ai_player.start_search(this.FALLBACK_DEPTH, 2000); // 2s timeout for fallback
        return [ai_player.best_operation, this.FALLBACK_DEPTH, (performance.now() / 1000.0) - fallback_start];
    }

    validate_egtb_move(boardArray, ai_player, move, table_type, win_rates, board_sum, threshold) {
        if (table_type === 512 && board_sum % 512 > 506 && 0.91109 < arrayMax(win_rates) && arrayMax(win_rates) < 0.91111) return true;

        ai_player.stop_search = false;
        ai_player.prune = (48 < board_sum % 256 && board_sum % 256 < 234) ? 1 : 0;
        ai_player.do_check = 0;
        let depth = ai_player.prune ? 8 : 6;
        ai_player.start_search(depth, 1000); // 1s timeout for validation

        let need_further_check = false;

        let scores = ai_player.scores;

        if (table_type === 1256 && (arrayMax(scores) - scores[argmax(win_rates)] < 50)) return true;

        if ((win_rates[argmax(scores)] === 0.0) && (arrayMax(scores) - scores[argmax(win_rates)] > 5)) {
            let win_rate = arrayMax(win_rates);
            if ((table_type === 256 && win_rate > 0.993) || (table_type === 512 && board_sum % 512 < 64 && win_rate > 0.84)) {
                need_further_check = false;
            } else {
                need_further_check = true;
            }
        }

        if (!need_further_check) {
            let target_score = scores[move - 1];
            let sorted_scores = [...scores].sort((a, b) => b - a);
            if ((target_score >= sorted_scores[0] - 16 && sorted_scores[2] > 2400) ||
                (target_score >= sorted_scores[0] - 24 && sorted_scores[2] > 2800) ||
                (target_score >= sorted_scores[0] - 8)) {
                return true;
            }
        }

        let max_d = table_type === 1256 ? 48 : 10;
        let min_d = table_type === 1256 ? 24 : 6;
        let _time_limit = table_type === 1256 ? 0.64 : 0.32;

        let [best_op, final_depth, scores_deep] = this.perform_iterative_search(ai_player, min_d, max_d, _time_limit);
        if (scores_deep.length === 0) return false;
        scores = scores_deep;

        if (table_type === 1256 && scores.length > 0 && (arrayMax(scores) - scores[argmax(win_rates)] < 100)) return true;

        let target_score = scores[move - 1];
        let sorted_scores = [...scores].sort((a, b) => b - a);

        if (target_score < -ai_player.dead_score / 2 && sorted_scores[0] > 0) return false;

        if ((target_score >= sorted_scores[0] - 24 && sorted_scores[2] > 2400) ||
            (target_score >= sorted_scores[0] - 36 && sorted_scores[2] > 2800) ||
            (target_score >= sorted_scores[0] - 12)) {
            return true;
        }

        if (!(threshold === 8 && table_type === 512 && board_sum % 256 < 96)) {
            let board_encoded = encode_board(boardArray) & 0xFFFFFFFFFFFFFFFFn;
            let after_board1 = move_board(board_encoded, best_op);
            let after_board2 = move_board(board_encoded, move);
            let win_rate1 = this.manager.probe_after_move(after_board1, threshold, [table_type], board_sum);
            let win_rate2 = this.manager.probe_after_move(after_board2, threshold, [table_type], board_sum);

            if (Math.max(win_rate2, win_rate1) < 0.2) return false;

            if (win_rate2 > win_rate1 && ((win_rate1 > 0 || target_score > 2000) ||
                (win_rate1 === 0 && target_score < -3000 && 60 < board_sum % 256 && board_sum % 256 < 200))) {
                return true;
            }
        }
        return false;
    }

    tiles_all_set(counts) {
        let last_dup = 0;
        let i = 0;
        for (i = 3; i < 15; i++) if (counts[i] > 1) last_dup = i;
        if (last_dup === 0) return false;
        for (i = last_dup + 1; i < 15; i++) if (counts[i] === 0) break;
        let final_big_tiles = arraySum(counts.slice(i)) + 1;
        return final_big_tiles < 5 && i > 9 && !(final_big_tiles + i < 14 && last_dup < 6);
    }

    danbianhuichuan_patch(boardArray, board_sum) {
        if (!(board_sum % 1024 >= 1016 && board_sum > 63000)) return 0;
        let board = encode_board(boardArray) & 0xFFFFFFFFFFFFFFFFn;
        board = this.manager.mask_large_tiles(board, 9);

        let arr = [board, ReverseLR(board), ReverseUD(board), ReverseUL(board), ReverseUR(board), Rotate180(board), RotateL(board), RotateR(board)];
        board = arr.reduce((max, current) => current > max ? current : max, arr[0]);

        if ((board & 0xfff0fff000000000n) === 0xfff0fff000000000n && (board & 0xf000ff0000000n) === 0x8000760000000n) return 1;
        return 0;
    }
}

// ==========================================
// Web Worker 消息监听逻辑
// ==========================================
let current_time_limit_ratio = 1.0;

self.onmessage = function (e) {
    if (e.data.type === 'calculate') {
        if (!ai_core) {
            console.log("WASM 尚未就绪，暂存任务...");
            pendingTask = e.data;
            return;
        }
        handleCalculate(e.data);
    } else if (e.data.type === 'update_speed') {
        current_time_limit_ratio = e.data.ratio;
        console.log(`[Config] Speed updated. time_limit_ratio: ${current_time_limit_ratio.toFixed(3)}`);
    }
};

// 将计算逻辑封装，确保初始化前后调用逻辑一致
function handleCalculate(data) {
    const hexStr = data.board_encoded;
    const boardBigInt = BigInt('0x' + hexStr);

    let boardArray = new Array(16).fill(0);
    let counts = new Array(16).fill(0);

    for (let i = 0; i < 16; i++) {
        let val = parseInt(hexStr[i], 16);
        boardArray[i] = val;
        counts[val]++;
    }

    if (!ai_player) {
        ai_player = new ai_core.AIPlayer(boardBigInt);
    } else {
        // 重置内部搜索树状态，防止之前盘面的死缓存让 AI 误判！
        if (typeof ai_player.reset_board === 'function') {
            ai_player.reset_board(boardBigInt);
        } else {
            ai_player.board = boardBigInt;
        }
    }

    runAI(boardArray, counts, hexStr);
}

let core_ai_logic = null;

function runAI(boardArray, counts, hexStr) {
    if (!core_ai_logic) {
        core_ai_logic = new CoreAILogic();
    }
    core_ai_logic.time_limit_ratio = current_time_limit_ratio; // 应用实时比率

    console.log(`%c--- AI Thinking (Board: ${hexStr}) ---`, "background: #222; color: #bada55; font-size: 14px;");
    console.log(printBoard(boardArray));

    const bestMove = core_ai_logic.calculate_step(ai_player, boardArray, counts);
    console.log(`%c>>> Decision: ${bestMove} <<<`, "background: #222; color: #ffeb3b; font-weight: bold; font-size: 16px;");

    postMessage({ type: 'move_result', best_move: bestMove });
}