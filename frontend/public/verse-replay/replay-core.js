(function attachReplayCore(globalScope) {
  'use strict';

  const REPLAY_ALPHABET =
    '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' +
    Array.from({ length: 64 }, (_, index) => String.fromCharCode(0xc0 + index)).join('') +
    '\xa4\xbe';

  const CHAR_TO_VALUE = new Map(
    Array.from(REPLAY_ALPHABET, (character, index) => [character, index]),
  );

  const DIRECTIONS = ['up', 'down', 'left', 'right'];
  const TIMING_BUCKETS = [
    [1, 255],
    [3, 1021],
    [12, 4084],
    [48, 16336],
    [192, 65344],
    [768, 265966],
    [3072, 1050094],
    [12288, 4186606],
  ];
  const UNKNOWN_TIMING_FALLBACK_MS = 100;

  // Each row records the first move that creates that power-of-two tile.
  const DEFAULT_MILESTONES = [
    { key: '32', label: '32', requirements: [32] },
    { key: '64', label: '64', requirements: [64] },
    { key: '128', label: '128', requirements: [128] },
    { key: '256', label: '256', requirements: [256] },
    { key: '512', label: '512', requirements: [512] },
    { key: '1024', label: '1024', requirements: [1024] },
    { key: '2048', label: '2048', requirements: [2048] },
    { key: '4096', label: '4096', requirements: [4096] },
    { key: '8192', label: '8192', requirements: [8192] },
    { key: '16384', label: '16384', requirements: [16384] },
    { key: '32768', label: '32768', requirements: [32768] },
    { key: '65536', label: '65536', requirements: [65536] },
  ];

  class ReplayFormatError extends Error {
    constructor(message) {
      super(message);
      this.name = 'ReplayFormatError';
    }
  }

  function decodeTiming(bucket, code) {
    if (bucket === 7 && code === 255) return null;
    const amount = TIMING_BUCKETS[bucket][0];
    const offset = bucket === 0
      ? 0
      : TIMING_BUCKETS[bucket - 1][1] + TIMING_BUCKETS[bucket - 1][0];
    return offset + amount * code;
  }

  function decodeRecord(chunk, recordIndex, width, height) {
    if (chunk.length !== 3) {
      throw new ReplayFormatError(`第 ${recordIndex + 1} 条记录不是 3 个字符。`);
    }

    const values = Array.from(chunk, (character) => CHAR_TO_VALUE.get(character));
    const badIndex = values.findIndex((value) => value === undefined);
    if (badIndex !== -1) {
      const code = chunk.charCodeAt(badIndex).toString(16).toUpperCase().padStart(2, '0');
      throw new ReplayFormatError(
        `第 ${recordIndex + 1} 条记录含有不支持的字符（0x${code}）。`,
      );
    }

    const encoded = (values[0] << 14) | (values[1] << 7) | values[2];
    const spawnCode = (encoded >> 2) & 0b11;
    const spawnX = (encoded >> 4) & 0b111;
    const spawnY = (encoded >> 7) & 0b111;
    const timingBucket = (encoded >> 10) & 0b111;
    const timingCode = (encoded >> 13) & 0xff;

    if (spawnCode > 1) {
      throw new ReplayFormatError(
        `第 ${recordIndex + 1} 条记录的出生方块码 ${spawnCode} 无效。`,
      );
    }
    if (spawnX >= width || spawnY >= height) {
      throw new ReplayFormatError(
        `第 ${recordIndex + 1} 条记录的出生坐标 (${spawnX}, ${spawnY}) 超出 ${width}×${height} 棋盘。`,
      );
    }

    return {
      direction: DIRECTIONS[encoded & 0b11],
      spawnExponent: spawnCode + 1,
      spawnValue: spawnCode === 1 ? 4 : 2,
      spawnX,
      spawnY,
      timingBucket,
      timingCode,
      deltaMs: decodeTiming(timingBucket, timingCode),
    };
  }

  function bytesToLatin1(source) {
    const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    let text = '';
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      text += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return text;
  }

  function collapseLine(line) {
    const compact = Array.from(line).filter(Boolean);
    const output = [];
    let addedScore = 0;

    for (let index = 0; index < compact.length; index += 1) {
      if (index + 1 < compact.length && compact[index] === compact[index + 1]) {
        const mergedExponent = compact[index] + 1;
        output.push(mergedExponent);
        addedScore += 2 ** mergedExponent;
        index += 1;
      } else {
        output.push(compact[index]);
      }
    }
    while (output.length < line.length) output.push(0);
    return { line: output, addedScore };
  }

  function arraysEqual(left, right) {
    if (left.length !== right.length) return false;
    for (let index = 0; index < left.length; index += 1) {
      if (left[index] !== right[index]) return false;
    }
    return true;
  }

  function moveBoard(board, width, height, direction) {
    const output = new Uint8Array(board);
    let addedScore = 0;

    if (direction === 'left' || direction === 'right') {
      for (let y = 0; y < height; y += 1) {
        const line = [];
        for (let x = 0; x < width; x += 1) line.push(board[y * width + x]);
        if (direction === 'right') line.reverse();
        const collapsed = collapseLine(line);
        if (direction === 'right') collapsed.line.reverse();
        addedScore += collapsed.addedScore;
        for (let x = 0; x < width; x += 1) output[y * width + x] = collapsed.line[x];
      }
    } else {
      for (let x = 0; x < width; x += 1) {
        const line = [];
        for (let y = 0; y < height; y += 1) line.push(board[y * width + x]);
        if (direction === 'down') line.reverse();
        const collapsed = collapseLine(line);
        if (direction === 'down') collapsed.line.reverse();
        addedScore += collapsed.addedScore;
        for (let y = 0; y < height; y += 1) output[y * width + x] = collapsed.line[y];
      }
    }

    return {
      board: output,
      moved: !arraysEqual(board, output),
      addedScore,
    };
  }

  function applySpecial32kRule(board, width, height, direction, moveNumber) {
    if (width !== 4 || height !== 4 || moveNumber < 27000) {
      return { board, addedScore: 0 };
    }

    const positions = [];
    for (let index = 0; index < board.length; index += 1) {
      if (board[index] === 15) positions.push(index);
    }
    if (positions.length !== 2) return { board, addedScore: 0 };

    const first = { x: positions[0] % width, y: Math.floor(positions[0] / width) };
    const second = { x: positions[1] % width, y: Math.floor(positions[1] / width) };
    const horizontal =
      first.y === second.y &&
      Math.abs(first.x - second.x) === 1 &&
      (direction === 'left' || direction === 'right');
    const vertical =
      first.x === second.x &&
      Math.abs(first.y - second.y) === 1 &&
      (direction === 'up' || direction === 'down');

    if (!horizontal && !vertical) return { board, addedScore: 0 };
    const adjusted = new Uint8Array(board);
    adjusted[positions[0]] = 14;
    adjusted[positions[1]] = 14;
    return { board: adjusted, addedScore: 32768 };
  }

  function planMoveTransitions(board, width, height, direction, moveNumber = 0) {
    const prepared = applySpecial32kRule(board, width, height, direction, moveNumber).board;
    const sources = [];
    const merges = [];
    const horizontal = direction === 'left' || direction === 'right';
    const lineCount = horizontal ? height : width;
    const lineLength = horizontal ? width : height;

    for (let lineNumber = 0; lineNumber < lineCount; lineNumber += 1) {
      const positions = [];
      for (let offset = 0; offset < lineLength; offset += 1) {
        const x = horizontal ? offset : lineNumber;
        const y = horizontal ? lineNumber : offset;
        positions.push(y * width + x);
      }
      if (direction === 'right' || direction === 'down') positions.reverse();

      const occupied = positions
        .filter((position) => prepared[position] !== 0)
        .map((position) => ({
          position,
          moveExponent: prepared[position],
          displayExponent: board[position] || prepared[position],
        }));

      let sourceIndex = 0;
      let destinationIndex = 0;
      while (sourceIndex < occupied.length) {
        const first = occupied[sourceIndex];
        const second = occupied[sourceIndex + 1];
        const toIndex = positions[destinationIndex];
        destinationIndex += 1;

        if (second && first.moveExponent === second.moveExponent) {
          sources.push({
            fromIndex: first.position,
            toIndex,
            exponent: first.displayExponent,
            moveExponent: first.moveExponent,
            merged: true,
          });
          sources.push({
            fromIndex: second.position,
            toIndex,
            exponent: second.displayExponent,
            moveExponent: second.moveExponent,
            merged: true,
          });
          merges.push({ toIndex, exponent: first.moveExponent + 1 });
          sourceIndex += 2;
        } else {
          sources.push({
            fromIndex: first.position,
            toIndex,
            exponent: first.displayExponent,
            moveExponent: first.moveExponent,
            merged: false,
          });
          sourceIndex += 1;
        }
      }
    }

    return { sources, merges };
  }

  function boardReachesRequirements(board, requirements) {
    const counts = new Map();
    for (const exponent of board) {
      if (!exponent) continue;
      const value = 2 ** exponent;
      counts.set(value, (counts.get(value) || 0) + 1);
    }

    for (const required of requirements) {
      for (const [value, count] of counts) {
        if (value > required && count > 0) return true;
      }
      const exactCount = counts.get(required) || 0;
      if (exactCount === 0) return false;
      counts.set(required, exactCount - 1);
    }
    return true;
  }

  function buildDecodedReplay(text) {
    const replayText = String(text).replace(/^\uFEFF/, '').trim();
    const header = replayText.match(/^(\d+)x(\d+)-([^_]*)_/);
    if (!header) {
      throw new ReplayFormatError('不支持的回放头；应类似 4x4-1_。');
    }

    const width = Number.parseInt(header[1], 10);
    const height = Number.parseInt(header[2], 10);
    const mode = header[3];
    if (width < 1 || width > 8 || height < 1 || height > 8) {
      throw new ReplayFormatError(`不支持 ${width}×${height} 棋盘；宽高必须在 1 到 8 之间。`);
    }

    const payload = replayText.slice(header[0].length);
    if (payload.length % 3 !== 0) {
      throw new ReplayFormatError(`回放数据长度 ${payload.length} 不能被 3 整除。`);
    }
    const recordCount = payload.length / 3;
    if (recordCount < 2) {
      throw new ReplayFormatError('回放缺少两个初始方块。');
    }

    const records = new Array(recordCount);
    for (let index = 0; index < recordCount; index += 1) {
      records[index] = decodeRecord(
        payload.slice(index * 3, index * 3 + 3),
        index,
        width,
        height,
      );
    }

    const moveCount = recordCount - 2;
    const cellCount = width * height;
    const snapshots = new Uint8Array((moveCount + 1) * cellCount);
    const scores = new Float64Array(moveCount + 1);
    const cumulativeMs = new Float64Array(moveCount + 1);
    const knownCumulativeMs = new Float64Array(moveCount + 1);
    const unknownCumulative = new Uint32Array(moveCount + 1);
    let board = new Uint8Array(cellCount);

    for (let index = 0; index < 2; index += 1) {
      const record = records[index];
      const position = record.spawnY * width + record.spawnX;
      if (board[position] !== 0) {
        throw new ReplayFormatError('两个初始方块占用了同一格。');
      }
      board[position] = record.spawnExponent;
    }
    snapshots.set(board, 0);

    const steps = new Array(moveCount);
    let score = 0;
    const milestones = DEFAULT_MILESTONES.map((milestone) => ({
      ...milestone,
      reachedStep: null,
      timeMs: null,
    }));

    for (let index = 0; index < moveCount; index += 1) {
      const record = records[index + 2];
      const moveNumber = index + 1;
      const special = applySpecial32kRule(board, width, height, record.direction, moveNumber);
      const moved = moveBoard(special.board, width, height, record.direction);
      if (!moved.moved) {
        throw new ReplayFormatError(`第 ${moveNumber} 步 ${record.direction} 没有改变棋盘。`);
      }

      board = moved.board;
      score += special.addedScore + moved.addedScore;
      const spawnPosition = record.spawnY * width + record.spawnX;
      if (board[spawnPosition] !== 0) {
        throw new ReplayFormatError(
          `第 ${moveNumber} 步的出生位置 (${record.spawnX}, ${record.spawnY}) 不为空。`,
        );
      }
      board[spawnPosition] = record.spawnExponent;

      const playbackDeltaMs = record.deltaMs === null
        ? UNKNOWN_TIMING_FALLBACK_MS
        : record.deltaMs;
      cumulativeMs[moveNumber] = cumulativeMs[index] + playbackDeltaMs;
      knownCumulativeMs[moveNumber] =
        knownCumulativeMs[index] + (record.deltaMs === null ? 0 : record.deltaMs);
      unknownCumulative[moveNumber] =
        unknownCumulative[index] + (record.deltaMs === null ? 1 : 0);
      scores[moveNumber] = score;
      snapshots.set(board, moveNumber * cellCount);

      steps[index] = {
        number: moveNumber,
        direction: record.direction,
        deltaMs: record.deltaMs,
        playbackDeltaMs,
        spawnValue: record.spawnValue,
        spawnX: record.spawnX,
        spawnY: record.spawnY,
      };

      for (const milestone of milestones) {
        if (
          milestone.reachedStep === null &&
          boardReachesRequirements(board, milestone.requirements)
        ) {
          milestone.reachedStep = moveNumber;
          milestone.timeMs = cumulativeMs[moveNumber];
        }
      }
    }

    const knownTimeMs = knownCumulativeMs[moveCount];
    const unknownTimings = unknownCumulative[moveCount];
    return {
      width,
      height,
      mode,
      moveCount,
      steps,
      snapshots,
      scores,
      cumulativeMs,
      knownCumulativeMs,
      unknownCumulative,
      knownTimeMs,
      unknownTimings,
      playbackTimeMs: cumulativeMs[moveCount],
      milestones,
      cellCount,
      getBoardAt(progress) {
        const bounded = Math.max(0, Math.min(moveCount, Number(progress) || 0));
        const start = bounded * cellCount;
        return snapshots.subarray(start, start + cellCount);
      },
    };
  }

  function snapshotToHex(board) {
    let encoded = 0n;
    for (let index = 0; index < board.length; index += 1) {
      const shift = BigInt((board.length - 1 - index) * 4);
      const exponent = Math.max(0, Math.min(15, Number(board[index]) || 0));
      encoded |= BigInt(exponent) << shift;
    }
    return encoded.toString(16).padStart(board.length, '0');
  }

  function clampNumber(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, Number(value) || 0));
  }

  function playbackDuration(replay, mode = 'original', constantMs = 100) {
    if (mode === 'constant') {
      return replay.moveCount * Math.max(0, Number(constantMs) || 0);
    }
    return replay.playbackTimeMs;
  }

  function progressAtPlaybackTimeline(replay, timeline, mode = 'original', constantMs = 100) {
    if (mode === 'constant') {
      const stepDuration = Math.max(0, Number(constantMs) || 0);
      if (stepDuration === 0) return replay.moveCount;
      const bounded = clampNumber(timeline, 0, playbackDuration(replay, mode, stepDuration));
      return Math.min(replay.moveCount, Math.floor(bounded / stepDuration));
    }

    const bounded = clampNumber(timeline, 0, replay.playbackTimeMs);
    let low = 0;
    let high = replay.moveCount;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (replay.cumulativeMs[middle] <= bounded) low = middle;
      else high = middle - 1;
    }
    return low;
  }

  function replayTimeAtPlaybackTimeline(replay, timeline, mode = 'original', constantMs = 100) {
    if (mode !== 'constant') {
      return clampNumber(timeline, 0, replay.playbackTimeMs);
    }

    const stepDuration = Math.max(0, Number(constantMs) || 0);
    if (stepDuration === 0) return replay.playbackTimeMs;
    const bounded = clampNumber(timeline, 0, playbackDuration(replay, mode, stepDuration));
    const fractionalProgress = bounded / stepDuration;
    const progress = Math.min(replay.moveCount, Math.floor(fractionalProgress));
    if (progress >= replay.moveCount) return replay.playbackTimeMs;
    const fraction = fractionalProgress - progress;
    const start = replay.cumulativeMs[progress];
    const end = replay.cumulativeMs[progress + 1];
    return start + (end - start) * fraction;
  }

  function playbackTimelineAtState(
    replay,
    progress,
    replayTimeMs,
    mode = 'original',
    constantMs = 100,
  ) {
    const boundedTime = clampNumber(replayTimeMs, 0, replay.playbackTimeMs);
    if (mode !== 'constant') return boundedTime;

    const stepDuration = Math.max(0, Number(constantMs) || 0);
    if (stepDuration === 0) return 0;
    const boundedProgress = Math.round(clampNumber(progress, 0, replay.moveCount));
    if (boundedProgress >= replay.moveCount) return playbackDuration(replay, mode, stepDuration);
    const start = replay.cumulativeMs[boundedProgress];
    const end = replay.cumulativeMs[boundedProgress + 1];
    const fraction = end > start
      ? clampNumber((boundedTime - start) / (end - start), 0, 1)
      : 0;
    return (boundedProgress + fraction) * stepDuration;
  }

  const api = {
    DEFAULT_MILESTONES,
    REPLAY_ALPHABET,
    ReplayFormatError,
    TIMING_BUCKETS,
    UNKNOWN_TIMING_FALLBACK_MS,
    decodeReplayBytes: (bytes) => buildDecodedReplay(bytesToLatin1(bytes)),
    decodeReplayText: buildDecodedReplay,
    decodeTiming,
    moveBoard,
    planMoveTransitions,
    playbackDuration,
    playbackTimelineAtState,
    progressAtPlaybackTimeline,
    replayTimeAtPlaybackTimeline,
    snapshotToHex,
  };

  globalScope.ReplayCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
