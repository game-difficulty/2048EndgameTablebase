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
  const NEXT_REPLAY_PREFIX = 'REPLAY_v1RPL_B64_';
  const NEXT_DIRECTIONS = ['up', 'right', 'down', 'left'];
  const STATE_REPLAY_RECORD_BYTES = 13;
  const MAX_REPLAY_FILE_BYTES = 2 * 1024 * 1024;
  const STATE_REPLAY_DIRECTIONS = new Map([
    [1, 'left'],
    [2, 'right'],
    [3, 'up'],
    [4, 'down'],
  ]);

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

  function arraysEqual(left, right) {
    if (left.length !== right.length) return false;
    for (let index = 0; index < left.length; index += 1) {
      if (left[index] !== right[index]) return false;
    }
    return true;
  }

  function planMoveTransitions(board, width, height, direction) {
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
        .filter((position) => board[position] !== 0)
        .map((position) => ({
          position,
          exponent: board[position],
        }));

      let sourceIndex = 0;
      let destinationIndex = 0;
      while (sourceIndex < occupied.length) {
        const first = occupied[sourceIndex];
        const second = occupied[sourceIndex + 1];
        const toIndex = positions[destinationIndex];
        destinationIndex += 1;

        if (second && first.exponent === second.exponent) {
          sources.push({
            fromIndex: first.position,
            toIndex,
            exponent: first.exponent,
            merged: true,
          });
          sources.push({
            fromIndex: second.position,
            toIndex,
            exponent: second.exponent,
            merged: true,
          });
          merges.push({ toIndex, exponent: first.exponent + 1 });
          sourceIndex += 2;
        } else {
          sources.push({
            fromIndex: first.position,
            toIndex,
            exponent: first.exponent,
            merged: false,
          });
          sourceIndex += 1;
        }
      }
    }

    return { sources, merges };
  }

  function moveBoard(board, width, height, direction) {
    const transition = planMoveTransitions(board, width, height, direction);
    const output = new Uint8Array(board.length);
    let addedScore = 0;

    for (const source of transition.sources) {
      if (!source.merged) output[source.toIndex] = source.exponent;
    }
    for (const merge of transition.merges) {
      output[merge.toIndex] = merge.exponent;
      addedScore += 2 ** merge.exponent;
    }

    return {
      board: output,
      moved: !arraysEqual(board, output),
      addedScore,
      transition: {
        direction,
        sources: transition.sources,
        merges: transition.merges,
      },
    };
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

  function readPackedBoard(view, offset) {
    const low = view.getUint32(offset, true);
    const high = view.getUint32(offset + 4, true);
    return BigInt(low) | (BigInt(high) << 32n);
  }

  function packedBoardToSnapshot(packed) {
    const board = new Uint8Array(16);
    for (let visualIndex = 0; visualIndex < board.length; visualIndex += 1) {
      const packedIndex = 15 - visualIndex;
      board[visualIndex] = Number((packed >> BigInt(packedIndex * 4)) & 0xfn);
    }
    return board;
  }

  function looksLikeStateReplay(bytes) {
    if (bytes.length < STATE_REPLAY_RECORD_BYTES) return false;
    // A state replay starts with score=0 and move=0: five zero bytes.
    for (let offset = 8; offset < STATE_REPLAY_RECORD_BYTES; offset += 1) {
      if (bytes[offset] !== 0) return false;
    }
    return true;
  }

  function projectPackedBoard(board) {
    const projected = new Uint8Array(board.length);
    for (let index = 0; index < board.length; index += 1) {
      projected[index] = Math.min(15, board[index]);
    }
    return projected;
  }

  function findProjectedSpawn(movedBoard, targetBoard) {
    const projected = projectPackedBoard(movedBoard);
    let spawn = null;
    for (let index = 0; index < movedBoard.length; index += 1) {
      if (projected[index] === targetBoard[index]) continue;
      if (
        spawn !== null ||
        movedBoard[index] !== 0 ||
        (targetBoard[index] !== 1 && targetBoard[index] !== 2)
      ) {
        return null;
      }
      spawn = { index, exponent: targetBoard[index] };
    }
    return spawn;
  }

  // Every format adapter ends here so snapshots, scoring and animation metadata
  // are rebuilt from the same full-width exponent board.
  function reconstructReplay({
    width,
    height,
    mode,
    format,
    initialBoard,
    records,
  }) {
    const cellCount = width * height;
    if (initialBoard.length !== cellCount) {
      throw new ReplayFormatError('回放初始棋盘尺寸无效。');
    }

    const moveCount = records.length;
    if (moveCount > 200000) throw new ReplayFormatError('回放不能超过 200000 步。');
    const snapshots = new Uint8Array((moveCount + 1) * cellCount);
    const scores = new Float64Array(moveCount + 1);
    const cumulativeMs = new Float64Array(moveCount + 1);
    const knownCumulativeMs = new Float64Array(moveCount + 1);
    const unknownCumulative = new Uint32Array(moveCount + 1);
    const steps = new Array(moveCount);
    const transitions = new Array(moveCount);
    const milestones = DEFAULT_MILESTONES.map((milestone) => ({
      ...milestone,
      reachedStep: null,
      timeMs: null,
    }));
    let board = new Uint8Array(initialBoard);
    let score = 0;
    snapshots.set(board, 0);

    records.forEach((record, index) => {
      const moveNumber = index + 1;
      const moved = moveBoard(board, width, height, record.direction);
      const constrained = record.targetPackedBoard instanceof Uint8Array;
      if (!moved.moved) {
        const message = constrained
          ? `第 ${moveNumber} 步无法还原为合法移动和一次出数。`
          : `第 ${moveNumber} 步 ${record.direction} 没有改变棋盘。`;
        throw new ReplayFormatError(message);
      }

      let spawnIndex = record.spawnIndex;
      let spawnExponent = record.spawnExponent;
      if (constrained) {
        const spawn = findProjectedSpawn(moved.board, record.targetPackedBoard);
        if (!spawn) {
          throw new ReplayFormatError(`第 ${moveNumber} 步无法还原为合法移动和一次出数。`);
        }
        spawnIndex = spawn.index;
        spawnExponent = spawn.exponent;
      }

      if (
        !Number.isInteger(spawnIndex) ||
        spawnIndex < 0 ||
        spawnIndex >= cellCount ||
        (spawnExponent !== 1 && spawnExponent !== 2) ||
        moved.board[spawnIndex] !== 0
      ) {
        throw new ReplayFormatError(`第 ${moveNumber} 步的出生位置或棋块无效。`);
      }

      const nextBoard = moved.board;
      nextBoard[spawnIndex] = spawnExponent;
      if (
        constrained &&
        !arraysEqual(projectPackedBoard(nextBoard), record.targetPackedBoard)
      ) {
        throw new ReplayFormatError(`第 ${moveNumber} 步无法还原为合法移动和一次出数。`);
      }

      const nextScore = score + moved.addedScore;
      if (record.scoreAfter !== undefined && record.scoreAfter !== nextScore) {
        throw new ReplayFormatError(
          `第 ${moveNumber} 步的分数增量与棋盘合并结果不一致。`,
        );
      }
      score = nextScore;
      board = nextBoard;

      const deltaMs = record.deltaMs ?? null;
      const playbackDeltaMs = deltaMs === null ? UNKNOWN_TIMING_FALLBACK_MS : deltaMs;
      cumulativeMs[moveNumber] = cumulativeMs[index] + playbackDeltaMs;
      knownCumulativeMs[moveNumber] = knownCumulativeMs[index] + (deltaMs === null ? 0 : deltaMs);
      unknownCumulative[moveNumber] = unknownCumulative[index] + (deltaMs === null ? 1 : 0);
      scores[moveNumber] = score;
      snapshots.set(board, moveNumber * cellCount);

      const spawn = {
        index: spawnIndex,
        exponent: spawnExponent,
        value: 2 ** spawnExponent,
      };
      transitions[index] = {
        ...moved.transition,
        spawn,
        scoreDelta: moved.addedScore,
      };
      steps[index] = {
        number: moveNumber,
        direction: record.direction,
        deltaMs,
        playbackDeltaMs,
        spawnValue: spawn.value,
        spawnX: spawnIndex % width,
        spawnY: Math.floor(spawnIndex / width),
        special32k: moved.transition.merges.some((merge) => merge.exponent === 16),
        source: record.source ?? null,
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
    });

    return {
      width,
      height,
      mode,
      format,
      moveCount,
      steps,
      transitions,
      snapshots,
      scores,
      cumulativeMs,
      knownCumulativeMs,
      unknownCumulative,
      knownTimeMs: knownCumulativeMs[moveCount],
      unknownTimings: unknownCumulative[moveCount],
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

  function decodeStateReplayBytes(source) {
    const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    if (!bytes.length || bytes.length % STATE_REPLAY_RECORD_BYTES !== 0) {
      throw new ReplayFormatError('13 字节 VRS 文件长度无效。');
    }

    const recordCount = bytes.length / STATE_REPLAY_RECORD_BYTES;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const records = new Array(recordCount);
    let previousScore = -1;
    for (let index = 0; index < recordCount; index += 1) {
      const offset = index * STATE_REPLAY_RECORD_BYTES;
      const score = view.getUint32(offset + 8, true);
      const moveCode = view.getUint8(offset + 12);
      if (index === 0 && (score !== 0 || moveCode !== 0)) {
        throw new ReplayFormatError('13 字节 VRS 的初始分数和方向必须为 0。');
      }
      if (score < previousScore) {
        throw new ReplayFormatError(`第 ${index + 1} 条记录的分数低于上一条。`);
      }
      records[index] = {
        board: packedBoardToSnapshot(readPackedBoard(view, offset)),
        score,
        moveCode,
      };
      previousScore = score;
    }

    const reconstructionRecords = records.slice(1).map((targetRecord, index) => {
      const direction = STATE_REPLAY_DIRECTIONS.get(targetRecord.moveCode);
      if (!direction) {
        throw new ReplayFormatError(
          `第 ${index + 2} 条记录的方向码 ${targetRecord.moveCode} 无效。`,
        );
      }
      return {
        direction,
        deltaMs: null,
        targetPackedBoard: targetRecord.board,
        scoreAfter: targetRecord.score,
      };
    });

    return reconstructReplay({
      width: 4,
      height: 4,
      mode: 'state-vrs',
      format: 'state-vrs13',
      initialBoard: records[0].board,
      records: reconstructionRecords,
    });
  }

  function decodeReplayBytes(source) {
    const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    if (bytes.length > MAX_REPLAY_FILE_BYTES) {
      throw new ReplayFormatError('回放文件不能超过 2 MB。');
    }
    if (bytes.length >= 4 && String.fromCharCode(...bytes.subarray(0, 4)) === 'RPL1') return buildDecodedRankedReplay(bytes);
    if (looksLikeStateReplay(bytes)) return decodeStateReplayBytes(bytes);
    return buildDecodedReplay(bytesToLatin1(bytes));
  }

  function buildDecodedReplay(text) {
    const replayText = String(text).replace(/^\uFEFF/, '').trim();
    if (replayText.length > MAX_REPLAY_FILE_BYTES) throw new ReplayFormatError('回放文件不能超过 2 MB。');
    if (replayText.startsWith(NEXT_REPLAY_PREFIX)) {
      return buildDecodedRankedReplay(replayText);
    }
    const header = replayText.match(/^(\d+)x(\d+)-([^_]*)_/);
    if (!header) {
      throw new ReplayFormatError('不支持的回放头；应类似 4x4-1_。');
    }

    // 2048Verse writes dimensions as rows x columns (height x width),
    // while the replay engine stores them as width x height.
    const height = Number.parseInt(header[1], 10);
    const width = Number.parseInt(header[2], 10);
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

    const cellCount = width * height;
    const initialBoard = new Uint8Array(cellCount);

    for (let index = 0; index < 2; index += 1) {
      const record = records[index];
      const position = record.spawnY * width + record.spawnX;
      if (initialBoard[position] !== 0) {
        throw new ReplayFormatError('两个初始方块占用了同一格。');
      }
      initialBoard[position] = record.spawnExponent;
    }

    return reconstructReplay({
      width,
      height,
      mode,
      format: 'legacy-text',
      initialBoard,
      records: records.slice(2).map((record) => ({
        direction: record.direction,
        spawnIndex: record.spawnY * width + record.spawnX,
        spawnExponent: record.spawnExponent,
        deltaMs: record.deltaMs,
      })),
    });
  }

  function decodeUleb128(bytes, state, limit) {
    let value = 0;
    let multiplier = 1;
    while (state.offset < limit) {
      const byte = bytes[state.offset];
      state.offset += 1;
      value += (byte & 0x7f) * multiplier;
      if (!(byte & 0x80)) return value;
      multiplier *= 128;
      if (multiplier > Number.MAX_SAFE_INTEGER) throw new ReplayFormatError('ULEB128 数值过大。');
    }
    throw new ReplayFormatError('回放在 ULEB128 中途结束。');
  }

  function crc32(bytes) {
    let value = 0xffffffff;
    for (const byte of bytes) {
      value ^= byte;
      for (let bit = 0; bit < 8; bit += 1) {
        value = (value & 1) ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
      }
    }
    return (value ^ 0xffffffff) >>> 0;
  }

  function rankedBytes(replayText) {
    const encoded = replayText.slice(NEXT_REPLAY_PREFIX.length).replace(/\s+/gu, '');
    let binary;
    try {
      binary = atob(encoded);
    } catch (_error) {
      throw new ReplayFormatError('2048next Base64 数据无效。');
    }
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  }

  function buildDecodedRankedReplay(replayText) {
    const bytes = replayText instanceof Uint8Array ? replayText : rankedBytes(replayText);
    if (bytes.length < 11 || String.fromCharCode(...bytes.subarray(0, 4)) !== 'RPL1') {
      throw new ReplayFormatError('2048next 回放头无效。');
    }
    const payloadEnd = bytes.length - 4;
    const expectedCrc = (
      bytes[payloadEnd]
      | (bytes[payloadEnd + 1] << 8)
      | (bytes[payloadEnd + 2] << 16)
      | (bytes[payloadEnd + 3] << 24)
    ) >>> 0;
    if (crc32(bytes.subarray(0, payloadEnd)) !== expectedCrc) {
      throw new ReplayFormatError('2048next 回放 CRC32 校验失败。');
    }
    const dimensions = bytes[4];
    const width = dimensions & 0x0f;
    const height = dimensions >>> 4;
    if (!['4x4', '3x4', '2x4', '3x3'].includes(`${height}x${width}`)) throw new ReplayFormatError('不支持的 RPL1 棋盘尺寸。');
    const flags = bytes[5];
    if (flags !== 0) throw new ReplayFormatError('排位回放含有不支持的头标志。');
    const initialCount = bytes[6];
    if (initialCount > width * height) throw new ReplayFormatError('回放初始棋块数量无效。');
    const state = { offset: 7 };
    let board = new Uint8Array(width * height);
    for (let index = 0; index < initialCount; index += 1) {
      if (state.offset >= payloadEnd) throw new ReplayFormatError('回放初始棋块数据不完整。');
      const packed = bytes[state.offset];
      state.offset += 1;
      const cell = packed & 0x0f;
      const exponent = ((packed >>> 4) & 1) + 1;
      if (cell >= board.length) throw new ReplayFormatError('回放初始棋块位置越界。');
      if (board[cell]) throw new ReplayFormatError('排位回放初始棋块位置重复。');
      board[cell] = exponent;
    }

    const rawMoves = [];
    let hasCheckpoint = false;
    let ended = false;
    while (state.offset < payloadEnd) {
      const type = bytes[state.offset];
      state.offset += 1;
      if (ended) throw new ReplayFormatError('End 记录后仍有数据。');
      if (type < 128) {
        if (rawMoves.length >= 200000) throw new ReplayFormatError('回放不能超过 200000 步。');
        const encodedDeltaMs = decodeUleb128(bytes, state, payloadEnd);
        rawMoves.push({
          direction: NEXT_DIRECTIONS[type & 3],
          spawnIndex: (type >>> 2) & 0x0f,
          spawnExponent: ((type >>> 6) & 1) + 1,
          deltaMs: encodedDeltaMs === 0xffffffff ? null : encodedDeltaMs,
        });
      } else if (type === 130) {
        if (width !== 4 || height !== 4 || rawMoves.length || hasCheckpoint || initialCount !== 0 || state.offset + 10 > payloadEnd) {
          throw new ReplayFormatError('回放起始局面记录无效。');
        }
        board = new Uint8Array(16);
        for (let index = 0; index < 16; index += 1) {
          for (let bit = 0; bit < 5; bit += 1) {
            const position = index * 5 + bit;
            board[index] |= ((bytes[state.offset + Math.floor(position / 8)] >>> (position % 8)) & 1) << bit;
          }
        }
        state.offset += 10;
        hasCheckpoint = true;
      } else if (type === 131) {
        decodeUleb128(bytes, state, payloadEnd);
        const length = decodeUleb128(bytes, state, payloadEnd);
        state.offset += length;
        if (state.offset > payloadEnd) throw new ReplayFormatError('扩展记录越界。');
      } else if (type === 132) {
        ended = true;
      } else {
        throw new ReplayFormatError(`排位回放含有不支持的记录 ${type}。`);
      }
    }
    if (!ended || (!initialCount && !hasCheckpoint)) throw new ReplayFormatError('回放缺少起始局面或结束标记。');

    return reconstructReplay({
      width,
      height,
      mode: 'ranked',
      format: 'rpl1',
      initialBoard: board,
      records: rawMoves,
    });
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
    decodeReplayBytes,
    decodeReplayText: buildDecodedReplay,
    decodeRankedReplayText: buildDecodedRankedReplay,
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
