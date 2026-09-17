(function (root) {
  'use strict';
  const requested = new URLSearchParams(root.location.search).get('lang');
  const english = requested === 'en' || (requested !== 'zh' && !root.navigator.language.toLowerCase().startsWith('zh'));
  const messages = {
    '2048Verse 回放计时器': '2048Verse Replay Viewer',
    '带连续秒表、单步用时和节点用时的 2048Verse 回放查看器': 'Watch 2048 replays with recorded move timings, elapsed time, and tile milestones.',
    '尚未载入回放': 'No replay loaded', '载入与步速设置': 'Replay files and playback speed',
    '输入回放代码': 'Paste replay code', '上传回放文件': 'Open replay file', '下载回放': 'Download replay',
    '回放数据': 'Replay statistics', '分数': 'Score', '单步用时': 'Move time', '用时': 'Elapsed time', '进度': 'Progress',
    '输入回放代码，或上传一个': 'Paste a replay code or open a', '文件': 'file',
    '也可以把文件直接拖到此页面': 'You can also drag and drop a replay file here.',
    '2048 回放棋盘': '2048 replay board', '回放控制': 'Playback controls', '当前局面编码': 'Current board code',
    '回放进度': 'Replay progress', '后退十步': 'Back 10 moves', '后退一步': 'Previous move',
    '前进一步': 'Next move', '前进十步': 'Forward 10 moves',
    '后退十步（Shift + ←）': 'Back 10 moves (Shift + Left arrow)', '后退一步（← / Backspace）': 'Previous move (Left arrow / Backspace)',
    '前进一步（→ / Enter）': 'Next move (Right arrow / Enter)', '前进十步（Shift + →）': 'Forward 10 moves (Shift + Right arrow)',
    '播放': 'Play', '暂停': 'Pause', '播放（空格）': 'Play (Space)', '暂停（空格）': 'Pause (Space)',
    '节点用时': 'Tile milestones', '到达时的总用时': 'Elapsed time when first reached',
    '载入回放后显示': 'Load a replay to see milestones', '松开以载入回放': 'Drop to open replay',
    '关闭': 'Close', '完整粘贴以': 'Paste the full code starting with', '开头的内容': '',
    '载入回放': 'Load replay', '步速设置': 'Playback speed', '原始步速': 'Recorded timing',
    '按回放记录的每一步用时播放': 'Use the time recorded for each move.', '播放倍速': 'Playback rate',
    '恒定步速': 'Fixed move interval', '每一步使用相同等待时间': 'Wait the same amount of time between moves.',
    '每步时长': 'Time per move', '应用设置': 'Apply', '正在解析回放…': 'Reading replay…',
    '回放格式错误': 'Invalid replay format', '无法读取回放': 'Could not read replay',
    '请选择扩展名为 .vrs 的回放文件。': 'Choose a .vrs replay file.',
    '正在载入 AI 直播回放…': 'Loading AI livestream replay…', 'AI 直播对局': 'AI livestream game',
    '回放已过期或不存在': 'This replay has expired or could not be found.', '暂时无法获取回放': 'The replay is temporarily unavailable.',
    '网络请求失败': 'Network request failed.', '直播回放载入失败': 'Could not load livestream replay',
    '正在载入已验证对局…': 'Loading verified game replay…', '排行榜对局': 'Ranked game',
    '排行榜对局载入失败': 'Could not load ranked game replay', '粘贴的回放代码': 'Pasted replay',
    '秒表采用绝对时间基准连续计时，不累计页面渲染延迟。': 'The timer follows the recorded timeline, without accumulating rendering delays.',
    '13 字节 VRS 文件长度无效。': 'Invalid length for a 13-byte-record VRS file.',
    '13 字节 VRS 的初始分数和方向必须为 0。': 'A 13-byte-record VRS replay must start with a score and direction code of 0.',
    '回放文件不能超过 500 KB。': 'Replay files must be no larger than 500 KB.',
    '不支持的回放头；应类似 4x4-1_。': 'Unsupported replay header. Expected a header such as 4x4-1_.',
    '回放缺少两个初始方块。': 'The replay is missing its two starting tiles.',
    '两个初始方块占用了同一格。': 'The two starting tiles occupy the same cell.',
    'ULEB128 数值过大。': 'A replay time value exceeds the supported range.',
    '回放在 ULEB128 中途结束。': 'The replay ends partway through an encoded time value.',
    '2048next Base64 数据无效。': 'Invalid Base64 data in the 2048next replay.',
    '2048next 回放头无效。': 'Invalid 2048next replay header.',
    '2048next 回放 CRC32 校验失败。': 'The 2048next replay failed its integrity check (CRC32).',
    '排位回放仅支持 4×4 棋盘。': 'Ranked replays support only 4×4 boards.',
    '排位回放含有不支持的头标志。': 'The ranked replay contains unsupported header flags.',
    '回放初始棋块数量无效。': 'Invalid starting tile count.',
    '回放初始棋块数据不完整。': 'The starting tile data is incomplete.',
    '排位回放初始棋块位置重复。': 'The ranked replay has overlapping starting tiles.',
    'End 记录后仍有数据。': 'Unexpected data after the end marker.',
    '回放起始局面记录无效。': 'Invalid starting board record.', '扩展记录越界。': 'An extension record extends beyond the replay data.',
    '回放缺少起始局面或结束标记。': 'The replay is missing its starting board or end marker.',
  };
  const patterns = [
    [/^原始步速 · (.+)×$/, (_, n) => `Recorded timing · ${n}×`],
    [/^恒定步速 · (.+) ms$/, (_, n) => `Fixed interval · ${n} ms/move`],
    [/^未知 · (.+) ms$/, (_, n) => `Unknown · ${n} ms`],
    [/^(.+) 节点$/, (_, n) => `${n} tile milestone`],
    [/^(.+) · (\d+×\d+) · (.+) 步$/, (_, name, size, moves) => `${translate(name)} · ${size} · ${moves} moves`],
    [/^(.+) · ([\d,]+) 分$/, (_, name, score) => `${translate(name)} · ${score} points`],
    [/^含 (\d+) 个未知间隔；统一按 (\d+) ms 计入回放用时。秒表采用绝对时间基准，不累计页面渲染延迟。$/, (_, count, ms) => `${count} moves have no recorded timing; each is counted as ${ms} ms. The timer does not accumulate rendering delays.`],
    [/^(回放格式错误|无法读取回放|无法载入直播回放|无法载入排行榜对局)：(.+)$/, (_, label, detail) => `${({ '无法载入直播回放':'Could not load livestream replay', '无法载入排行榜对局':'Could not load ranked game replay' })[label] || translate(label)}: ${translate(detail)}`],
    [/^第 (\d+) 条记录不是 3 个字符。$/, (_, n) => `Record ${n} must contain exactly 3 characters.`],
    [/^第 (\d+) 条记录含有不支持的字符（(.+)）。$/, (_, n, code) => `Record ${n} contains an unsupported character (${code}).`],
    [/^第 (\d+) 条记录的出生方块码 (.+) 无效。$/, (_, n, code) => `Record ${n} has an invalid spawn tile code: ${code}.`],
    [/^第 (\d+) 条记录的出生坐标 (.+) 超出 (.+) 棋盘。$/, (_, n, pos, size) => `Record ${n} spawns a tile at ${pos}, outside the ${size} board.`],
    [/^第 (\d+) 条记录的分数低于上一条。$/, (_, n) => `The score in record ${n} is lower than in the previous record.`],
    [/^第 (\d+) 条记录的方向码 (.+) 无效。$/, (_, n, code) => `Record ${n} has an invalid direction code: ${code}.`],
    [/^第 (\d+) 步无法还原为合法移动和一次出数。$/, (_, n) => `Move ${n} cannot be reconstructed as a legal move followed by one tile spawn.`],
    [/^不支持 (.+) 棋盘；宽高必须在 1 到 8 之间。$/, (_, size) => `Unsupported board size ${size}. Width and height must each be between 1 and 8.`],
    [/^回放数据长度 (\d+) 不能被 3 整除。$/, (_, n) => `The replay data length (${n}) is not a multiple of 3.`],
    [/^第 (\d+) 步(?: (\w+) )?没有改变棋盘。$/, (_, n) => `Move ${n} does not change the board.`],
    [/^第 (\d+) 步的出生位置 (.+) 不为空。$/, (_, n, pos) => `Move ${n} spawns a tile in occupied cell ${pos}.`],
    [/^第 (\d+) 步出生位置不为空。$/, (_, n) => `Move ${n} spawns a tile in an occupied cell.`],
    [/^排位回放含有不支持的记录 (.+)。$/, (_, code) => `The ranked replay contains an unsupported record type: ${code}.`],
  ];
  function translate(text) {
    if (!english || typeof text !== 'string') return text;
    if (Object.hasOwn(messages, text)) return messages[text];
    for (const [pattern, replacement] of patterns) if (pattern.test(text)) return text.replace(pattern, replacement);
    return text;
  }
  function translatePage() {
    document.documentElement.lang = english ? 'en' : 'zh-CN';
    document.documentElement.translate = false;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.parentElement.closest('script,style')) continue;
      const text = node.textContent.trim();
      if (Object.hasOwn(messages, text)) node.textContent = node.textContent.replace(text, translate(text));
    }
    for (const el of document.querySelectorAll('[aria-label],[title]')) {
      for (const attr of ['aria-label','title']) if (el.hasAttribute(attr)) el.setAttribute(attr, translate(el.getAttribute(attr)));
    }
    document.title = translate(document.title);
    const description = document.querySelector('meta[name=description]');
    description.content = translate(description.content);
  }
  root.ReplayI18n = { translate, translatePage, english };
})(window);
