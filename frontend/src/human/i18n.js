import { ref } from 'vue';
import { createLocalStorageStore } from '../services/storage/localStorageStore.js';

const preferences = createLocalStorageStore({ key: 'user-preferences', version: 1, defaultValue: {} });
export const language = ref('zh');
const en = {
  '回放仅保存在本地':'Replay saved locally only',
  '回放上传失败，请保存回放':'Replay upload failed — save your replay',
  '本局回放尚未确认上传成功。请立即下载或复制回放并妥善保存，后续可交由站长审核并手动录入。请勿清除浏览器数据。':'This replay has not been confirmed as uploaded. Download or copy it now and keep it safe. You can later send it to the site owner for review and manual entry. Do not clear your browser data.',
  '对局编号':'Game ID',
  '无法读取回放，请保留浏览器数据并联系站长。':'Could not read the replay. Keep your browser data and contact the site owner.',
  '截至第':'Through move', '复制回放':'Copy replay', '下载回放文件':'Download replay file', '在回放站查看 ↗':'Open replay viewer ↗',
  '回放代码':'Replay code', '回放已复制':'Replay copied', '无法读取当前回放，请重新进入本局后重试。':'Could not read this replay. Reopen the game and retry.',
  '无法自动复制，请选中下方代码手动复制。':'Copy was unavailable. Select the code below and copy it manually.',
  '请允许浏览器打开回放页面。':'Allow your browser to open the replay page.',
  '回放页面未能接收记录，请下载文件后在回放站打开。':'The replay page could not receive this game. Download the file and open it in the viewer.',
  '2048 首页':'2048 home', '主导航':'Main navigation', '对局':'Play', '对局记录':'History', '规则':'Rules', '设置':'Settings',
  '语言':'Language', '显示节点':'Show times', '显示排行':'Show rankings', '本地预览':'Local preview', '登录 / 体验':'Sign in / Try', '退出':'Sign out',
  '重试':'Retry', '补传历史':'Retry upload', '节点用时':'Milestone times', '隐藏节点用时':'Hide milestone times', '步':'moves',
  '首次达成':'First reached', '用时':'Time', '节点用时列表':'Milestone times', '练习与暂停计入连续用时。':'Practice and pauses count toward elapsed time.',
  '棋盘变体':'Board variants', '分数':'Score', '最高分':'Best', '练习板':'Practice', '重新开始':'New game', '重新开始（R）':'New game (R)',
  '继续本局':'Keep playing', '开始新局':'Start a new game', '回看本局':'Replay this game', '明确重开':'Start over', '去练习':'Practice',
  '练习局面编码':'Practice position code', '设置局面':'Set position', '棋块调色盘':'Tile palette', '隐藏 32k':'Hide 32k',
  '↶ 撤销':'↶ Undo', '重做 ↷':'Redo ↷', '重置局面':'Reset position', '清空棋盘':'Clear board', '手动出数':'Manual spawn',
  '等待出数：空格左键出 2，右键出 4。':'Place a tile in an empty cell: left click for 2, right click for 4.',
  '触屏点放：':'Tap to place:', '练习新增':'Practice score:', '分 ·':'points ·', '返回正式局 →':'Back to game →',
  '已保存':'Saved', '· 已校验':'· Verified', '高分局 · 需联网':'High score · Online required', '暂停':'Pause',
  '/ WASD / HJKL 移动':'/ WASD / HJKL to move', '滑动棋盘':'Swipe to move', 'R 重开':'R to restart',
  '排行榜':'Rankings', '隐藏排行榜':'Hide rankings', '总榜':'All time', '本周':'This week', '排名 / 玩家':'Rank / Player',
  '排行榜列表':'Rankings', '正在读取榜单…':'Loading rankings…', '暂无成绩':'No scores yet',
  '最大块':'Largest tile', '回放 ↗':'Replay ↗', '完整榜单':'Full rankings',
  '返回棋盘':'Back to board', '登录后查看正式对局记录':'Sign in to view your game history', '登录 / 本地体验':'Sign in / Local trial',
  '已归档对局':'Archived games', '自然结束':'Completed', '最佳':'Best', '对局历史':'Game history', '刷新':'Refresh',
  '还没有归档对局':'No archived games yet', '完成、重开或放弃的正式局会出现在这里。':'Completed, restarted and abandoned ranked games appear here.',
  '分':'points', '查看回放 ↗':'Replay ↗', '加载更早对局':'Load older games', '对局回放':'Game replay', '返回对局':'Back to game',
  '回放':'Replay', '当前分数':'Score', '当前步数':'Move', '最终分数':'Final score', '从此步练习 ↗':'Practice from here ↗',
  '下载二进制回放':'Download replay', '练习使用新的随机出数，':'Practice uses fresh random tiles.', '不会改变原局。':'Your original game stays unchanged.',
  '播放速度':'Playback speed', '回放步号':'Replay move', '跳到第':'Go to move', '播放时折叠超过 2 秒的等待':'Playback caps idle gaps at 2 seconds',
  '关闭':'Close', '确定结束这局，重新开始？':'End this game and start over?', '本局':'This game:', '分，最大棋块':'points, largest tile', '，用时':', time',
  '旧局会保留为重开记录。当前棋盘不会从服务器恢复。':'This game will be archived as restarted. The server cannot restore your current board.',
  '保存记录并重开':'Save and restart', '登录':'Sign in', '使用本地体验账号':'Use local trial account', '邮箱':'Email', '密码':'Password', '对局规则':'Game rules',
  '四种棋盘各自保存。同账号、同浏览器、同变体只有一局；不同设备不共享进行中存档。':'Each variant is saved separately. One active game per account, browser and variant. Active games are not shared across devices.',
  '标准出数：90% 出 2，10% 出 4。正式局禁止悔棋、AI、查表与他人喂招。':'Tiles spawn as 2 (90%) or 4 (10%). Ranked games prohibit undo, AI, tablebases and outside move advice.',
  '随时可去练习、摆盘、手动出数。练习不标记、不影响排位，也不延续原局随机序列。':'Practice, board editing and manual spawning are always available. Practice does not affect ranking eligibility or continue the original random sequence.',
  '超过变体阈值后必须联网，断线暂停操作。重入时本地进度不能落后于服务器留档。':'Above the variant threshold, a connection is required. Disconnection pauses play. On return, local progress must not be behind the server record.',
  '服务器只保存和验证，不恢复或覆盖本地棋盘。不要清除浏览器存储。':'The server stores and verifies records without restoring or replacing local boards. Keep your browser storage intact.',
  '自然结束且验证通过的正式局自动上榜，低分死亡局同样上传。重开或放弃的对局仅在超过高分阈值时上传，失败不额外提醒；上榜须经站长审核后手动准入。':'Verified games that end naturally rank automatically, including low-score games. Restarted or abandoned games upload only above the high-score threshold, with no extra failure alerts; ranking requires manual approval by the site owner.',
  '深色模式':'Dark mode', '重开确认':'Confirm restart', '开启后，每次重开都先确认。':'Ask for confirmation before every restart.',
  '棋块主题':'Tile theme', '主站自定义配色':'Custom palette', '棋盘、节点用时和调色盘使用同一套配色。':'The board, milestone tiles and palette share the same colors.',
  '确定重置练习局面？':'Reset this practice position?', '当前练习会回到进入练习板时的局面。':'Practice will return to its starting position.',
  '继续练习':'Keep practicing', '重置练习':'Reset practice', '尚无已验证成绩':'No verified scores yet', '访客练习':'Guest play', '正式对局':'Ranked game',
  '新游戏':'New game', '检查中…':'Checking…', '重新检查':'Check again', '当前局面含大于 32k 的棋块，无法用短编码表示':'Tiles above 32k cannot be represented by a short position code',
  '输入局面编码':'Enter position code', '浏览':'Browse', '擦除':'Erase', '选择棋块开始摆盘，再点一次回到浏览。':'Select a tile to edit. Select it again to browse.',
  '左键涂棋块 · 右键升一级 · 中键降一级':'Left: paint · Right: increase · Middle: decrease', '当前局面已无有效移动':'No moves available',
  '独立随机出数，原局保持不变':'Independent random tiles; original game unchanged', '当前为访客练习。登录后开始正式对局，保留战绩与回放。':'Guest play. Sign in for ranked games, stats and saved replays.',
  '我的对局记录':'My games', '已验证':'Verified', '未通过':'Not verified', '当前浏览器的本地记录':'Local browser record', '已封存对局 · 只读回放':'Archived game · Read-only replay',
  '播放':'Play', '本地体验账号使用独立数据，不连接线上账号。':'The local trial uses separate data, independent of online accounts.', '使用主站账号登录。':'Sign in with your main-site account.',
  '登录中…':'Signing in…', '使用已有账号登录':'Sign in with your account', '准备棋盘':'Preparing board', '正在检查本地进度':'Checking local progress',
  '需要连接服务器':'Connection required', '本局无法继续排位':'This game cannot continue as ranked', '本地存档不可用':'Local storage unavailable',
  '本地存档缺失':'Local save missing', '此变体正在另一页面进行':'This variant is open in another tab', '本局已暂停':'Game paused', '本局结束':'Game over',
  '本局已归档':'Game archived', '稍候':'Please wait', '访客练习保留在本地':'Guest game saved locally', '回放已验证并归档':'Replay verified and archived', '回放等待上传':'Replay awaiting upload',
  '请回到原页面，或关闭原页面后重新检查。其他变体仍可独立游玩。':'Return to the original tab, or close it and check again. Other variants remain available.',
  '棋盘保持不变，连续计时仍在进行。':'Your board is unchanged. Elapsed time continues.', '只验证本地记录，不从服务器加载棋盘。':'Verifying the local record. No board is loaded from the server.',
  '重置练习确认':'Confirm practice reset', '提示':'Notice', '重开':'Restarted', '放弃':'Abandoned', '中断':'Interrupted', '本地服务未就绪，请确认服务已启动后重试。':'The local service is unavailable. Start it and retry.',
  '榜单暂不可用':'Rankings unavailable', '无法读取玩家记录，请稍后重试。':'Could not load player records. Please retry.', '回放不存在、尚未封存，或你没有读取权限。':'The replay is missing, not yet archived, or not accessible to you.',
  '检测到本地进度落后于服务器记录，本局已判定回档，不能继续排位。':'Local progress is behind the server record. This game was flagged for rollback and cannot continue as ranked.',
  '本地记录与服务器已留档前缀不同，本局不能继续排位。':'The local replay differs from the server record. This game cannot continue as ranked.',
  '本局验证未通过，已保留记录，不能继续排位。':'Verification failed. The record is retained, but ranked play cannot continue.',
  '同账号、同浏览器、同变体只能保留一局。':'Only one active game is allowed per account, browser and variant.',
  '此浏览器的该变体已有进行中对局，但没有找到本地存档。可以明确重开，无法从服务器恢复。':'An active game exists for this variant, but its local save is missing. You may restart; server recovery is unavailable.',
  '对局写入权已变化，请重新登录后继续。':'Game ownership changed. Sign in again to continue.', '对局写入权已变化，请重新进入。':'Game ownership changed. Reopen this game.',
  '对局写入权已变化，请重新登录后检查。':'Game ownership changed. Sign in and check again.', '对局写入权已变化，请重新登录。':'Game ownership changed. Sign in again.',
  '对局写入权已变化，请重新检查。':'Game ownership changed. Please check again.', '另一页面已修改本地进度，请重新进入。':'Another tab changed local progress. Reopen the game.',
  '浏览器不支持安全的多标签页锁，请使用较新的浏览器。':'This browser does not support the required tab locking. Use a newer browser.',
  '出数记录验证不一致，本局不能继续排位。':'Tile spawn verification failed. Ranked play cannot continue.', '高分记录缺少越线后的联网校验，本局不能继续排位。':'Required online verification is missing after crossing the high-score threshold.',
  '本地保存失败，已暂停操作。请检查浏览器存储空间。':'Local saving failed. Play is paused. Check your browser storage.', '登录已失效，请重新登录后检查。':'Your session expired. Sign in and check again.',
  '无法连接服务器，请联网后重试。本地棋盘保持不变。':'Cannot reach the server. Reconnect and retry. Your local board is unchanged.', '当前低分局可离线游玩，超过阈值后必须联网。':'You may play offline below the threshold. A connection is required above it.',
  '高分对局需要联网，连接恢复后可继续。':'High-score games require a connection. Play can resume when connected.', '连接暂不可用':'Connection unavailable', '空格':'empty',
  'Enter 重做 · Backspace 撤销':'Enter to redo · Backspace to undo',
  '选择棋块 {0}':'Select tile {0}', '第 {0} 行第 {1} 列，{2}':'Row {0}, column {1}, {2}', '{0} 行 {1} 列棋盘':'{0} rows × {1} columns',
  '请输入不超过 {0} 位的十六进制局面编码。':'Enter a hexadecimal position code of at most {0} digits.',
  '超过 {0} 分后需保持联网，定期留档。四种变体各自保存。':'Above {0} points, stay online for periodic recording. Each variant is saved separately.',
  '{0} 分 · {1}':'{0} points · {1}', '历史回放仍在本地，待补传：{0}':'Replays are saved locally, awaiting upload: {0}',
};
const patterns = Object.entries(en).filter(([key]) => key.includes('{0}')).map(([key,value]) => ({
  regex: new RegExp('^'+key.split(/(\{\d+\})/).map(part => /^\{\d+\}$/.test(part) ? '(.+?)' : part.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('')+'$'), value,
}));
export function t(text) {
  if (language.value !== 'en' || typeof text !== 'string') return text;
  const key = text.trim();
  if (en[key]) return text.replace(key,en[key]);
  for (const {regex,value} of patterns) {
    const match = key.match(regex);
    if (match) return text.replace(key,value.replace(/\{(\d+)\}/g,(_,i) => t(match[Number(i)+1])));
  }
  return text;
}
export function refreshLanguage() {
  language.value = preferences.read().language === 'en' ? 'en' : 'zh';
  document.documentElement.lang = language.value === 'en' ? 'en' : 'zh-CN';
  document.title = language.value === 'en' ? '2048 · Human Play' : '2048 · 人类对局';
}
export function setLanguage(value) {
  if (!['zh','en'].includes(value)) return;
  preferences.update(current => ({ ...current, language: value }));
  refreshLanguage();
}
