<template>
  <div class="human-shell">
    <header class="site-header">
      <a class="brand" href="#game" :aria-label="t(&quot;2048 首页&quot;)">2048</a>
      <nav :aria-label="t(&quot;主导航&quot;)"><a href="#game" :class="{ selected: view === 'game' }">{{ t("对局") }}</a><button :disabled="!run || busy" @click="openReplayExport">{{ t("回放") }}</button><button @click="modal = 'rules'">{{ t("规则") }}</button><button @click="modal = 'settings'">{{ t("设置") }}</button><button v-if="view === 'game' && timingHidden" @click="timingHidden = false">{{ t("显示节点") }}</button><button v-if="view === 'game' && rankingHidden" @click="rankingHidden = false">{{ t("显示排行") }}</button></nav>
      <div class="account-area"><span v-if="localPreview" class="local-badge">{{ t("本地预览") }}</span>
        <button v-if="user" class="account-button" @click="openPlayer(user.id)">{{ user.display_name }}</button>
        <button v-else class="account-button" @click="modal = 'login'">{{ t("登录 / 体验") }}</button>
        <button v-if="user" class="text-button logout" @click="logout">{{ t("退出") }}</button>
      </div>
    </header>

    <main>
      <div v-if="bootError" class="notice danger" role="alert">{{ t(bootError) }}<button @click="boot">{{ t("重试") }}</button></div>
      <div v-if="archiveNotice" class="notice" role="status">{{ t(archiveNotice) }} <button @click="session.flushArchives()">{{ t("补传历史") }}</button></div>

      <template v-if="view === 'game'">
        <div class="play-layout" :class="{ 'without-timing': timingHidden, 'without-ranking': rankingHidden }" :style="{ '--board-height': `${boardHeight}px` }">
          <aside v-if="!timingHidden" class="panel timing-panel"><div class="side-header"><div class="panel-heading"><h2>{{ t("节点用时") }}</h2><button class="hide-panel" :aria-label="t(&quot;隐藏节点用时&quot;)" :title="t(&quot;隐藏节点用时&quot;)" @click="timingHidden = true">−</button></div>
            <div class="timer-row"><div class="timer">{{ duration(elapsed) }}</div><span class="muted small">{{ number(run?.seq || 0) }}{{ t(" 步") }}</span></div></div>
            <div class="milestone-heading"><span>{{ t("首次达成") }}</span><span>{{ t("用时") }}</span></div>
            <div class="side-body milestone-list" role="region" :aria-label="t(&quot;节点用时列表&quot;)" tabindex="0">
            <div v-for="tile in visibleNodes" :key="tile" class="milestone" :class="{ reached: run?.nodes[tile] }">
              <span class="node-tile" :class="{ 'node-tile-large': tile >= 1024, 'node-tile-huge': tile >= 16384 }" :style="tileStyle(tile)">{{ tile }}</span><strong class="node-time">{{ run?.nodes[tile] ? nodeTime(run.nodes[tile].elapsed) : '—' }}</strong>
            </div>
            </div>
            <div class="panel-footer timing-note">{{ t("练习与暂停计入连续用时。") }}</div>
          </aside>

          <section class="game-column">
            <div class="variant-switch" role="group" :aria-label="t(&quot;棋盘变体&quot;)"><button v-for="p in policies?.variants || []" :key="p.id" :class="{ active: variant === p.id }" :disabled="busy" @click="changeVariant(p.id)">{{ p.id.replace('x', ' × ') }}</button></div>
            <div class="score-row"><h1 class="game-title">2048 <small>{{ variant.replace('x', ' × ') }}</small></h1><div class="score-box score-main" :aria-label="t(&quot;分数&quot;)"><span>SCORE</span><strong>{{ number(run?.score || 0) }}</strong></div><div class="score-box" :aria-label="t(&quot;最高分&quot;)"><span>BEST</span><strong>{{ number(bests[variant] || 0) }}</strong></div></div>
            <div class="game-mode-row"><div class="mode-tabs"><button :class="{ active: !practice }" @click="returnToGame">{{ t(run?.guest ? '访客练习' : '正式对局') }}</button><button :class="{ active: practice }" @click="openPractice">{{ t("练习板 ") }}<span>↗</span></button></div><button class="new-game" @click="requestRestart" :disabled="busy || gate === 'other-tab'" :aria-label="t(&quot;重新开始&quot;)" :title="t(&quot;重新开始（R）&quot;)">{{ t(practice ? '重置练习' : '新游戏') }}</button></div>

            <HumanBoard ref="humanBoard" :key="practice ? `practice-${practice.variant}` : run?.id" :board="displayBoard" :transition="practice ? practiceTransition : transition" :rows="boardDimensions[0]" :cols="boardDimensions[1]" :editable="!!practice && (selectedTile !== null || practice.pending)" :hide32k="!!practice && hide32k" :touch-button="practice?.pending && manualTile === 4 ? 2 : 0" @cell="practiceCell" @move="onMove">
              <template v-if="!practice && gate !== 'ready'" #overlay>
                <div class="gate-card" role="status">
                  <h2>{{ t(gateTitle) }}</h2><p>{{ t(gateDescription) }}</p>
                  <div class="gate-actions"><button v-if="['network', 'checking', 'other-tab', 'missing'].includes(gate)" class="primary" :disabled="busy" @click="gate === 'other-tab' || !run ? session.activate() : session.retry()">{{ t(busy ? '检查中…' : '重新检查') }}</button>
                    <button v-if="gate === 'paused'" class="primary" @click="session.resume()">{{ t("继续本局") }}</button>
                    <button v-if="gate === 'ended'" class="primary" @click="requestRestart">{{ t("开始新局") }}</button>
                    <button v-if="gate === 'ended'" @click="openLocalReplay(0)">{{ t("回看本局") }}</button>
                    <button v-if="['rejected','missing','storage'].includes(gate)" :disabled="busy" @click="requestRestart">{{ t("明确重开") }}</button>
                    <button v-if="run && ['network','rejected','paused','ended','checking'].includes(gate)" @click="openPractice">{{ t("去练习") }}</button>
                  </div>
                </div>
              </template>
            </HumanBoard>

            <div class="game-details"><template v-if="practice">
              <form class="practice-position" @submit.prevent="setPracticeBoard"><input v-model="practiceHex" :aria-label="t(&quot;练习局面编码&quot;)" :placeholder="t(displayBoard.some(v => v > 32768) ? '当前局面含大于 32k 的棋块，无法用短编码表示' : '输入局面编码')" autocomplete="off" spellcheck="false" @focus="$event.target.select()"><button type="submit">{{ t("设置局面") }}</button></form>
              <p v-if="practiceError" class="error-text" role="alert">{{ t(practiceError) }}</p>
              <div class="practice-palette"><div class="palette-heading"><strong>{{ t("棋块调色盘") }}</strong><label><input v-model="hide32k" type="checkbox" @change="focusPracticeBoard">{{ t("隐藏 32k") }}</label><span class="palette-status" :style="selectedTile === null ? {} : tileStyle(selectedTile)">{{ t(selectedTile === null ? '浏览' : selectedTile === 0 ? '擦除' : selectedTile) }}</span></div>
                <div class="palette-grid"><button v-for="value in PRACTICE_PALETTE" :key="value" type="button" :class="{ selected: selectedTile === value }" :style="tileStyle(value)" :aria-label="t(value ? `选择棋块 ${value}` : '擦除')" :aria-pressed="selectedTile === value" @click="togglePalette(value)">{{ t(value === 0 ? '擦除' : value >= 1024 ? `${value / 1024}k` : value) }}</button></div>
                <div class="palette-help">{{ t(selectedTile === null ? '选择棋块开始摆盘，再点一次回到浏览。' : '左键涂棋块 · 右键升一级 · 中键降一级') }}</div>
              </div>
              <div class="practice-controls"><button @click="practiceUndo" :disabled="!undoStack.length">{{ t("↶ 撤销") }}</button><button @click="practiceRedo" :disabled="!redoStack.length">{{ t("重做 ↷") }}</button><button @click="requestRestart">{{ t("重置局面") }}</button><button @click="clearPractice">{{ t("清空棋盘") }}</button><label><input v-model="manualSpawn" type="checkbox" @change="focusPracticeBoard">{{ t("手动出数") }}</label></div>
              <div v-if="practice.pending" class="manual-spawn-controls" role="status"><span>{{ t("等待出数：空格左键出 2，右键出 4。") }}</span><div><span>{{ t("触屏点放：") }}</span><button v-for="v in [2,4]" :key="v" :class="{ selected: manualTile === v }" :style="tileStyle(v)" :aria-pressed="manualTile === v" @click="manualTile = v">{{ v }}</button></div></div>
              <p class="practice-hotkeys">{{ t('Enter 重做 · Backspace 撤销') }}</p><div class="practice-foot"><span>{{ t("练习新增 ") }}{{ number(practice.score) }}{{ t(" 分 · ") }}{{ t(isPracticeOver ? '当前局面已无有效移动' : '独立随机出数，原局保持不变') }}</span><button class="text-button" @click="returnToGame">{{ t("返回正式局 →") }}</button></div>
            </template>
            <template v-else><div class="under-board"><span class="save-state">{{ t("已保存 ") }}{{ savedSeq }}{{ t(" 步") }}<span v-if="high">{{ t(" · 已校验 ") }}{{ run?.serverSeq || 0 }}{{ t(" 步") }}</span></span><span v-if="high" class="online-state">{{ t("高分局 · 需联网") }}</span><button class="text-button" @click="session.pause()" :disabled="gate !== 'ready' || busy">{{ t("暂停") }}</button></div>
              <div class="keyboard-hint"><span class="key">↑</span><span class="key">←</span><span class="key">↓</span><span class="key">→</span>{{ t(" / WASD / HJKL 移动 ") }}<span class="hint-separator">·</span>{{ t(" 滑动棋盘 ") }}<span class="hint-separator">·</span>{{ t(" R 重开") }}</div>
              <p class="policy-note">{{ t(run?.guest ? '当前为访客练习。登录后开始正式对局，保留战绩与回放。' : `超过 ${number(run?.threshold ?? activePolicy?.threshold)} 分后需保持联网，定期留档。四种变体各自保存。`) }}</p>
            </template></div>
          </section>

          <aside v-if="!rankingHidden" class="panel ranking-panel"><div class="side-header"><div class="panel-heading"><h2>{{ t("排行榜") }}</h2><button class="hide-panel" :aria-label="t(&quot;隐藏排行榜&quot;)" :title="t(&quot;隐藏排行榜&quot;)" @click="rankingHidden = true">−</button></div><div class="period-tabs"><button :class="{ active: period === 'all' }" @click="period = 'all'">{{ t("总榜") }}</button><button :class="{ active: period === 'week' }" @click="period = 'week'">{{ t("本周") }}</button></div></div>
            <div class="rank-table-heading"><span>{{ t("排名 / 玩家") }}</span><span>{{ t("分数") }}</span></div>
            <div class="side-body ranking-body" role="region" :aria-label="t(&quot;排行榜列表&quot;)" tabindex="0">
            <div v-if="boardLoading" class="empty-state">{{ t("正在读取榜单…") }}</div><div v-else-if="boardError" class="empty-state">{{ boardError }}<button @click="loadBoard">{{ t("重试") }}</button></div><div v-else-if="!leaders.length" class="empty-state"><strong>{{ t("暂无成绩") }}</strong></div>
            <div v-else class="rank-list"><div v-for="item in leaders" :key="item.id" class="rank-row"><span class="rank-number" :class="{ podium: item.rank <= 3 }">{{ item.rank }}</span><button class="rank-name" @click="openPlayer(item.user_id)"><strong>{{ item.display_name }}</strong><small>{{ t("最大块 ") }}{{ number(item.max_tile) }}</small></button><button class="rank-score" @click="openReplay(item.id)">{{ number(item.score) }}<small>{{ t("回放 ↗") }}</small></button></div></div>
            </div><div class="panel-footer"><button class="wide-link" @click="openFullLeaderboard">{{ t("完整榜单") }}</button></div>
          </aside>
        </div>
      </template>

      <section v-else-if="view === 'history' || view === 'player'" class="history-view"><div class="section-top"><h1>{{ playerData?.player.display_name || t('我的对局记录') }}</h1><a class="button-link" href="#game">{{ t("返回棋盘") }}</a></div>
        <div v-if="!user && view === 'history'" class="panel large-empty"><h2>{{ t("登录后查看正式对局记录") }}</h2><button class="primary" @click="modal = 'login'">{{ t("登录 / 本地体验") }}</button></div>
        <template v-else><div class="stats-strip"><div><span>{{ t("已归档对局") }}</span><strong>{{ playerData?.stats.games || 0 }}</strong></div><div><span>{{ t("自然结束") }}</span><strong>{{ playerData?.stats.completed || 0 }}</strong></div><div v-for="(dims,key) in VARIANTS" :key="key"><span>{{ key.replace('x',' × ') }}{{ t(" 最佳") }}</span><strong>{{ number(playerData?.bests[key] || 0) }}</strong></div></div>
          <div class="panel history-table"><div class="table-title"><h2>{{ t("对局历史") }}</h2><button @click="loadPlayer(false)">{{ t("刷新") }}</button></div><p v-if="historyError" class="notice danger">{{ t(historyError) }}</p><div v-if="!historyEntries.length" class="large-empty"><h3>{{ t("还没有归档对局") }}</h3><p>{{ t("完成、重开或放弃的正式局会出现在这里。") }}</p></div><div v-for="item in historyEntries" :key="item.id" class="history-row"><span class="variant-tag">{{ item.variant.replace('x','×') }}</span><div><strong>{{ number(item.score) }}{{ t(" 分") }}</strong><small>{{ date(item.ended_at) }} · {{ reasonText(item.reason) }}</small></div><span>{{ number(item.max_tile) }}<small>{{ t("最大块") }}</small></span><span>{{ duration(item.elapsed) }}<small>{{ item.moves }}{{ t(" 步") }}</small></span><span class="verified">{{ t(item.eligibility === 'eligible' ? '已验证' : '未通过') }}</span><button @click="openReplay(item.id)">{{ t("查看回放 ↗") }}</button></div><button v-if="playerData?.next_cursor" class="wide-link" @click="loadPlayer(true)">{{ t("加载更早对局") }}</button></div>
        </template>
      </section>

      <section v-else-if="view === 'replay'" class="replay-view"><div class="section-top"><h1>{{ t("对局回放") }}</h1><a class="button-link" href="#game">{{ t("返回对局") }}</a></div><p v-if="replayError" class="notice danger">{{ t(replayError) }}</p>
        <div v-if="replayData" class="replay-layout"><div class="panel replay-summary"><h2>{{ replayData.header.variant.replace('x',' × ') }}{{ t(" 回放") }}</h2><p>{{ t(localReplay ? '当前浏览器的本地记录' : '已封存对局 · 只读回放') }}</p><strong class="large-number">{{ number(replayState?.score || 0) }}</strong><span class="muted">{{ t("当前分数") }}</span><dl><dt>{{ t("当前步数") }}</dt><dd>{{ replayStep }} / {{ replayData.total }}</dd><dt>{{ t("节点用时") }}</dt><dd>{{ duration(replayState?.elapsed || 0) }}</dd><dt>{{ t("最终分数") }}</dt><dd>{{ number(replayData.final.score) }}</dd></dl><button class="primary" @click="practiceFromReplay">{{ t("从此步练习 ↗") }}</button><button v-if="replayBuffer" @click="downloadReplay">{{ t("下载二进制回放") }}</button><p class="small muted">{{ t("练习使用新的随机出数，") }}<br>{{ t("不会改变原局。") }}</p></div><div><HumanBoard :board="replayState.board" :transition="replayTransition" :rows="VARIANTS[replayData.header.variant][0]" :cols="VARIANTS[replayData.header.variant][1]" /><div class="replay-controls"><button @click="seekReplay(0)">|‹</button><button @click="seekReplay(replayStep - 1)">‹</button><button class="primary" @click="replayPlaying = !replayPlaying">{{ t(replayPlaying ? '暂停' : '播放') }}</button><button @click="seekReplay(replayStep + 1)">›</button><button @click="seekReplay(replayData.total)">›|</button><select v-model.number="replaySpeed" :aria-label="t(&quot;播放速度&quot;)"><option :value="1">1×</option><option :value="4">4×</option><option :value="16">16×</option></select></div><input class="replay-slider" type="range" :min="0" :max="replayData.total" :value="replayStep" :aria-label="t(&quot;回放步号&quot;)" @input="seekReplay(Number($event.target.value))"><div class="replay-step-entry"><label>{{ t("跳到第 ") }}<input type="number" :min="0" :max="replayData.total" :value="replayStep" @change="seekReplay(Number($event.target.value))">{{ t(" 步") }}</label><span class="small muted">{{ t("播放时折叠超过 2 秒的等待") }}</span></div></div></div>
      </section>
    </main>

    <div v-if="modal" class="modal-backdrop" @click.self="closeModal"><section class="modal" role="dialog" aria-modal="true" :aria-label="t(modalTitle)" @keydown.esc="closeModal"><button class="modal-close" @click="closeModal" :aria-label="t(&quot;关闭&quot;)">×</button>
      <template v-if="['replay-export', 'archive-failure'].includes(modal)"><h2>{{ t(modal === 'archive-failure' ? '回放上传失败，请保存回放' : '回放') }}</h2>
        <template v-if="modal === 'archive-failure'"><p role="alert">{{ t('本局回放尚未确认上传成功。请立即下载或复制回放并妥善保存，后续可交由站长审核并手动录入。请勿清除浏览器数据。') }}</p><p v-if="currentFailure" class="small">{{ t('对局编号') }}：{{ currentFailure.run.id }}</p></template>
        <p v-if="currentExport" class="replay-export-summary">{{ currentExport.variant.replace('x', ' × ') }} · {{ number(currentExport.score) }} {{ t("分") }} · {{ t("截至第") }} {{ number(currentExport.moves) }} {{ t("步") }}</p>
        <div class="replay-export-actions"><button :disabled="!currentExport" @click="copyCurrentReplay">{{ t("复制回放") }}</button><button ref="safeButton" :disabled="!currentExport" @click="downloadCurrentReplay">{{ t("下载回放文件") }}</button><button class="primary" :disabled="!currentExport || exportBusy" @click="viewCurrentReplay">{{ t("在回放站查看 ↗") }}</button></div>
        <p v-if="exportNotice" role="status">{{ t(exportNotice) }}</p>
        <label v-if="exportCopyFallback" class="replay-copy-fallback">{{ t("回放代码") }}<textarea readonly :value="currentExport?.text" @focus="$event.target.select()" @click="$event.target.select()"></textarea></label>
      </template>
      <template v-else-if="modal === 'restart'"><h2>{{ t("确定结束这局，重新开始？") }}</h2><p>{{ t("本局 ") }}{{ number(run?.score || 0) }}{{ t(" 分，最大棋块 ") }}{{ number(Math.max(...(run?.board || [0]))) }}{{ t("，用时 ") }}{{ duration(elapsed) }}。</p><p>{{ t("旧局会保留为重开记录。当前棋盘不会从服务器恢复。") }}</p><div class="modal-actions"><button ref="safeButton" class="primary" @click="closeModal">{{ t("继续本局") }}</button><button @click="confirmRestart">{{ t("保存记录并重开") }}</button></div></template>
      <template v-else-if="modal === 'login'"><h2>{{ t("登录") }}</h2><p>{{ t(localPreview ? '本地体验账号使用独立数据，不连接线上账号。' : '使用主站账号登录。') }}</p><button v-if="localPreview" class="primary full-width" :disabled="authBusy" @click="localLogin">{{ t("使用本地体验账号") }}</button><form @submit.prevent="login"><label>{{ t("邮箱") }}<input v-model="email" type="email" required autocomplete="username"></label><label>{{ t("密码") }}<input v-model="password" type="password" required autocomplete="current-password"></label><p v-if="authError" class="error-text">{{ t(authError) }}</p><button :disabled="authBusy" class="full-width" type="submit">{{ t(authBusy ? '登录中…' : '使用已有账号登录') }}</button></form></template>
      <template v-else-if="modal === 'rules'"><h2>{{ t("对局规则") }}</h2><ul class="rules-list"><li>{{ t("四种棋盘各自保存。同账号、同浏览器、同变体只有一局；不同设备不共享进行中存档。") }}</li><li>{{ t("标准出数：90% 出 2，10% 出 4。正式局禁止悔棋、AI、查表与他人喂招。") }}</li><li>{{ t("随时可去练习、摆盘、手动出数。练习不标记、不影响排位，也不延续原局随机序列。") }}</li><li>{{ t("超过变体阈值后必须联网，断线暂停操作。重入时本地进度不能落后于服务器留档。") }}</li><li>{{ t("服务器只保存和验证，不恢复或覆盖本地棋盘。不要清除浏览器存储。") }}</li><li>{{ t("自然结束且验证通过的正式局自动上榜，低分死亡局同样上传。重开或放弃的对局仅在超过高分阈值时上传，失败不额外提醒；上榜须经站长审核后手动准入。") }}</li></ul></template>
      <template v-else-if="modal === 'settings'"><h2>{{ t("设置") }}</h2><label class="theme-setting language-setting">{{ t("语言") }}<select :value="language" :aria-label="t('语言')" @change="setLanguage($event.target.value)"><option value="zh">简体中文</option><option value="en">English</option></select></label><label class="setting-toggle"><span>{{ t("深色模式") }}</span><input type="checkbox" :checked="darkMode" @change="setDarkMode($event.target.checked)"></label><label class="setting-toggle"><span>{{ t("重开确认") }}</span><input type="checkbox" v-model="alwaysConfirmRestart"></label><p class="setting-help">{{ t("开启后，每次重开都先确认。") }}</p><label class="theme-setting">{{ t("棋块主题") }}<select :value="themeName" :aria-label="t(&quot;棋块主题&quot;)" @change="chooseTheme($event.target.value)"><option v-if="themeName === 'custom'" value="custom">{{ t("主站自定义配色") }}</option><option v-for="name in themeNames" :key="name" :value="name">{{ name }}</option></select></label><div class="theme-preview"><span v-for="value in PRACTICE_PALETTE.slice(1)" :key="value" :style="tileStyle(value)">{{ value >= 1024 ? `${value / 1024}k` : value }}</span></div><p>{{ t("棋盘、节点用时和调色盘使用同一套配色。") }}</p></template>
      <template v-else-if="modal === 'practice-restart'"><h2>{{ t("确定重置练习局面？") }}</h2><p>{{ t("当前练习会回到进入练习板时的局面。") }}</p><div class="modal-actions"><button ref="safeButton" class="primary" @click="closeModal">{{ t("继续练习") }}</button><button @click="closeModal(); resetPractice()">{{ t("重置练习") }}</button></div></template>
      <template v-else-if="modal === 'leaderboard'"><h2>{{ variant.replace('x',' × ') }} · {{ t(period === 'week' ? '本周' : '总榜') }}</h2><div v-if="fullLoading" class="large-empty">{{ t("正在读取榜单…") }}</div><div v-else-if="fullError" class="large-empty">{{ t(fullError) }}<button @click="openFullLeaderboard">{{ t("重试") }}</button></div><div v-else-if="!fullLeaders.length" class="large-empty">{{ t("尚无已验证成绩") }}</div><div v-for="item in fullLeaders" :key="item.id" class="rank-row"><span class="rank-number">{{ item.rank }}</span><button class="rank-name" @click="closeModal(); openPlayer(item.user_id)">{{ item.display_name }}</button><button class="rank-score" @click="closeModal(); openReplay(item.id)">{{ number(item.score) }} ↗</button></div></template>
    </section></div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, shallowRef, watch } from 'vue';
import HumanBoard from './HumanBoard.vue';
import { authClient } from '../services/auth/authClient.js';
import { storeDeviceSession, clearDeviceSession } from '../services/auth/sessionTokenStore.js';
import { json, request } from './client.js';
import { useHumanSession } from './session.js';
import { needsReplayUpload } from './archivePolicy.js';
import { VARIANTS, DIRECTIONS, NODE_TILES, move, randomSpawn, isOver, clone, buildReplay, parseReplay } from './engine.js';
import { createLocalStorageStore } from '../services/storage/localStorageStore.js';
import { useHumanAppearance, tileStyle } from './appearance.js';
import { PRACTICE_PALETTE, practiceCellValue, practiceBoardHex, parsePracticeHex, nodeTime } from './practice.js';
import { t, language, setLanguage } from './i18n.js';
import { exportCurrentReplay, openReplayViewer } from './replayExport.js';

const { themeName, themeNames, hide32k, darkMode, setDarkMode, chooseTheme, refresh: refreshAppearance } = useHumanAppearance();

const user = shallowRef(null), policies = shallowRef(null), localPreview = ref(false), bootError = ref('');
const session = useHumanSession(user, policies);
const { run, variant, gate, busy, error, archiveNotice, archiveFailures, savedSeq, transition } = session;
const currentFailure = computed(() => archiveFailures.value.find(item => item.run.userId === user.value?.id));
const clock = ref(Date.now()), modal = ref(''), safeButton = ref(null), view = ref('game');
const fullLeaders = ref([]), fullLoading = ref(false), fullError = ref('');
const displayRequests = new Map();
function displayJson(path) {
  const key = `${user.value?.id || 'guest'}:${path}`;
  if (!displayRequests.has(key)) displayRequests.set(key, json(path).finally(() => displayRequests.delete(key)));
  return displayRequests.get(key);
}
const period = ref('all'), leaders = ref([]), boardLoading = ref(false), boardError = ref(''), bests = ref({});
const email = ref(''), password = ref(''), authBusy = ref(false), authError = ref('');
const playerData = shallowRef(null), historyEntries = ref([]), historyError = ref(''), playerId = ref(null);
const practice = shallowRef(null), practiceOrigin = shallowRef(null), undoStack = ref([]), redoStack = ref([]), manualSpawn = ref(false), selectedTile = ref(null);
const practiceHex = ref(''), practiceError = ref(''), manualTile = ref(2);
const humanBoard = ref(null);
const currentExport = shallowRef(null), exportNotice = ref(''), exportCopyFallback = ref(false), exportBusy = ref(false);
const practiceTransition = shallowRef(null);
const layoutStore = createLocalStorageStore({ key: 'human-layout', version: 1, defaultValue: {} });
const layoutPrefs = layoutStore.read();
const settingsStore = createLocalStorageStore({ key: 'human-settings', version: 1, defaultValue: {} });
const alwaysConfirmRestart = ref(!!settingsStore.read().alwaysConfirmRestart);
watch(alwaysConfirmRestart, value => settingsStore.update(current => ({ ...current, alwaysConfirmRestart: value })));
const timingHidden = ref(!!layoutPrefs.timingHidden), rankingHidden = ref(!!layoutPrefs.rankingHidden);
const boardHeight = ref(500);
// 38px tile + 6px gap, with 8px padding and a 1px border on each side.
const visibleNodes = computed(() => {
  const defaultCount = Math.max(0, Math.floor((boardHeight.value - 18 + 6) / 44));
  return NODE_TILES.filter((tile, index) => index < defaultCount || run.value?.nodes?.[tile]);
});
let boardObserver;
watch(humanBoard, board => {
  boardObserver?.disconnect();
  if (!board?.$el) return;
  const element = board.$el;
  boardObserver = new ResizeObserver(() => { boardHeight.value = element.getBoundingClientRect().height; });
  boardObserver.observe(element);
}, { flush: 'post' });
watch([timingHidden, rankingHidden], ([timingHidden, rankingHidden]) => layoutStore.write({ timingHidden, rankingHidden }));
onUnmounted(() => boardObserver?.disconnect());
const replayData = shallowRef(null), replayState = shallowRef(null), replayStep = ref(0), replayPlaying = ref(false), replaySpeed = ref(1), replayError = ref(''), localReplay = ref(false);
const replayTransition = shallowRef(null);
let replayBuffer = null, replayId = '', replayTimeout, clockTimer, boardSerial = 0, playerSerial = 0;
const activePolicy = computed(() => policies.value?.variants.find(v => v.id === variant.value));
const high = computed(() => !!session.high());
const elapsed = computed(() => run.value?.reason ? run.value.elapsed : run.value?.firstMoveAt ? Math.max(run.value.elapsed, clock.value - run.value.firstMoveAt) : 0);
const displayBoard = computed(() => practice.value?.board || run.value?.board || Array((VARIANTS[variant.value] || [4,4]).reduce((a,b) => a*b)).fill(0));
const boardDimensions = computed(() => VARIANTS[practice.value?.variant || variant.value]);
const isPracticeOver = computed(() => practice.value && isOver(practice.value.board, practice.value.variant));
const gateTitle = computed(() => ({ loading: '准备棋盘', checking: '正在检查本地进度', network: '需要连接服务器', rejected: '本局无法继续排位', storage: '本地存档不可用', missing: '本地存档缺失', 'other-tab': '此变体正在另一页面进行', paused: '本局已暂停', ended: run.value?.archived ? '本局已归档' : '本局结束' }[gate.value] || '稍候'));
const gateDescription = computed(() => ['network','rejected','storage','missing'].includes(gate.value) ? error.value : gate.value === 'ended' ? `${number(run.value?.score || 0)} 分 · ${run.value?.guest ? '访客练习保留在本地' : run.value?.archived ? '回放已验证并归档' : needsReplayUpload(run.value) ? '回放等待上传' : '回放仅保存在本地'}` : gate.value === 'other-tab' ? '请回到原页面，或关闭原页面后重新检查。其他变体仍可独立游玩。' : gate.value === 'paused' ? '棋盘保持不变，连续计时仍在进行。' : '只验证本地记录，不从服务器加载棋盘。');
const modalTitle = computed(() => ({ restart: '重开确认', 'practice-restart': '重置练习确认', login: '登录', rules: '对局规则', leaderboard: '排行榜', settings: '设置', 'replay-export': '回放', 'archive-failure': '回放上传失败，请保存回放' }[modal.value] || '提示'));
const number = value => new Intl.NumberFormat(language.value === 'en' ? 'en-US' : 'zh-CN').format(value || 0);
function duration(ms) { const s = Math.floor(Math.max(0, ms) / 1000); return `${Math.floor(s / 3600).toString().padStart(2,'0')}:${Math.floor(s / 60 % 60).toString().padStart(2,'0')}:${(s % 60).toString().padStart(2,'0')}`; }
const date = timestamp => new Date(timestamp * 1000).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
const reasonText = reason => t({ game_over: '自然结束', restarted: '重开', abandoned: '放弃', interrupted: '中断' }[reason] || reason);

let booting = true;
async function boot() {
  bootError.value = '';
  try {
    policies.value = await json('/api/human/config');
    const identity = await authClient.me(); user.value = identity.user || null;
    localPreview.value = !!(await json('/api/human/local-preview').catch(() => ({}))).local_preview;
    await session.activate(); await loadBoard(); await loadBests(); await route();
  } catch { bootError.value = '本地服务未就绪，请确认服务已启动后重试。'; }
  finally { booting = false; }
}
async function localLogin() {
  authBusy.value = true; authError.value = '';
  try { const identity = await json('/api/human/local-session', { method: 'POST' }); storeDeviceSession(identity); user.value = identity.user; closeModal(); practice.value = null; await session.activate(); await loadBests(); }
  catch (e) { authError.value = e.message; } finally { authBusy.value = false; }
}
async function login() {
  authBusy.value = true; authError.value = '';
  try { const identity = await authClient.login({ email: email.value, password: password.value }); user.value = identity.user; password.value = ''; closeModal(); practice.value = null; await session.activate(); await loadBests(); }
  catch (e) { authError.value = e.message; } finally { authBusy.value = false; }
}
async function logout() { if (busy.value) return; await authClient.logout().catch(() => {}); clearDeviceSession(); user.value = null; bests.value = {}; practice.value = null; await session.activate(); location.hash = 'game'; }
async function changeVariant(id) { if (id === variant.value) return; practice.value = null; await session.activate(id); }
async function loadBoard() {
  const serial = ++boardSerial; boardLoading.value = true; boardError.value = '';
  try { const result = await displayJson(`/api/human/leaderboards?variant=${variant.value}&period=${period.value}&limit=10`); if (serial === boardSerial) leaders.value = result.entries; }
  catch { if (serial === boardSerial) boardError.value = '榜单暂不可用'; }
  finally { if (serial === boardSerial) boardLoading.value = false; }
}
async function openFullLeaderboard() {
  modal.value = 'leaderboard'; fullLoading.value = true; fullError.value = ''; fullLeaders.value = [];
  const key = `${variant.value}:${period.value}`;
  try {
    const data = await displayJson(`/api/human/leaderboards?variant=${variant.value}&period=${period.value}&limit=100`);
    if (key === `${variant.value}:${period.value}`) fullLeaders.value = data.entries;
  } catch { fullError.value = '榜单暂不可用'; }
  finally { fullLoading.value = false; }
}
async function loadBests() {
  const id = user.value?.id; if (!id) return;
  const data = await displayJson('/api/human/me/bests').catch(() => null);
  if (data && id === user.value?.id) bests.value = data.bests;
}

function requestRestart() {
  if (practice.value) { if (alwaysConfirmRestart.value) modal.value = 'practice-restart'; else resetPractice(); return; }
  if (busy.value) return;
  if (alwaysConfirmRestart.value || (run.value?.score || 0) >= (activePolicy.value?.restart_threshold || 0) || high.value || ['missing','rejected','storage'].includes(gate.value)) modal.value = 'restart';
  else session.restart();
}
async function confirmRestart() { closeModal(); await session.restart(); }
function closeModal() {
  if (modal.value === 'archive-failure' && currentFailure.value) session.dismissArchiveFailure(currentFailure.value.run.id);
  modal.value = '';
}
function setPractice(board, v) {
  practiceTransition.value = null;
  practiceOrigin.value = { board: [...board], variant: v, score: 0, pending: false };
  practice.value = clone(practiceOrigin.value); undoStack.value = []; redoStack.value = []; selectedTile.value = null; practiceError.value = ''; manualTile.value = 2;
}
function openPractice() { if (!practice.value) setPractice(run.value?.board || displayBoard.value, variant.value); }
function returnToGame() { practice.value = null; selectedTile.value = null; if (high.value && !run.value?.reason) session.retry(); }
function resetPractice() { if (practiceOrigin.value) { practiceTransition.value = null; practice.value = clone(practiceOrigin.value); undoStack.value = []; redoStack.value = []; selectedTile.value = null; } }
function rememberPractice() { undoStack.value.push(clone(practice.value)); if (undoStack.value.length > 1000) undoStack.value.shift(); redoStack.value = []; }
function practiceUndo() { if (!undoStack.value.length) return; practiceTransition.value = null; redoStack.value.push(clone(practice.value)); practice.value = undoStack.value.pop(); }
function practiceRedo() { if (!redoStack.value.length) return; practiceTransition.value = null; undoStack.value.push(clone(practice.value)); practice.value = redoStack.value.pop(); }
function clearPractice() { rememberPractice(); practiceTransition.value = null; practice.value = { ...practice.value, board: practice.value.board.map(() => 0), score: 0, pending: false }; }
function togglePalette(value) { selectedTile.value = selectedTile.value === value ? null : value; }
function focusPracticeBoard() { nextTick(() => humanBoard.value?.$el?.focus()); }
function setPracticeBoard() {
  if (!practice.value) return;
  const board = parsePracticeHex(practiceHex.value, practice.value.board.length);
  if (!board) { practiceError.value = `请输入不超过 ${practice.value.board.length} 位的十六进制局面编码。`; return; }
  rememberPractice(); selectedTile.value = null; practiceError.value = '';
  practiceTransition.value = null;
  practice.value = { ...practice.value, board, score: 0, pending: false };
}
function practiceCell(index, button = 0) {
  if (!practice.value) return;
  const current = practice.value.board[index];
  const value = practiceCellValue(current, selectedTile.value, button, practice.value.pending);
  if (value === current) return;
  if (!practice.value.pending) rememberPractice();
  const board = [...practice.value.board]; board[index] = value;
  practiceTransition.value = practice.value.pending ? { fromBoard: practice.value.board, toBoard: board, spawn: index } : null;
  practice.value = { ...practice.value, board, pending: false };
}
function onMove(direction) {
  if (modal.value || view.value !== 'game') return;
  if (!practice.value) { session.play(direction); return; }
  if (practice.value.pending) return;
  const moved = move(practice.value.board, ...VARIANTS[practice.value.variant], direction);
  if (!moved.changed) return;
  rememberPractice(); if (!manualSpawn.value) randomSpawn(moved.board);
  practiceTransition.value = { fromBoard: practice.value.board, toBoard: moved.board, direction };
  practice.value = { ...practice.value, board: moved.board, score: practice.value.score + moved.score, pending: manualSpawn.value };
}
function keydown(e) {
  if (e.target.closest?.('.side-body')) return;
  if (e.isComposing || e.ctrlKey || e.metaKey || e.altKey || e.target.isContentEditable || ['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
  if (e.key === 'Escape') { closeModal(); return; }
  if (modal.value || view.value !== 'game') return;
  if (practice.value && ['Enter','NumpadEnter'].includes(e.code)) {
    e.preventDefault(); e.stopPropagation(); if (!e.repeat) practiceRedo(); return;
  }
  if (e.repeat) return;
  if (DIRECTIONS[e.code] !== undefined) { e.preventDefault(); onMove(DIRECTIONS[e.code]); }
  else if (e.key.toLowerCase() === 'r') { e.preventDefault(); requestRestart(); }
  else if (practice.value && e.key.toLowerCase() === 'z') { e.preventDefault(); practiceUndo(); }
  else if (practice.value && ['Backspace','Delete'].includes(e.code)) { e.preventDefault(); practiceUndo(); }
  else if (practice.value && e.code === 'KeyE') { e.preventDefault(); togglePalette(0); }
  else if (practice.value && e.code === 'KeyQ') { e.preventDefault(); manualSpawn.value = !manualSpawn.value; }
}
function openReplayExport() {
  if (!run.value || busy.value) return;
  exportNotice.value = ''; exportCopyFallback.value = false; currentExport.value = null;
  try { currentExport.value = exportCurrentReplay(run.value, session.getEvents()); }
  catch { exportNotice.value = '无法读取当前回放，请重新进入本局后重试。'; }
  modal.value = 'replay-export';
}
async function copyCurrentReplay() {
  try { await navigator.clipboard.writeText(currentExport.value.text); exportNotice.value = '回放已复制'; exportCopyFallback.value = false; }
  catch { exportNotice.value = '无法自动复制，请选中下方代码手动复制。'; exportCopyFallback.value = true; }
}
function downloadCurrentReplay() {
  const exported = currentExport.value;
  const url = URL.createObjectURL(new Blob([exported.binary], { type: 'application/octet-stream' }));
  const link = document.createElement('a'); link.href = url; link.download = exported.filename;
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 60000);
}
async function viewCurrentReplay() {
  exportBusy.value = true; exportNotice.value = '';
  try { await openReplayViewer(currentExport.value, language.value); }
  catch (e) { exportNotice.value = e.message === 'popup_blocked' ? '请允许浏览器打开回放页面。' : '回放页面未能接收记录，请下载文件后在回放站打开。'; }
  finally { exportBusy.value = false; }
}
function openPlayer(id) { location.hash = `player/${id}`; }
function openReplay(id) { location.hash = `replay/${id}?step=0`; }
async function loadPlayer(more = false) {
  const id = view.value === 'history' ? user.value?.id : playerId.value; if (!id) return;
  const serial = ++playerSerial; historyError.value = '';
  try {
    const result = await displayJson(`/api/human/players/${id}${more && playerData.value?.next_cursor ? `?before=${playerData.value.next_cursor}` : ''}`);
    if (serial !== playerSerial) return; playerData.value = result; historyEntries.value = more ? [...historyEntries.value, ...result.entries] : result.entries;
  } catch { historyError.value = '无法读取玩家记录，请稍后重试。'; }
}
async function openLocalReplay(step) {
  if (!run.value) return; replayPlaying.value = false; replayBuffer = null; localReplay.value = true; replayId = '';
  replayData.value = buildReplay({ header: { run_id: run.value.id, variant: run.value.variant, seed: run.value.seed }, events: session.getEvents() });
  view.value = 'replay'; seekReplay(step); location.hash = 'local-replay';
}
function seekReplay(step) {
  if (!replayData.value || !Number.isFinite(step)) return;
  const target = Math.max(0, Math.min(replayData.value.total, Math.trunc(step)));
  const state = replayData.value.seek(target);
  replayTransition.value = replayPlaying.value && replayState.value && target === replayStep.value + 1
    ? { fromBoard: replayState.value.board, toBoard: state.board, direction: replayData.value.events[replayStep.value][0] & 3 } : null;
  replayStep.value = target; replayState.value = state;
  if (replayId) history.replaceState(null, '', `#replay/${replayId}?step=${replayStep.value}`);
  if (replayStep.value === replayData.value.total) replayPlaying.value = false;
}
function schedulePlayback() {
  clearTimeout(replayTimeout); if (!replayPlaying.value || !replayData.value) return;
  const delta = replayData.value.events[replayStep.value]?.[1] || 100;
  replayTimeout = setTimeout(() => { seekReplay(replayStep.value + 1); schedulePlayback(); }, Math.max(16, Math.min(2000, delta) / replaySpeed.value));
}
function practiceFromReplay() { setPractice(replayState.value.board, replayData.value.header.variant); location.hash = 'game'; }
function downloadReplay() {
  const url = URL.createObjectURL(new Blob([replayBuffer], { type: 'application/octet-stream' })); const link = document.createElement('a');
  link.href = url; link.download = `${replayId}.hpr`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function route() {
  const path = location.hash.slice(1) || 'game'; replayPlaying.value = false;
  if (path.startsWith('replay/')) {
    view.value = 'replay'; replayError.value = ''; replayData.value = null; localReplay.value = false;
    const [id, query] = path.slice(7).split('?'); replayId = id;
    try { replayBuffer = await (await request(`/api/human/replays/${encodeURIComponent(id)}`)).arrayBuffer(); replayData.value = buildReplay(parseReplay(replayBuffer)); seekReplay(Number(new URLSearchParams(query).get('step') || 0)); }
    catch { replayError.value = '回放不存在、尚未封存，或你没有读取权限。'; }
  } else if (path.startsWith('player/')) { view.value = 'player'; playerId.value = Number(path.slice(7)); await loadPlayer(); }
  else if (path === 'history') {
    view.value = 'history';
    await session.flushArchives();
    if (view.value === 'history') await loadPlayer();
  }
  else if (path === 'local-replay' && replayData.value) view.value = 'replay';
  else { view.value = 'game'; if (high.value && !run.value?.reason) await session.retry(); }
}
watch([variant, period], () => { if (!booting) loadBoard(); });
watch(() => [run.value?.id, run.value?.archived], ([id, archived], [oldId, oldArchived]) => { if (archived && id === oldId && !oldArchived) { loadBoard(); loadBests(); } });
watch(currentFailure, async failure => {
  if (!failure) {
    if (modal.value === 'archive-failure') { modal.value = ''; currentExport.value = null; }
    return;
  }
  modal.value = 'archive-failure'; currentExport.value = null;
  exportNotice.value = ''; exportCopyFallback.value = false;
  try {
    const events = await session.failedReplayEvents(failure);
    if (currentFailure.value !== failure || modal.value !== 'archive-failure') return;
    const replay = exportCurrentReplay(failure.run, events);
    replay.filename = replay.filename.replace('.vrs', `-${failure.run.id}.vrs`);
    currentExport.value = replay;
    await nextTick(); safeButton.value?.focus();
  } catch {
    if (currentFailure.value === failure && modal.value === 'archive-failure') exportNotice.value = '无法读取回放，请保留浏览器数据并联系站长。';
  }
});
watch(modal, async value => { if (['restart', 'practice-restart'].includes(value)) { await nextTick(); safeButton.value?.focus(); } });
watch([replayPlaying, replaySpeed], schedulePlayback);
watch(() => practice.value?.board, board => { practiceHex.value = board ? practiceBoardHex(board) : ''; practiceError.value = ''; });
watch(manualSpawn, enabled => {
  if (!enabled && practice.value?.pending) {
    const board = [...practice.value.board]; const spawn = randomSpawn(board);
    practiceTransition.value = spawn ? { fromBoard: practice.value.board, toBoard: board, spawn: spawn.index } : null;
    practice.value = { ...practice.value, board, pending: false };
  }
});
function refreshVisibleAppearance() { if (!document.hidden) refreshAppearance(); }
onMounted(() => { session.start(); clockTimer = setInterval(() => { clock.value = session.now(); }, 500); window.addEventListener('keydown', keydown); window.addEventListener('hashchange', route); window.addEventListener('focus', refreshAppearance); window.addEventListener('storage', refreshAppearance); document.addEventListener('visibilitychange', refreshVisibleAppearance); boot(); });
onUnmounted(() => { session.stop(); clearInterval(clockTimer); clearTimeout(replayTimeout); window.removeEventListener('keydown', keydown); window.removeEventListener('hashchange', route); window.removeEventListener('focus', refreshAppearance); window.removeEventListener('storage', refreshAppearance); document.removeEventListener('visibilitychange', refreshVisibleAppearance); });
</script>
