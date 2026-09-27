<template>
  <div class="human-shell">
    <header class="site-header" :class="{ 'with-metrics': view === 'game' && !practice && (playSettings.showSpeed || playSettings.showFourPercent) }">
      <a class="brand" href="/#game" :aria-label="t(&quot;2048 首页&quot;)" @click.prevent="goGame">2048</a>
      <nav :aria-label="t(&quot;主导航&quot;)"><a href="/#game" :class="{ selected: view === 'game' }" @click.prevent="goGame">{{ t("对局") }}</a><a href="/leaderboard" :class="{ selected: view === 'leaderboard' }" @click.prevent="openFullLeaderboard">{{ t("排行榜") }}</a><a href="/analysis" :class="{ selected: view === 'analysis' }" @click.prevent="openAnalysisLibrary">{{ t('分析库') }}</a><button :disabled="!run || controlsBusy" @click="openReplayExport">{{ t("回放") }}</button><button v-if="user" :class="{ selected: view === 'profile' }" @click="openPlayer(user.display_name)">{{ t('个人主页') }}</button><button @click="modal = 'rules'">{{ t("规则") }}</button><button @click="modal = 'settings'">{{ t("设置") }}</button><a href="https://2048tables.online/" target="_blank" rel="noopener">{{ t('前往主站') }} ↗</a><button v-if="view === 'game' && timingHidden" @click="timingHidden = false">{{ t("显示节点") }}</button><button v-if="view === 'game' && rankingHidden" @click="rankingHidden = false">{{ t("显示排行") }}</button></nav>
      <div class="account-area"><span v-if="localPreview" class="local-badge">{{ t("本地预览") }}</span>
        <HumanAccountMenu v-if="user" :user="user" @saved="handleAccountSaved" @refresh="refreshIdentity" @logout="logout" />
        <button v-else class="account-button" @click="openAuthDialog('login')">{{ t("登录 / 注册") }}</button>
      </div>
      <span v-if="view === 'game' && playSettings.showFourPercent && !practice" class="site-metric site-metric-left">4: {{ fourPercent.toFixed(1) }}% ({{ run?.fourCount || 0 }}/{{ run?.spawnCount || 2 }})</span>
      <span v-if="view === 'game' && playSettings.showSpeed && !practice" class="site-metric site-metric-right">IPS {{ ips }} · MPS {{ mps }}</span>
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
            <div v-for="row in visibleNodeRows" :key="row.key" class="milestone" :class="{ reached: row.time }">
              <span class="node-tile" :class="{ 'node-tile-large': row.tile >= 1024, 'node-tile-huge': row.tile >= 16384 }" :style="[tileStyle(row.tile),{'--node-indent':`${Math.min(row.depth,4)*5}px`}]">{{ row.tile }}</span><strong class="node-time">{{ row.time ? nodeTime(row.time.elapsed) : '—' }}</strong>
            </div>
            </div>
            <div class="panel-footer timing-note">{{ t("练习与暂停计入连续用时。") }}</div>
          </aside>

          <section class="game-column">
            <div class="variant-switch" role="group" :aria-label="t(&quot;棋盘变体&quot;)"><button v-for="p in policies?.variants || []" :key="p.id" :class="{ active: variant === p.id }" :disabled="controlsBusy" @click="changeVariant(p.id)">{{ p.id.replace('x', ' × ') }}</button></div>
            <div class="score-row"><h1 class="game-title">2048 <small>{{ variant.replace('x', ' × ') }}</small></h1><div class="score-box score-main" :aria-label="t(&quot;分数&quot;)"><span>SCORE</span><strong>{{ number(run?.score || 0) }}</strong></div><div class="score-box" :aria-label="t(&quot;最高分&quot;)"><span>BEST</span><strong>{{ number(currentBest) }}</strong></div></div>
            <div class="game-mode-row"><div class="mode-tabs"><button :class="{ active: !practice }" @click="returnToGame">{{ t(run?.guest ? '访客练习' : '正式对局') }}</button><button :class="{ active: practice }" @click="openPractice">{{ t("练习板") }}</button></div><button class="new-game" @click="requestRestart" :disabled="controlsBusy || gate === 'other-tab'" :aria-label="t(&quot;重新开始&quot;)" :title="t(&quot;重新开始（R）&quot;)">{{ t(practice ? '重置练习' : '新游戏') }}</button></div>

            <HumanBoard ref="humanBoard" :key="practice ? `practice-${practice.variant}` : run?.id" :board="displayBoard" :transition="practice ? practiceTransition : transition" :rows="boardDimensions[0]" :cols="boardDimensions[1]" :editable="!!practice && (selectedTile !== null || practice.pending)" :hide32k="!!practice && hide32k" :touch-button="practice?.pending && manualTile === 4 ? 2 : 0" :swipe-sensitivity="playSettings.swipeSensitivity" :animate="animationEnabled" @cell="practiceCell" @move="onMove">
              <template v-if="!practice && gate !== 'ready' && (gate !== 'ended' || terminalOverlayVisible)" #overlay>
                <div class="gate-card" role="status">
                  <button v-if="gate === 'ended'" class="gate-dismiss" type="button" :aria-label="t('关闭')" @click="dismissTerminalOverlay">×</button>
                  <h2>{{ t(gateTitle) }}</h2><p>{{ t(gateDescription) }}</p>
                  <div class="gate-actions"><button v-if="['network', 'checking', 'other-tab', 'missing'].includes(gate)" class="primary" :disabled="busy" @click="gate === 'other-tab' || !run ? session.activate() : session.retry()">{{ t(busy ? '检查中…' : '重新检查') }}</button>
                    <button v-if="gate === 'paused'" class="primary" @click="session.resume()">{{ t("继续本局") }}</button>
                    <button v-if="gate === 'ended'" class="primary" @click="requestRestart">{{ t("开始新局") }}</button>
                    <button v-if="gate === 'ended'" @click="openLocalReplay">{{ t("回看本局") }}</button>
                    <button v-if="['rejected','missing','storage'].includes(gate)" :disabled="busy" @click="requestRestart">{{ t("明确重开") }}</button>
                    <button v-if="run && ['network','rejected','paused','ended','checking'].includes(gate)" @click="openPractice">{{ t("去练习") }}</button>
                  </div>
                </div>
              </template>
            </HumanBoard>

            <div class="game-details"><template v-if="practice">
              <form class="practice-position" @submit.prevent="setPracticeBoard"><input v-model="practiceHex" :aria-label="t(&quot;练习局面编码&quot;)" :placeholder="t(displayBoard.some(v => v > 131072) ? '当前局面含大于 131k 的棋块，无法用短编码表示' : '输入局面编码')" autocomplete="off" spellcheck="false" @focus="$event.target.select()"><button type="submit">{{ t("设置局面") }}</button></form>
              <p v-if="practiceError" class="error-text" role="alert">{{ t(practiceError) }}</p>
              <div class="practice-palette"><div class="palette-heading"><strong>{{ t("棋块调色盘") }}</strong><label><input v-model="hide32k" type="checkbox" @change="focusPracticeBoard">{{ t("隐藏 32k") }}</label><span class="palette-status" :style="selectedTile === null ? {} : tileStyle(selectedTile)">{{ t(selectedTile === null ? '浏览' : selectedTile === 0 ? '擦除' : selectedTile) }}</span></div>
                <div class="palette-grid"><button v-for="value in PRACTICE_PALETTE" :key="value" type="button" :class="{ selected: selectedTile === value }" :style="tileStyle(value)" :aria-label="t(value ? `选择棋块 ${value}` : '擦除')" :aria-pressed="selectedTile === value" @click="togglePalette(value)"><span class="palette-label">{{ t(value === 0 ? '擦除' : value >= 1024 ? `${value / 1024}k` : value) }}</span></button></div>
                <div class="palette-help">{{ t(selectedTile === null ? '选择棋块开始摆盘，再点一次回到浏览。' : '左键涂棋块 · 右键升一级 · 中键降一级') }}</div>
              </div>
              <div class="practice-controls"><button @click="practiceUndo" :disabled="!undoStack.length">{{ t("↶ 撤销") }}</button><button @click="practiceRedo" :disabled="!redoStack.length">{{ t("重做 ↷") }}</button><button @click="requestRestart">{{ t("重置局面") }}</button><button @click="clearPractice">{{ t("清空棋盘") }}</button><label><input v-model="manualSpawn" type="checkbox" @change="focusPracticeBoard">{{ t("手动出数") }}</label></div>
              <div v-if="practice.pending" class="manual-spawn-controls" role="status"><span>{{ t("等待出数：空格左键出 2，右键出 4。") }}</span><div><span>{{ t("触屏点放：") }}</span><button v-for="v in [2,4]" :key="v" :class="{ selected: manualTile === v }" :style="tileStyle(v)" :aria-pressed="manualTile === v" @click="manualTile = v">{{ v }}</button></div></div>
              <p class="practice-hotkeys">{{ t('Enter 重做 · Backspace 撤销') }}</p><div class="practice-foot"><span>{{ t("练习新增 ") }}{{ number(practice.score) }}{{ t(" 分 · ") }}{{ t(isPracticeOver ? '当前局面已无有效移动' : '独立随机出数，原局保持不变') }}</span><button class="text-button" @click="returnToGame">{{ t("返回正式局 →") }}</button></div>
            </template>
            <template v-else><div class="under-board"><span class="save-state">{{ t("已保存 ") }}{{ savedSeq }}{{ t(" 步") }}<span v-if="high">{{ t(" · 已校验 ") }}{{ run?.serverSeq || 0 }}{{ t(" 步") }}</span></span><span v-if="high" class="online-state">{{ t("高分局 · 需联网") }}</span><button class="text-button" @click="pauseGame" :disabled="gate !== 'ready' || controlsBusy">{{ t("暂停") }}</button></div>
              <div class="keyboard-hint"><span class="key">↑</span><span class="key">←</span><span class="key">↓</span><span class="key">→</span>{{ t(" / WASD / HJKL 移动 ") }}<span class="hint-separator">·</span>{{ t(" 滑动棋盘 ") }}<span class="hint-separator">·</span>{{ t(" R 重开") }}</div>
              <p class="policy-note">{{ t(run?.guest ? '当前为访客练习。登录后开始正式对局，保留战绩与回放。' : `超过 ${number(run?.threshold ?? activePolicy?.threshold)} 分后需保持联网，定期留档。四种变体各自保存。`) }}</p>
            </template></div>
          </section>

          <aside v-if="!rankingHidden" class="panel ranking-panel"><div class="side-header"><div class="panel-heading"><h2>{{ t("排行榜") }}</h2><button class="hide-panel" :aria-label="t(&quot;隐藏排行榜&quot;)" :title="t(&quot;隐藏排行榜&quot;)" @click="rankingHidden = true">−</button></div><div class="period-tabs"><button :class="{ active: period === 'all' }" @click="period = 'all'">{{ t("总榜") }}</button><button :class="{ active: period === 'week' }" @click="period = 'week'">{{ t("近7天") }}</button></div></div>
            <div class="rank-table-heading"><span>{{ t("排名 / 玩家") }}</span><span>{{ t("分数") }}</span></div>
            <div class="side-body ranking-body" role="region" :aria-label="t(&quot;排行榜列表&quot;)" tabindex="0">
            <div class="rank-list" :class="{ loading: boardLoading }">
              <div v-for="(item, index) in leaderRows" :key="item?.id || `empty-${index}`" class="rank-row" :class="{ placeholder: !item }"><span class="rank-number" :class="{ podium: item?.rank <= 3 }">{{ item?.rank || '-' }}</span><template v-if="item"><button class="rank-name notranslate" translate="no" :title="item.display_name" @click="openPlayer(item.display_name)"><strong v-fit-ranking-name>{{ item.display_name }}</strong></button><button class="rank-score" :disabled="!item.has_replay" @click="openReplay(item)">{{ number(item.score) }}</button></template></div>
              <div class="rank-row own-rank" :class="{ placeholder: !ownLeader }"><span class="rank-number" :class="{ podium: ownLeader?.rank <= 3 }">{{ ownLeader?.rank || '-' }}</span><template v-if="ownLeader"><button class="rank-name notranslate" translate="no" :title="ownLeader.display_name" @click="openPlayer(ownLeader.display_name)"><strong v-fit-ranking-name>{{ ownLeader.display_name }}</strong></button><button class="rank-score" :disabled="!ownLeader.has_replay" @click="openReplay(ownLeader)">{{ number(ownLeader.score) }}</button></template></div>
            </div>
            <div v-if="boardLoading" class="ranking-status">{{ t("正在读取榜单…") }}</div><div v-else-if="boardError" class="ranking-status">{{ boardError }}<button @click="loadBoard(true)">{{ t("重试") }}</button></div>
            </div><div class="panel-footer"><button class="wide-link" @click="openFullLeaderboard">{{ t("完整榜单") }}</button></div>
          </aside>
        </div>
      </template>

      <KeepAlive><PlayerProfile v-if="view === 'profile'" :username="profileName" :viewer="user" :play-settings="playSettings" @back="goGame" @update:play-settings="updatePlaySettings" @replay="openReplay" @analyze="openAnalysis" /></KeepAlive>

      <HumanLeaderboardPage v-if="view === 'leaderboard'" @back="goGame" @player="openPlayer" @replay="openReplay" />

      <HumanAnalysisLibrary v-if="view === 'analysis'" @back="goGame" @player="openPlayer" @replay="openReplay" />

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
      <template v-else-if="modal === 'rules'"><h2>{{ t("对局规则") }}</h2><ul class="rules-list"><li>{{ t("四种棋盘各自保存。同账号、同浏览器、同变体只有一局；不同设备不共享进行中存档。") }}</li><li>{{ t("标准出数：90% 出 2，10% 出 4。正式局禁止悔棋、AI、查表与他人喂招。") }}</li><li>{{ t("随时可去练习板摆盘、手动出数。练习不影响参与排位。") }}</li><li>{{ t("得分超过阈值后必须联网，断线暂停操作。服务器会间隔存档。") }}</li><li>{{ t("服务器会对已上传对局进行保存和验证，但不可恢复本地对局。不要清除浏览器存储，并请为 C 盘预留一些空间。") }}</li><li>{{ t("对局结束时会上传归档。上传失败会提醒保存回放。请联系站长处理。") }}</li></ul></template>
      <template v-else-if="modal === 'settings'"><h2>{{ t("设置") }}</h2><p v-if="preferenceSyncStatus === 'error'" role="alert">{{ language === 'zh' ? '账号设置尚未同步，请检查网络。' : 'Account settings have not synced. Check your connection.' }} <button @click="retryAccountPreferences">{{ language === 'zh' ? '重试' : 'Retry' }}</button></p>
        <section class="live-setting"><div><strong>{{ t('直播当前对局') }}</strong><p>{{ t('开启后会创建公开直播间。关闭或离开直播不会影响本局操作。') }}</p></div><label class="setting-toggle"><input type="checkbox" :checked="live.enabled.value" :disabled="!user || !run || !!run?.reason || live.state.value === 'connecting'" @change="toggleLive($event.target.checked)"></label></section>
        <div v-if="live.enabled.value" class="live-setting-status"><span :class="['live-dot',{on:live.state.value==='live'}]"></span><span>{{ t(liveStateLabel) }}<small v-if="run">{{ run.variant.replace('x',' × ') }} · {{ number(run.score) }} {{ t('分') }}</small></span><button v-if="live.room.value" @click="live.share(language)">{{ t('分享直播间') }}</button><a v-if="live.room.value" :href="live.room.value.url" target="_blank" rel="noopener">{{ t('打开直播间') }} ↗</a></div><p v-if="live.notice.value" class="setting-help">{{ t(live.notice.value) }}</p><p class="setting-help">{{ t('直播会公开昵称、头像、棋盘、分数、节点用时和操作') }}</p><p class="setting-help">{{ t('礼物实际消耗的常驻 Token 部分，将有 50% 计入主播的常驻 Token。') }}</p>
        <label class="theme-setting language-setting">{{ t("语言") }}<select :value="language" :aria-label="t('语言')" @change="setLanguage($event.target.value)"><option value="zh">简体中文</option><option value="en">English</option></select></label><label class="setting-toggle"><span>{{ t("深色模式") }}</span><input type="checkbox" :checked="darkMode" @change="setDarkMode($event.target.checked)"></label><label class="setting-toggle"><span>{{ t("重开确认") }}</span><input type="checkbox" :checked="alwaysConfirmRestart" @change="updatePlaySettings({...playSettings,alwaysConfirmRestart:$event.target.checked})"></label><p class="setting-help">{{ t("开启后，每次重开都先确认。") }}</p><label class="theme-setting">{{ t("棋块主题") }}<select :value="themeName" :aria-label="t(&quot;棋块主题&quot;)" @change="chooseTheme($event.target.value)"><option v-if="themeName === 'custom'" value="custom">{{ t("主站自定义配色") }}</option><option v-for="name in themeNames" :key="name" :value="name">{{ name }}</option></select></label><div class="theme-preview"><span v-for="value in THEME_TILE_VALUES" :key="value" :style="tileStyle(value)">{{ value }}</span></div><p>{{ t("棋盘、节点用时和调色盘使用同一套配色。") }}</p></template>
      <template v-else-if="modal === 'practice-reminder'"><h2>{{ t('当前是练习板') }}</h2><p>{{ t('你已在练习板走过 40 步。练习出数独立随机，不会计入正式对局。') }}</p><div class="modal-actions"><button class="primary" @click="closeModal(); returnToGame()">{{ t('返回正式局 →') }}</button><button ref="safeButton" @click="closeModal">{{ t('继续练习') }}</button></div></template>
    </section></div>
    <HumanAnalysisDialog v-if="analysisRunId" :run-id="analysisRunId" @close="analysisRunId = ''" />
    <Teleport to="body"><div v-if="authDialogOpen" class="human-auth-overlay" @keydown.esc="closeAuthDialog">
      <button class="human-auth-backdrop" type="button" :aria-label="t('关闭')" @click="closeAuthDialog" />
      <div class="human-auth-dialog"><button class="human-auth-close" type="button" :aria-label="t('关闭')" @click="closeAuthDialog">×</button><AuthPage :initial-mode="authDialogMode" @authenticated="handleAuthenticated" /><button v-if="localPreview" class="human-local-login" type="button" :disabled="authBusy" @click="localLogin">{{ t(authBusy ? '登录中…' : '使用本地体验账号') }}</button></div>
    </div></Teleport>
  </div>
</template>

<script setup>
import { computed, defineAsyncComponent, nextTick, onMounted, onUnmounted, ref, shallowRef, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import HumanBoard from './HumanBoard.vue';
import { authClient } from '../services/auth/authClient.js';
import { storeDeviceSession, clearDeviceSession } from '../services/auth/sessionTokenStore.js';
import { json } from './client.js';
import { useHumanSession } from './session.js';
import { needsReplayUpload } from './archivePolicy.js';
import { fitSingleLineText as vFitRankingName } from './fitSingleLineText.js';
import { VARIANTS, DIRECTIONS, NODE_TILES, move, randomSpawn, isOver, clone } from './engine.js';
import { createLocalStorageStore } from '../services/storage/localStorageStore.js';
import { activateAccountPreferences, preferenceSyncStatus, retryAccountPreferences, saveAccountPreferences } from '../services/preferences/accountPreferences.js';
import { useHumanAppearance, tileStyle } from './appearance.js';
import { PRACTICE_PALETTE, createPracticeMoveReminder, practiceCellValue, practiceBoardHex, parsePracticeHex, nodeTime } from './practice.js';
import { VTH_TILE_VALUES as THEME_TILE_VALUES } from '../services/preferences/savedThemes.js';
import { saveTimerSplits, timerSplitRow } from './timerSplits.js';
import { t, language, setLanguage } from './i18n.js';
import { exportCurrentReplay, openReplayViewer } from './replayExport.js';
import { SPEED_REFRESH_MS, addSpeedSample, countSpeedSamples } from './speedMetrics.js';
import { createLiveBroadcast } from './liveBroadcast.js';
import { activeBestScore } from './bestScore.js';
import { captureLiveAppearance } from './liveAppearance.js';
import { createTerminalOverlay } from './terminalOverlay.js';

const AuthPage = defineAsyncComponent(() => import('../features/auth/AuthPage.vue'));
const HumanAccountMenu = defineAsyncComponent(() => import('./HumanAccountMenu.vue'));
const PlayerProfile = defineAsyncComponent(() => import('./PlayerProfile.vue'));
const HumanLeaderboardPage = defineAsyncComponent(() => import('./HumanLeaderboardPage.vue'));
const HumanAnalysisDialog = defineAsyncComponent(() => import('./HumanAnalysisDialog.vue'));
const HumanAnalysisLibrary = defineAsyncComponent(() => import('./HumanAnalysisLibrary.vue'));

const { themeName, themeNames, hide32k, darkMode, animationEnabled, paletteRevision, setDarkMode, chooseTheme, refresh: refreshAppearance } = useHumanAppearance();
const { locale: sharedLocale } = useI18n();
watch(language, value => { sharedLocale.value = value; }, { immediate: true });

const user = shallowRef(null), policies = shallowRef(null), localPreview = ref(false), bootError = ref('');
const session = useHumanSession(user, policies);
const live = createLiveBroadcast(session, selectedVariant => activeBestScore(bests.value, session.run.value, selectedVariant));
watch(paletteRevision, () => nextTick(() => live.updateAppearance(captureLiveAppearance())), { immediate: true });
const { run, variant, gate, busy, error, archiveNotice, archiveFailures, savedSeq, transition } = session;
const controlsBusy = computed(() => busy.value && !session.moveBusy.value);
const liveStateLabel = computed(() => ({connecting:'正在连接直播',reconnecting:'正在恢复直播',live:'直播中',error:'直播连接失败',off:'直播已关闭'}[live.state.value] || '直播已关闭'));
async function toggleLive(value) { try { if (value) await live.start(); else await live.stop(); } catch { /* status is shown in the settings modal */ } }
const currentFailure = computed(() => archiveFailures.value.find(item => item.run.userId === user.value?.id));
function initialProfileName() {
  if (!location.pathname.startsWith('/user/')) return '';
  try { return decodeURIComponent(location.pathname.slice(6).replace(/\/$/, '')); }
  catch { return ''; }
}
function initialView() {
  if (location.pathname.startsWith('/user/')) return 'profile';
  if (location.pathname === '/leaderboard' || location.pathname === '/leaderboard/') return 'leaderboard';
  if (location.pathname === '/analysis' || location.pathname === '/analysis/') return 'analysis';
  return 'game';
}
const clock = ref(Date.now()), modal = ref(''), safeButton = ref(null), view = ref(initialView());
const terminalOverlayVisible = ref(false);
const terminalOverlay = createTerminalOverlay({ onVisible: value => { terminalOverlayVisible.value = value; } });
function dismissTerminalOverlay() { terminalOverlay.dismiss(run.value?.id); }
const displayRequests = new Map();
const displayCache = new Map();
function displayJson(path, { force = false } = {}) {
  const key = `${user.value?.id || 'guest'}:${path}`;
  if (!force && displayCache.has(key)) return Promise.resolve(displayCache.get(key));
  if (!displayRequests.has(key)) displayRequests.set(key, json(path).then(result => {
    displayCache.set(key, result); return result;
  }).finally(() => displayRequests.delete(key)));
  return displayRequests.get(key);
}
const period = ref('all'), leaders = ref([]), ownLeader = ref(null), boardLoading = ref(false), boardError = ref(''), bests = ref({});
const currentBest = computed(() => activeBestScore(bests.value, run.value, variant.value));
const leaderRows = computed(() => Array.from({ length: 10 }, (_, index) => leaders.value[index] || null));
const authBusy = ref(false), authDialogOpen = ref(false), authDialogMode = ref('login');
const profileName = ref(initialProfileName());
const analysisRunId = ref('');
const practice = shallowRef(null), practiceOrigin = shallowRef(null), undoStack = ref([]), redoStack = ref([]), manualSpawn = ref(false), selectedTile = ref(null);
const practiceMoveReminder = createPracticeMoveReminder();
const practiceHex = ref(''), practiceError = ref(''), manualTile = ref(2);
const humanBoard = ref(null);
const currentExport = shallowRef(null), exportNotice = ref(''), exportCopyFallback = ref(false), exportBusy = ref(false);
const practiceTransition = shallowRef(null);
const layoutStore = createLocalStorageStore({ key: 'human-layout', version: 1, defaultValue: {} });
const layoutPrefs = layoutStore.read();
const settingsStore = createLocalStorageStore({ key: 'human-settings', version: 1, defaultValue: {} });
const alwaysConfirmRestart = ref(!!settingsStore.read().alwaysConfirmRestart);
const playSettings = ref({ swipeSensitivity: 100, showSpeed: false, showFourPercent: false, ...settingsStore.read() });
function updatePlaySettings(value) { playSettings.value = value; alwaysConfirmRestart.value = !!value.alwaysConfirmRestart; settingsStore.update(current => ({ ...current, ...value })); saveAccountPreferences(value); }
function refreshPlaySettings() { playSettings.value = { swipeSensitivity: 100, showSpeed: false, showFourPercent: false, ...settingsStore.read() }; alwaysConfirmRestart.value = !!playSettings.value.alwaysConfirmRestart; refreshAppearance(); }
const inputTimes = shallowRef([]), moveTimes = shallowRef([]), speedClock = ref(Date.now());
const ips = computed(() => countSpeedSamples(inputTimes.value, speedClock.value));
const mps = computed(() => countSpeedSamples(moveTimes.value, speedClock.value));
const fourPercent = computed(() => run.value?.spawnCount ? 100 * (run.value.fourCount || 0) / run.value.spawnCount : 0);
watch(() => run.value?.id, () => { inputTimes.value = []; moveTimes.value = []; });
watch(() => run.value?.seq, (next, old) => { if (next > old && run.value?.id) {
  const time = session.now(); moveTimes.value = addSpeedSample(moveTimes.value, time); speedClock.value = time;
} });
const timingHidden = ref(!!layoutPrefs.timingHidden), rankingHidden = ref(!!layoutPrefs.rankingHidden);
const boardHeight = ref(500);
// 38px tile + 6px gap, with 8px padding and a 1px border on each side.
const visibleNodeRows = computed(() => {
  const defaultCount = Math.max(0, Math.floor((boardHeight.value - 18 + 6) / 44));
  const configured = run.value?.timerSplits || NODE_TILES.map(String);
  return configured.map(expression => {
    const row = timerSplitRow(expression);
    return { ...row, time: run.value?.splitTimes?.[row.key] || (row.depth === 0 ? run.value?.nodes?.[row.tile] : null) };
  }).filter((row, index) => index < defaultCount || row.time);
});
let boardObserver, boardResizeFallback;
watch(humanBoard, board => {
  boardObserver?.disconnect();
  if (boardResizeFallback) window.removeEventListener('resize', boardResizeFallback);
  boardResizeFallback = null;
  if (!board?.$el) return;
  const element = board.$el;
  const measure = () => { boardHeight.value = element.getBoundingClientRect().height; };
  measure();
  if (typeof ResizeObserver === 'function') {
    boardObserver = new ResizeObserver(measure);
    boardObserver.observe(element);
  } else {
    boardResizeFallback = measure;
    window.addEventListener('resize', boardResizeFallback);
  }
}, { flush: 'post' });
watch([timingHidden, rankingHidden], ([timingHidden, rankingHidden]) => layoutStore.write({ timingHidden, rankingHidden }));
onUnmounted(() => {
  boardObserver?.disconnect();
  if (boardResizeFallback) window.removeEventListener('resize', boardResizeFallback);
});
let clockTimer, speedTimer, boardSerial = 0;
const activePolicy = computed(() => policies.value?.variants.find(v => v.id === variant.value));
const high = computed(() => !!session.high());
const elapsed = computed(() => run.value?.reason ? run.value.elapsed : run.value?.firstMoveAt ? Math.max(run.value.elapsed, clock.value - run.value.firstMoveAt) : 0);
const displayBoard = computed(() => practice.value?.board || run.value?.board || Array((VARIANTS[variant.value] || [4,4]).reduce((a,b) => a*b)).fill(0));
const boardDimensions = computed(() => VARIANTS[practice.value?.variant || variant.value]);
const isPracticeOver = computed(() => practice.value && isOver(practice.value.board, practice.value.variant));
const gateTitle = computed(() => ({ loading: '准备棋盘', checking: '正在检查本地进度', network: '需要连接服务器', rejected: '本局无法继续排位', storage: '本地存档不可用', missing: '本地存档缺失', 'other-tab': '此变体正在另一页面进行', paused: '本局已暂停', ended: run.value?.archived ? '本局已归档' : '本局结束' }[gate.value] || '稍候'));
const gateDescription = computed(() => ['network','rejected','storage','missing'].includes(gate.value) ? error.value : gate.value === 'ended' ? `${number(run.value?.score || 0)} 分 · ${run.value?.guest ? '访客练习保留在本地' : run.value?.archived ? '回放已验证并归档' : needsReplayUpload(run.value) ? '回放等待上传' : '回放仅保存在本地'}` : gate.value === 'other-tab' ? '请回到原页面，或关闭原页面后重新检查。其他变体仍可独立游玩。' : gate.value === 'paused' ? '棋盘保持不变，连续计时仍在进行。' : '只验证本地记录，不从服务器加载棋盘。');
const modalTitle = computed(() => ({ restart: '重开确认', 'practice-reminder': '当前是练习板', rules: '对局规则', settings: '设置', 'replay-export': '回放', 'archive-failure': '回放上传失败，请保存回放' }[modal.value] || '提示'));
const number = value => new Intl.NumberFormat(language.value === 'en' ? 'en-US' : 'zh-CN').format(value || 0);
function duration(ms) { const s = Math.floor(Math.max(0, ms) / 1000); return `${Math.floor(s / 3600).toString().padStart(2,'0')}:${Math.floor(s / 60 % 60).toString().padStart(2,'0')}:${(s % 60).toString().padStart(2,'0')}`; }
const date = timestamp => new Date(timestamp * 1000).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
const reasonText = reason => t({ game_over: '自然结束', restarted: '重开', abandoned: '放弃', interrupted: '中断' }[reason] || reason);

let booting = true;
async function boot() {
  bootError.value = '';
  try {
    const previewRequest = ['localhost', '127.0.0.1'].includes(location.hostname)
      ? json('/api/human/local-preview').catch(() => ({})) : Promise.resolve({});
    const [nextPolicies, identity, preview] = await Promise.all([
      json('/api/human/config'), authClient.me(), previewRequest,
    ]);
    policies.value = nextPolicies; user.value = identity.user || null;
    void activateAccountPreferences(user.value?.id);
    if (user.value) {
      try { saveTimerSplits((await json('/api/human/me/settings')).timer_splits); } catch { /* cached/default splits remain usable */ }
    }
    localPreview.value = !!preview.local_preview;
    if (location.pathname.startsWith('/user/') || location.pathname.startsWith('/leaderboard') || location.pathname.startsWith('/analysis')) {
      await route();
    } else {
      await session.activate();
      if (user.value) void live.restore();
      await route();
    }
  } catch { bootError.value = '本地服务未就绪，请确认服务已启动后重试。'; }
  finally { booting = false; }
}
async function localLogin() {
  authBusy.value = true;
  try { const identity = await json('/api/human/local-session', { method: 'POST' }); storeDeviceSession(identity); await handleAuthenticated(identity.user); }
  finally { authBusy.value = false; }
}
function openAuthDialog(mode = 'login') { authDialogMode.value = mode; authDialogOpen.value = true; }
function closeAuthDialog() { authDialogOpen.value = false; }
async function handleAuthenticated(nextUser) {
  user.value = nextUser; closeAuthDialog(); void activateAccountPreferences(nextUser?.id); practice.value = null;
  try { saveTimerSplits((await json('/api/human/me/settings')).timer_splits); } catch { /* retain local settings */ }
  displayCache.clear();
  if (!location.pathname.startsWith('/user/') && !location.pathname.startsWith('/leaderboard') && !location.pathname.startsWith('/analysis')) { await session.activate(); await Promise.all([loadBests(), loadBoard()]); }
}
function handleAccountSaved(nextUser) { if (nextUser) user.value = nextUser; }
async function refreshIdentity() {
  try { const identity = await authClient.me(); user.value = identity.user || null; void activateAccountPreferences(user.value?.id); }
  catch { user.value = null; void activateAccountPreferences(null); }
}
async function logout() {
  await session.waitForMove(); if (busy.value) return;
  if (live.enabled.value) await live.stop();
  await authClient.logout().catch(() => {}); clearDeviceSession(); user.value = null; ownLeader.value = null; await activateAccountPreferences(null); bests.value = {}; practice.value = null;
  displayCache.clear();
  if (location.pathname.startsWith('/user/')) { await goGame(); return; }
  if (location.pathname.startsWith('/leaderboard') || location.pathname.startsWith('/analysis')) { await route(); return; }
  await session.activate(); history.replaceState(null, '', '/#game'); view.value = 'game';
}
async function changeVariant(id) { if (id === variant.value) return; await session.waitForMove(); if (id === variant.value || busy.value) return; practice.value = null; await session.activate(id); }
async function loadBoard(force = false) {
  const serial = ++boardSerial; boardLoading.value = true; boardError.value = '';
  try { const result = await displayJson(`/api/human/leaderboards?variant=${variant.value}&period=${period.value}&limit=10`, { force }); if (serial === boardSerial) { leaders.value = result.entries; ownLeader.value = result.me || null; } }
  catch { if (serial === boardSerial) boardError.value = '榜单暂不可用'; }
  finally { if (serial === boardSerial) boardLoading.value = false; }
}
async function openFullLeaderboard() {
  await session.waitForMove();
  const path = `/leaderboard?type=score&variant=${encodeURIComponent(variant.value)}&period=${encodeURIComponent(period.value)}&page=1`;
  if (`${location.pathname}${location.search}` !== path) history.pushState(null, '', path);
  view.value = 'leaderboard'; modal.value = '';
}
async function loadBests(force = false) {
  const id = user.value?.id; if (!id) return;
  const data = await displayJson('/api/human/me/bests', { force }).catch(() => null);
  if (data && id === user.value?.id) {
    const merged = { ...bests.value };
    for (const [key, value] of Object.entries(data.bests || {})) {
      merged[key] = Math.max(Number(merged[key]) || 0, Number(value) || 0);
    }
    bests.value = merged;
  }
}
async function openAnalysisLibrary() {
  await session.waitForMove();
  if (location.pathname !== '/analysis') history.pushState(null, '', '/analysis');
  view.value = 'analysis'; modal.value = '';
}

async function requestRestart() {
  if (practice.value) { resetPractice(); return; }
  await session.waitForMove();
  if (busy.value) return;
  if (alwaysConfirmRestart.value || (run.value?.score || 0) >= (activePolicy.value?.restart_threshold || 0) || high.value || ['missing','rejected','storage'].includes(gate.value)) modal.value = 'restart';
  else session.restart();
}
async function confirmRestart() { closeModal(); await session.restart(); }
function closeModal() {
  if (modal.value === 'archive-failure' && currentFailure.value) session.dismissArchiveFailure(currentFailure.value.run.id);
  modal.value = '';
}
function setPractice(board, v, originKey = `board:${v}:${board.join(',')}`) {
  practiceTransition.value = null;
  practiceMoveReminder.start(originKey);
  practiceOrigin.value = { board: [...board], variant: v, score: 0, pending: false };
  practice.value = clone(practiceOrigin.value); undoStack.value = []; redoStack.value = []; selectedTile.value = null; practiceError.value = ''; manualTile.value = 2;
}
async function openPractice() { if (!practice.value) { await session.waitForMove(); if (!practice.value) setPractice(run.value?.board || displayBoard.value, variant.value, `game:${run.value?.id || variant.value}:${run.value?.seq ?? 0}`); } }
async function pauseGame() { await session.waitForMove(); if (gate.value === 'ready' && !busy.value) session.pause(); }
function returnToGame() { practice.value = null; selectedTile.value = null; if (high.value && !run.value?.reason) session.retry(); }
function resetPractice() { if (practiceOrigin.value) { practiceTransition.value = null; practice.value = clone(practiceOrigin.value); practiceMoveReminder.reset(); undoStack.value = []; redoStack.value = []; selectedTile.value = null; } }
function rememberPractice() { undoStack.value.push({ state: clone(practice.value), moves: practiceMoveReminder.moves }); if (undoStack.value.length > 1000) undoStack.value.shift(); redoStack.value = []; }
function practiceUndo() { if (!undoStack.value.length) return; practiceTransition.value = null; redoStack.value.push({ state: clone(practice.value), moves: practiceMoveReminder.moves }); const previous = undoStack.value.pop(); practice.value = previous.state; practiceMoveReminder.restore(previous.moves); }
function practiceRedo() { if (!redoStack.value.length) return; practiceTransition.value = null; undoStack.value.push({ state: clone(practice.value), moves: practiceMoveReminder.moves }); const next = redoStack.value.pop(); practice.value = next.state; practiceMoveReminder.restore(next.moves); }
function clearPractice() { rememberPractice(); practiceTransition.value = null; practice.value = { ...practice.value, board: practice.value.board.map(() => 0), score: 0, pending: false }; }
function togglePalette(value) { selectedTile.value = selectedTile.value === value ? null : value; }
function focusPracticeBoard() { nextTick(() => humanBoard.value?.$el?.focus()); }
function setPracticeBoard() {
  if (!practice.value) return;
  const board = parsePracticeHex(practiceHex.value, practice.value.board.length);
  if (!board) { practiceError.value = `请输入不超过 ${practice.value.board.length} 位、仅含 0–9 与 a–h 的局面编码。`; return; }
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
  if (!practice.value) {
    const time = session.now(); inputTimes.value = addSpeedSample(inputTimes.value, time); speedClock.value = time;
    session.play(direction); return;
  }
  if (practice.value.pending) return;
  const moved = move(practice.value.board, ...VARIANTS[practice.value.variant], direction);
  if (!moved.changed) return;
  rememberPractice(); if (!manualSpawn.value) randomSpawn(moved.board);
  practiceTransition.value = { fromBoard: practice.value.board, toBoard: moved.board, direction };
  practice.value = { ...practice.value, board: moved.board, score: practice.value.score + moved.score, pending: manualSpawn.value };
  if (practiceMoveReminder.moved()) modal.value = 'practice-reminder';
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
async function openReplayExport() {
  await session.waitForMove();
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
function openPlayer(name) {
  const path = `/user/${encodeURIComponent(name)}`;
  if (`${location.pathname}${location.search}` !== path) history.pushState(null, '', path);
  profileName.value = name; view.value = 'profile'; modal.value = '';
}
async function openReplay(item) {
  const id = typeof item === 'string' ? item : item?.id || item?.run_id;
  if (!id) return;
  const url = new URL('/verse-replay/', location.href);
  url.searchParams.set('human-run', id);
  url.searchParams.set('lang', language.value);
  window.open(url.href, '_blank', 'noopener');
}
function openAnalysis(item) {
  if (!user.value) { openAuthDialog('login'); return; }
  analysisRunId.value = typeof item === 'string' ? item : item?.id || item?.run_id || '';
}
function openLocalReplay() {
  if (!run.value) return;
  try {
    const replay = exportCurrentReplay(run.value, session.getEvents());
    void openReplayViewer(replay, language.value).catch(() => {
      currentExport.value = replay;
      exportNotice.value = '回放页面未能接收记录，请下载文件后在回放站打开。';
      modal.value = 'replay-export';
    });
  } catch {
    exportNotice.value = '无法读取当前回放，请重新进入本局后重试。';
    modal.value = 'replay-export';
  }
}
async function route() {
  const path = location.hash.slice(1) || 'game';
  if (location.pathname.startsWith('/user/')) {
    try { profileName.value = decodeURIComponent(location.pathname.slice(6).replace(/\/$/, '')); view.value = 'profile'; }
    catch { view.value = 'game'; }
  } else if (location.pathname === '/leaderboard' || location.pathname === '/leaderboard/') {
    view.value = 'leaderboard';
  } else if (location.pathname === '/analysis' || location.pathname === '/analysis/') {
    view.value = 'analysis';
  } else if (path.startsWith('replay/')) {
    const id = decodeURIComponent(path.slice(7).split('?')[0]);
    const target = new URL('/verse-replay/', location.href);
    target.searchParams.set('human-run', id); target.searchParams.set('lang', language.value);
    location.replace(target.href); return;
  }
  else await showGame();
}
async function showGame() {
  view.value = 'game'; practice.value = null; selectedTile.value = null;
  if (!run.value) await session.activate();
  else if (high.value && !run.value.reason) await session.retry();
  if (user.value) void live.restore();
  void loadBoard(); void loadBests();
}
async function goGame() {
  if (location.pathname !== '/' || location.hash !== '#game') history.pushState(null, '', '/#game');
  await showGame();
}
watch([variant, period], () => { if (!booting) loadBoard(); });
watch(() => run.value?.seq, () => live.publishTail());
watch(() => [run.value?.variant, run.value?.score], ([key, score]) => {
  const value = Math.max(0, Number(score) || 0);
  if (key && value > (Number(bests.value[key]) || 0)) bests.value = { ...bests.value, [key]: value };
});
watch(currentBest, value => live.updateBest(value));
watch(() => run.value?.id, (id, oldId) => { if (oldId && id && id !== oldId) void live.runChanged(); });
watch(() => run.value?.reason, (reason, oldReason) => { if (reason && !oldReason) live.finish(); });
watch(() => [run.value?.id, gate.value], ([id, currentGate]) => terminalOverlay.update(id, currentGate === 'ended'), { immediate: true });
watch(() => [run.value?.id, run.value?.archived], ([id, archived], [oldId, oldArchived]) => { if (archived && id === oldId && !oldArchived) { loadBoard(true); loadBests(true); } });
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
watch(modal, async value => { if (['restart', 'practice-restart', 'practice-reminder'].includes(value)) { await nextTick(); safeButton.value?.focus(); } });
watch(() => practice.value?.board, board => { practiceHex.value = board ? practiceBoardHex(board) : ''; practiceError.value = ''; });
watch(manualSpawn, enabled => {
  if (!enabled && practice.value?.pending) {
    const board = [...practice.value.board]; const spawn = randomSpawn(board);
    practiceTransition.value = spawn ? { fromBoard: practice.value.board, toBoard: board, spawn: spawn.index } : null;
    practice.value = { ...practice.value, board, pending: false };
  }
});
function refreshVisibleAppearance() { if (!document.hidden) refreshAppearance(); }
function leavePage() { session.stop(); }
function restorePage(event) {
  if (!event.persisted) return;
  session.start(); refreshAppearance();
  if (run.value) void session.activate(variant.value);
}
let routeQueued = false;
function scheduleRoute() {
  if (routeQueued) return;
  routeQueued = true; queueMicrotask(() => { routeQueued = false; void route(); });
}
onMounted(() => {
  session.start();
  clockTimer = setInterval(() => { clock.value = session.now(); }, 500);
  speedTimer = setInterval(() => {
    if (view.value === 'game' && !practice.value && playSettings.value.showSpeed) speedClock.value = session.now();
  }, SPEED_REFRESH_MS);
  window.addEventListener('keydown', keydown); window.addEventListener('hashchange', scheduleRoute); window.addEventListener('popstate', scheduleRoute); window.addEventListener('pagehide', leavePage); window.addEventListener('pageshow', restorePage); window.addEventListener('focus', refreshAppearance); window.addEventListener('storage', refreshAppearance); window.addEventListener('human-preferences-changed', refreshAppearance); window.addEventListener('account-preferences-changed', refreshPlaySettings); document.addEventListener('visibilitychange', refreshVisibleAppearance); boot();
});
onUnmounted(() => {
  terminalOverlay.dispose(); live.dispose(); session.stop(); clearInterval(clockTimer); clearInterval(speedTimer); window.removeEventListener('keydown', keydown); window.removeEventListener('hashchange', scheduleRoute); window.removeEventListener('popstate', scheduleRoute); window.removeEventListener('pagehide', leavePage); window.removeEventListener('pageshow', restorePage); window.removeEventListener('focus', refreshAppearance); window.removeEventListener('storage', refreshAppearance); window.removeEventListener('human-preferences-changed', refreshAppearance); window.removeEventListener('account-preferences-changed', refreshPlaySettings); document.removeEventListener('visibilitychange', refreshVisibleAppearance);
});
</script>
