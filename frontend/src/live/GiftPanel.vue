<template>
  <section ref="panel" class="gift-panel">
    <div class="gift-bar">
      <div class="gift-leading"><slot name="leading" /></div>
      <div class="common-gifts">
        <GiftChoice v-for="gift in commonGifts" :key="gift.id" :gift="gift" :lang="lang" :disabled="sendDisabled" @send="prepareGiftSend" />
      </div>
      <div class="gift-bar-tools">
        <button class="expand-gifts" :aria-expanded="expanded" :title="t('全部礼物','All gifts')" @click="helpOpen=false;expanded=!expanded"><ChevronUp :size="20" :class="{ rotated:expanded }" /></button>
        <button :title="t('我的礼物与设置','My gifts and settings')" @click="openSettings"><Settings2 :size="17" /></button>
      </div>
    </div>
    <section v-if="expanded" class="gift-drawer" :aria-label="t('全部礼物','All gifts')" @keydown.esc="expanded=false">
      <header><Gift :size="18" /><h2>{{ t('全部礼物','All gifts') }}</h2><button @click="expanded=false" :aria-label="t('关闭','Close')"><X :size="17" /></button></header>
      <div class="gift-grid"><button class="red-gift-choice" @click="expanded=false;emit('red-envelope')"><img src="/live-gifts/red-envelope.webp" alt="" /><b>{{ t('红包','Red envelope') }}</b></button><GiftChoice v-for="gift in catalog.gifts" :key="gift.id" :gift="gift" :lang="lang" :disabled="sendDisabled" @send="prepareGiftSend" /></div>
    </section>
    <section v-if="helpOpen" class="gift-drawer contribution-drawer" :aria-label="t('贡献度说明','How contribution works')">
      <header><CircleHelp :size="18" /><h2>{{ t('贡献度说明','How contribution works') }}</h2><button ref="helpClose" @click="helpOpen=false" :aria-label="t('关闭','Close')"><X :size="17" /></button></header>
      <p>{{ t('本场直播中，你的支持都会计入贡献度。','Your support adds to your contribution this stream.') }}</p>
      <dl>
        <div><dt>{{ t('送礼','Gifts') }}</dt><dd>{{ t('每消费 1 Token，+1','+1 per Token spent') }}</dd></div>
        <div><dt>{{ t('观看','Watching') }}</dt><dd>{{ t('每满 10 分钟，+1','+1 per 10 minutes') }}</dd></div>
        <div><dt>{{ t('点赞','Likes') }}</dt><dd>{{ t('每次 +1，最多 +10','+1 each, up to 10') }}</dd></div>
        <div><dt>{{ t('发言','Chat') }}</dt><dd>{{ t('每条 +1，最多 +10','+1 per message, up to 10') }}</dd></div>
      </dl>
      <p class="contribution-note">{{ t('按贡献度排序，荣耀赞助者优先展示。','Ranked by contribution, with Gold Supporters shown first.') }}</p>
    </section>
    <p v-if="status" class="gift-note" role="status">{{ status }} <button v-if="uncertain && !busy" @click="processQueue">{{ t('核对并重试', 'Check and retry') }}</button></p>

    <dialog ref="confirmDialog" @cancel="confirmation = null" class="gift-dialog">
      <form @submit.prevent="confirmSend"><header><h2>{{ t('确认送出', 'Confirm gift') }}</h2><button type="button" @click="closeConfirm" :aria-label="t('关闭', 'Close')"><X :size="18" /></button></header>
        <p>{{ confirmation?.name }} ×{{ confirmation?.request.quantity }}</p><strong class="confirm-total">{{ tokens(confirmation?.request.expected_cost_units) }} Token</strong>
        <p>{{ t('送出后将扣除额度，不影响直播对局。', 'Tokens will be deducted. The live game will not be affected.') }}</p>
        <button type="submit" class="send-gift">{{ t('确认送出', 'Confirm and send') }}</button>
      </form>
    </dialog>
    <dialog ref="quickDialog" class="gift-dialog"><form @submit.prevent="enableQuick"><header><h2>{{ t('本次连送预算', 'Quick-send budget') }}</h2><button type="button" @click="quickDialog.close()" :aria-label="t('关闭', 'Close')"><X :size="18" /></button></header><label>Token <input type="number" v-model="budgetInput" min="1" max="100000" step="1" required /></label><p>{{ t('每次点击均会扣费，达到预算后停止。', 'Each click spends Tokens. Sending stops at this budget.') }}</p><button class="send-gift" type="submit">{{ t('开启连送', 'Enable quick send') }}</button></form></dialog>
    <dialog ref="settingsDialog" class="gift-dialog settings-dialog"><header><h2>{{ t('我的礼物', 'My gifts') }}</h2><button @click="settingsDialog.close()" :aria-label="t('关闭', 'Close')"><X :size="18" /></button></header>
      <form @submit.prevent="saveSettings"><label>{{ t('每日送礼上限（Token）', 'Daily gift budget (Tokens)') }}<input type="number" min="0" max="100000000" step="0.001" v-model="dailyInput" :placeholder="t('不限', 'No limit')" /></label><label class="check-label"><input type="checkbox" v-model="entranceEnabled" />{{ t('赞助者入场提示', 'Supporter entrance announcement') }}</label><button type="submit" :disabled="saving">{{ t('保存', 'Save') }}</button></form>
      <div class="quick-row"><label><input type="checkbox" :checked="quick" :disabled="busy || uncertain" @change="toggleQuick" />{{ t('大额连送免逐笔确认','Quick send without per-gift confirmation') }}</label><span v-if="quick">{{ t('剩余预算','Budget left') }} {{ tokens(Math.max(0,quickBudget-quickSpent)) }} Token</span></div>
      <p>{{ t('今日已送出', 'Spent today') }} {{ tokens(account?.spent_units) }} Token</p><p v-if="settingsStatus" role="status">{{ settingsStatus }}</p>
      <div class="orders"><div v-for="order in account?.orders || []" :key="order.request_id"><GiftIcon :id="order.gift_id" :size="20" /><span>{{ name(order.gift_id) }} ×{{ order.quantity }}<small>{{ new Date(order.created_at * 1000).toLocaleString() }}</small></span><b>{{ tokens(order.cost_units) }}</b></div></div>
    </dialog>
    <dialog ref="balanceDialog" class="gift-dialog"><header><h2>{{ t('Token 余额不足', 'Not enough Tokens') }}</h2><button @click="balanceDialog.close()" :aria-label="t('关闭', 'Close')"><X :size="18" /></button></header><p>{{ t('可以赞助充值，或等待周度额度刷新。', 'Sponsor to top up, or wait for your weekly allowance.') }}</p><div class="balance-actions"><button @click="balanceDialog.close(); sponsorOpen = true">{{ t('赞助充值', 'Sponsor') }}</button><button @click="balanceDialog.close(); guideOpen = true">{{ t('额度说明', 'Token guide') }}</button></div></dialog>
    <SponsorDialog :open="sponsorOpen" :user="user" @close="sponsorOpen = false" /><QuotaGuideDialog :open="guideOpen" @close="guideOpen = false" />
  </section>
</template>
<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue';
import { Gift, Settings2, ChevronUp, X, CircleHelp } from '@lucide/vue';
import GiftIcon from './GiftIcon.vue';
import GiftChoice from './GiftChoice.vue';
import { giftPrice } from './giftPrice.js';
import SponsorDialog from '../features/billing/SponsorDialog.vue';
import QuotaGuideDialog from '../features/billing/QuotaGuideDialog.vue';
import { giftApi, sendGift } from './giftApi.js';
const props = defineProps({ user: Object, online: Boolean, lang: String });
const emit = defineEmits(['login', 'catalog', 'red-envelope']);
const t = (zh,en) => props.lang === 'zh' ? zh : en;
const expanded=ref(false);
const helpOpen=ref(false), helpClose=ref(null);
async function showContributionHelp(){expanded.value=false;helpOpen.value=true;await nextTick();helpClose.value?.closest('section')?.scrollIntoView({block:'nearest'});helpClose.value?.focus({preventScroll:true});}
const panel=ref(null);
function dismiss(event){
  if(event.type==='keydown'){if(event.key==='Escape'){expanded.value=false;helpOpen.value=false;}return;}
  if(!panel.value?.contains(event.target) && !event.target.closest('.live-gift-popover')){expanded.value=false;helpOpen.value=false;}
}
const commonIds=['two','four','heart','flowers','moai','button','tea','whale','rip'];
const commonGifts=computed(()=>commonIds.map(id=>catalog.value.gifts.find(g=>g.id===id)).filter(Boolean));
const sendDisabled=computed(()=>queue.value.length>=5 || uncertain.value);
const catalog = ref({ gifts: [] }), account = ref(null), selectedId = ref('two'), quantity = ref(1);
const selected = computed(() => catalog.value.gifts.find(item => item.id === selectedId.value));
const cost = computed(() => giftPrice(selected.value,quantity.value));
const name = id => catalog.value.gifts.find(item => item.id === id)?.[props.lang] || id;
const tokens = value => ((value || 0) / 1000).toLocaleString(props.lang === 'zh' ? 'zh-CN' : 'en-US', { maximumFractionDigits: 3 });
const status = ref(''), settingsStatus = ref(''), busy = ref(false), saving = ref(false), uncertain = ref(false), queue = ref([]);
const quick = ref(false), quickBudget = ref(0), quickSpent = ref(0), budgetInput = ref(128);
const confirmation = ref(null), confirmDialog = ref(null), quickDialog = ref(null), settingsDialog = ref(null), balanceDialog = ref(null);
const dailyInput = ref(''), entranceEnabled = ref(true), sponsorOpen = ref(false), guideOpen = ref(false);
let generation = 0;
watch(() => props.lang, () => { status.value = ''; settingsStatus.value = ''; });
function storageKey() { return `live:gift-pending:${props.user?.id}`; }
function persist() { try { sessionStorage.setItem(storageKey(), JSON.stringify(queue.value)); } catch {} }
async function loadCatalog() { catalog.value = await giftApi('catalog'); emit('catalog', catalog.value.gifts); }
async function loadAccount() { const id = props.user?.id; if (!id) return; const data = await giftApi('me'); if (props.user?.id === id) account.value = data; }
watch(() => props.user?.id, async id => {
  generation++; quick.value = false; queue.value = []; uncertain.value = false; busy.value = false; account.value = null; status.value = '';
  for (const dialog of [confirmDialog, quickDialog, settingsDialog, balanceDialog]) dialog.value?.close();
  if (!id) return;
  try { const saved = JSON.parse(sessionStorage.getItem(storageKey()) || '[]'); if (Array.isArray(saved)) queue.value = saved.slice(0,5); } catch {}
  try { await loadAccount(); } catch { status.value = t('暂时无法读取额度，请稍后重试。', 'Could not load your balance. Try again shortly.'); }
  if (props.user?.id === id && queue.value.length) { uncertain.value = true; status.value = t('有一笔送礼等待核对，请核对并重试。', 'A gift needs confirmation. Check and retry.'); }
}, { immediate: true });
function makeRequest() { return { request_id: crypto.randomUUID(), gift_id: selectedId.value, quantity: quantity.value, expected_cost_units: cost.value, quote_version: catalog.value.version }; }
function prepareGiftSend(id,n) {
  if(!props.user){emit('login');return;}
  if(!props.online){status.value=t('主播离线期间暂停送礼','Gifts are paused while the broadcaster is offline');return;}
  selectedId.value=id;quantity.value=n;prepareSend();
}
function prepareSend() {
  if (!selected.value || !props.online || uncertain.value || cost.value === null) return;
  const request = makeRequest();
  if (quick.value && quickSpent.value + request.expected_cost_units > quickBudget.value) { status.value = t('本次连送预算不足。', 'Quick-send budget reached.'); return; }
  if (request.expected_cost_units >= 2048000 && !quick.value) { confirmation.value = { request, name: name(request.gift_id) }; confirmDialog.value.showModal(); }
  else enqueue(request);
}
function closeConfirm() { confirmation.value = null; confirmDialog.value.close(); }
function confirmSend() { const request = confirmation.value?.request; closeConfirm(); if (request) enqueue(request); }
function enqueue(request) {
  if (queue.value.length >= 5 || uncertain.value || !props.user || !props.online) return;
  if (quick.value && quickSpent.value + request.expected_cost_units > quickBudget.value) { status.value = t('本次连送预算不足。', 'Quick-send budget reached.'); return; }
  const reserved = quick.value ? request.expected_cost_units : 0;
  quickSpent.value += reserved;
  queue.value.push({ request, reserved }); persist(); processQueue();
}
async function processQueue() {
  if (busy.value || !queue.value.length) return;
  const epoch = generation;
  const recovering = uncertain.value;
  busy.value = true; uncertain.value = false;
  try {
    while (queue.value.length && epoch === generation) {
      const item = queue.value[0]; status.value = t('发送中…', 'Sending…');
      try {
        const data = await sendGift(item.request, recovering);
        if (epoch !== generation) return;
        if (account.value) account.value.token_balance = data.token_balance;
        queue.value.shift(); persist();
        status.value = `${t('已送出', 'Sent')} ${name(data.gift_id)} ×${data.combo_count}`;
      } catch (error) {
        if (epoch !== generation) return;
        // Uncertain delivery retains the original id and reserves the full session budget.
        if (!error.status || error.status >= 500) { uncertain.value = true; status.value = t('暂未确认送达，请核对并重试；不会重复扣费。', 'Delivery unconfirmed. Check and retry; you will not be charged twice.'); break; }
        quickSpent.value = Math.max(0, quickSpent.value - queue.value.reduce((sum,item) => sum + item.reserved, 0));
        queue.value = []; persist();
        if (error.status === 402) balanceDialog.value.showModal();
        const messages = {
          gift_price_changed: ['价格已更新，请确认新价格后重新送出。', 'Prices changed. Review the new price before sending.'],
          gift_daily_budget: ['已达到每日送礼上限。', 'Daily gift budget reached.'],
          stream_offline: ['主播已离线，本次未送出。', 'The broadcaster is offline. Gift not sent.'],
        };
        status.value = error.status === 429 ? t('送得太快了，请稍后再送。', 'Please slow down before sending again.') : error.status === 401 ? t('请重新登录后送礼。', 'Please sign in again.') : t(...(messages[error.detail] || ['礼物未送出，请稍后重试。', 'Gift not sent. Please try again.']));
        if (error.detail === 'gift_price_changed') await loadCatalog();
        break;
      }
    }
    await loadAccount().catch(() => {});
  } finally { if (epoch === generation) { busy.value = false; if(queue.value.length && !uncertain.value) processQueue(); } }
}
function toggleQuick(event) { event.target.checked = quick.value; if (quick.value) quick.value = false; else quickDialog.value.showModal(); }
function enableQuick() { const budget = Math.round(Number(budgetInput.value) * 1000); if (budget <= 0 || !Number.isFinite(budget)) return; quickBudget.value = budget; quickSpent.value = 0; quick.value = true; quickDialog.value.close(); }
async function openSettings() {
  if (!props.user) { emit('login'); return; }
  try { await loadAccount(); dailyInput.value = account.value.daily_limit_units == null ? '' : account.value.daily_limit_units / 1000; entranceEnabled.value = account.value.entrance_enabled; settingsStatus.value = ''; settingsDialog.value.showModal(); }
  catch { status.value = t('读取失败，请重试。', 'Could not load settings.'); }
}
async function saveSettings() {
  saving.value = true;
  try { account.value = await giftApi('preferences', { daily_limit_units: dailyInput.value === '' ? null : Math.round(Number(dailyInput.value) * 1000), entrance_enabled: entranceEnabled.value }); settingsStatus.value = t('已保存', 'Saved'); }
  catch { settingsStatus.value = t('保存失败，请重试。', 'Could not save. Please retry.'); }
  finally { saving.value = false; }
}
onMounted(() => {document.addEventListener('pointerdown',dismiss);document.addEventListener('keydown',dismiss);loadCatalog().catch(() => { status.value = t('礼物暂时无法加载。', 'Gifts are temporarily unavailable.'); });});
onUnmounted(() => { generation++;document.removeEventListener('pointerdown',dismiss);document.removeEventListener('keydown',dismiss); });
defineExpose({ refreshBalance: () => loadAccount().catch(() => {}), showContributionHelp });
</script>
<style scoped>
.gift-panel { position:relative;border-top:1px solid var(--border-main);margin-top:14px;padding-top:6px;z-index:45; }
header { display:flex;align-items:center;gap:8px;margin-bottom:12px; }h2 { font-size:15px;margin:0; }header > button { margin-left:auto; }
button { display:inline-flex;align-items:center;justify-content:center;gap:6px;background:var(--bg-card);color:var(--text-main);border:1px solid var(--border-main);border-radius:5px;padding:7px 10px;min-height:30px;cursor:pointer; }button:disabled { opacity:.5;cursor:default; }
.gift-bar { display:flex;gap:8px;align-items:stretch; }.gift-leading { flex:0 0 78px;width:78px; }.common-gifts { flex:1;min-width:0;display:flex;justify-content:flex-end;gap:8px;overflow-x:auto; }.common-gifts :deep(.gift-choice) { flex:0 0 70px; }
.gift-bar-tools { width:100px;flex-shrink:0;display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:5px;align-content:center; }.gift-bar-tools > span { width:100%;text-align:center;font-size:11px;overflow-wrap:anywhere; }.gift-bar-tools small { color:var(--text-secondary);font-size:10px; }.expand-gifts { height:42px; }.rotated { transform:rotate(180deg); }
.gift-drawer { position:absolute;bottom:calc(100% + 6px);right:0;width:370px;max-width:100%;background:var(--bg-main);border:1px solid var(--border-main);border-radius:8px;box-shadow:0 8px 30px #0005;padding:12px; }
.contribution-drawer { padding:16px; }.contribution-drawer p { font-size:12px;line-height:1.6;color:var(--text-secondary);margin:12px 0; }.contribution-drawer dl { margin:0; }.contribution-drawer dl>div { display:flex;justify-content:space-between;gap:16px;padding:10px 0;border-bottom:1px solid var(--border-main);font-size:12px; }.contribution-drawer dt { font-weight:700; }.contribution-drawer dd { margin:0;text-align:right; }.contribution-drawer .contribution-note { margin-bottom:0;font-size:11px; }
.gift-grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;max-height:min(540px,60vh);overflow:auto;overscroll-behavior:contain; }
.gift-grid .red-gift-choice { flex-direction:column;min-height:95px;background:transparent;border:1px solid transparent;font-size:12px; }.red-gift-choice img { width:54px;height:54px;object-fit:contain; }.gift-grid .red-gift-choice:hover { border-color:#d96851;box-shadow:0 3px 12px #0004; }
.send-gift { background:var(--accent);color:var(--text-on-accent,#10202f);font-weight:700; }
.quick-row { display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:12px; }.quick-row label,.check-label { display:flex;gap:6px;align-items:center; }
.gift-note { margin:4px 0;font-size:11px;line-height:20px;color:var(--text-secondary);max-width:100%;overflow-wrap:anywhere; }
.gift-dialog { position:fixed;inset:0;margin:auto;background:var(--bg-main);color:var(--text-main);border:1px solid var(--border-main);border-radius:8px;padding:22px;width:min(430px,calc(100vw - 28px));max-height:85vh;overflow:auto; }.gift-dialog::backdrop { background:#0009; }.gift-dialog p { font-size:13px;line-height:1.6; }.gift-dialog label { font-size:13px; }.gift-dialog input[type=number] { display:block;width:100%;margin:8px 0;padding:8px;color:var(--text-main);background:var(--bg-input);border:1px solid var(--border-main);border-radius:4px; }.gift-dialog form { display:grid;gap:12px; }.confirm-total { font-size:24px;color:var(--accent); }.balance-actions { display:flex;gap:10px; }
.orders { max-height:280px;overflow:auto; }.orders > div { display:flex;align-items:center;gap:8px;padding:8px 0;border-top:1px solid var(--border-main);font-size:12px; }.orders > div > span:not(.gift-icon) { flex:1; }.orders small { display:block;font-size:10px;color:var(--text-secondary);margin-top:3px; }
:deep(.sponsor-overlay),:deep(.quota-guide-overlay) { position:fixed; }
:deep(.sponsor-panel),:deep(.quota-guide-panel) { background:var(--bg-main); }
@media(max-width:600px) {
  :deep(.sponsor-panel > .grid) { grid-template-columns:minmax(0,1fr); }
  :deep(.sponsor-qr) { width:160px; }
  :deep(.sponsor-close) { min-width:64px;flex-shrink:0; }
  :deep(.weekly-tier-grid) { grid-template-columns:minmax(0,1fr); }
  :deep(.section-heading) { flex-direction:column;align-items:flex-start;gap:6px; }
}
</style>
