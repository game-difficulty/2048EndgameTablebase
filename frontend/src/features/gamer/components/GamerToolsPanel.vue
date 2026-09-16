<template>
  <nav class="gamer-tools" :aria-label="t('对局工具','Game tools')">
    <button v-for="tool in tools" :key="tool.id" type="button" :title="tool.label" @click="open(tool.id)"><component :is="tool.icon" :size="18" /><span>{{ tool.label }}</span></button>
  </nav>
  <Teleport to="body">
    <dialog ref="dialog" class="gamer-tool-dialog" @close="closed" @cancel="closed" @click.self="close" @keydown.stop>
      <header><h2>{{ tools.find(tool => tool.id === view)?.label }}</h2><button type="button" :title="t('关闭','Close')" :aria-label="t('关闭','Close')" @click="close"><X :size="20" /></button></header>
      <section v-if="view === 'settings'" class="gamer-tool-body">
        <GamerMatchOptionsPanel :options="options" :table-enabled="tableEnabled" @table-mode="emit('table-mode',$event)" @change="(key,value) => emit('change',key,value)" />
        <div class="table-heading"><h3>{{ t('可用定式','Allowed tables') }}</h3><span>{{ selectedCount }}/{{ tables.length }}</span><button @click="emit('table-selection',tables.map(table => table.fullPattern))">{{ t('全选','All') }}</button><button @click="emit('table-selection',[])">{{ t('全不选','None') }}</button></div>
        <input v-model="filter" class="table-filter" type="search" :placeholder="t('筛选定式','Filter tables')" :aria-label="t('筛选定式','Filter tables')" />
        <p v-if="loading" class="tool-note">{{ t('正在读取定式…','Loading tables…') }}</p>
        <p v-else-if="error" class="tool-note">{{ t('暂时无法读取定式','Could not load tables') }}<button @click="emit('load-tables')"><RefreshCw :size="14" />{{ t('重试','Retry') }}</button></p>
        <div v-else class="gamer-table-list">
          <label v-for="table in filteredTables" :key="table.fullPattern"><input type="checkbox" :checked="selected(table.fullPattern)" @change="select(table.fullPattern,$event.target.checked)" /><span>{{ table.fullPattern }}</span><small>{{ t('出4','4 rate') }} {{ Math.round(table.spawnRate*100) }}%</small></label>
          <p v-if="!filteredTables.length" class="tool-note">{{ t('没有匹配的定式','No matching tables') }}</p>
        </div>
      </section>
      <section v-else-if="view === 'replay'" class="gamer-tool-body">
        <p class="replay-count">{{ t('已记录','Recorded') }} <strong>{{ statistics.moves.toLocaleString() }}</strong> {{ t('步','moves') }}</p>
        <p v-if="statistics.partial" class="tool-note">{{ partialNote }}</p>
        <div class="replay-actions"><button @click="download"><Download :size="18" />{{ t('下载回放','Download') }}</button><button @click="copy"><Copy :size="18" />{{ t('复制回放','Copy replay') }}</button></div>
        <a class="replay-link" href="https://2048tables.online/verse-replay/" target="_blank" rel="noopener noreferrer">{{ t('打开回放站','Open replay viewer') }}<ExternalLink :size="14" /></a>
        <textarea v-if="fallbackText" ref="fallback" :value="fallbackText" readonly rows="4" :aria-label="t('回放内容','Replay text')" />
        <p v-if="status" class="tool-note" role="status">{{ status }}</p>
      </section>
      <section v-else-if="view === 'stats'" class="gamer-tool-body">
        <dl class="gamer-game-statistics"><div><dt>{{ t('总步数','Total moves') }}</dt><dd>{{ statistics.moves.toLocaleString() }}</dd></div><div><dt>{{ t('出4步数','Moves spawning 4') }}</dt><dd>{{ statistics.fours.toLocaleString() }}</dd></div><div><dt>{{ t('实际出4率','Observed 4 rate') }}</dt><dd>{{ statistics.rate == null ? '—' : (statistics.rate*100).toFixed(2)+'%' }}</dd></div></dl>
        <p class="tool-note">{{ t('按当前保留的走法统计，不含开局棋块。','Current branch only; starting tiles are excluded.') }}</p>
        <p v-if="statistics.partial" class="tool-note">{{ partialNote }}</p>
      </section>
    </dialog>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Settings2, Clapperboard, ChartNoAxesColumnIncreasing, X, Copy, Download, ExternalLink, RefreshCw } from '@lucide/vue';
import { tableAllowed } from '../engine/tableSelection.js';
import GamerMatchOptionsPanel from './GamerMatchOptionsPanel.vue';
import { downloadText } from '../../../services/files/browserFiles.js';
const props = defineProps({active:Boolean,options:Array,tableEnabled:Boolean,tables:Array,selection:Array,loading:Boolean,error:Boolean,statistics:Object,getReplay:Function});
const emit=defineEmits(['table-mode','change','table-selection','load-tables','dialog-open']);
const {locale}=useI18n();
const t=(zh,en)=>locale.value.startsWith('zh')?zh:en;
const tools=computed(()=>[{id:'settings',icon:Settings2,label:t('设置','Settings')},{id:'replay',icon:Clapperboard,label:t('回放','Replay')},{id:'stats',icon:ChartNoAxesColumnIncreasing,label:t('统计','Stats')}]);
const partialNote=computed(()=>t('旧存档缺少此前的记录，回放与统计从恢复后的局面起算。','Earlier moves are missing from this save. Replay and statistics start at the restored board.'));
const dialog=ref(null),view=ref(''),filter=ref(''),status=ref(''),fallbackText=ref(''),fallback=ref(null);
const selected=name=>tableAllowed({fullPattern:name},props.selection);
const selectedCount=computed(()=>props.tables.filter(table=>selected(table.fullPattern)).length);
const filteredTables=computed(()=>props.tables.filter(table=>table.fullPattern.toLowerCase().includes(filter.value.trim().toLowerCase())).slice().sort((a,b)=>a.fullPattern.localeCompare(b.fullPattern,undefined,{numeric:true})));
function select(name,enabled){const names=new Set(props.selection ?? props.tables.filter(table=>selected(table.fullPattern)).map(table=>table.fullPattern));if(enabled)names.add(name);else names.delete(name);emit('table-selection',[...names]);}
function open(id){view.value=id;status.value='';fallbackText.value='';emit('dialog-open',true);dialog.value.showModal();if(id==='settings')emit('load-tables');}
function closed(){emit('dialog-open',false);}
function close(){dialog.value?.close();closed();}
function exportError(error){status.value=error?.message==='replay_too_large' ? t('回放已超过 500 KB，无法导出。','Replay exceeds the 500 KB limit.') : t('回放暂时无法导出，请重试。','Could not export the replay. Please retry.');}
function download(){status.value='';try{const data=props.getReplay();downloadText(data.text,data.filename);status.value=t('已导出当前对局','Current game exported');}catch(e){exportError(e);}}
async function copy(){
  status.value='';let text;
  try{text=props.getReplay().text;}catch(e){exportError(e);return;}
  try{await navigator.clipboard.writeText(text);status.value=t('回放已复制','Replay copied');fallbackText.value='';}
  catch{fallbackText.value=text;status.value=t('无法自动复制，请复制下方已选中的内容。','Automatic copy is unavailable. Copy the selected text below.');await nextTick();fallback.value?.focus();fallback.value?.select();}
}
watch(()=>props.active,value=>{if(!value)close();});
onUnmounted(()=>{emit('dialog-open',false);});
</script>

<style scoped>
.gamer-tools{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;flex-shrink:0}.gamer-tools button{display:flex;align-items:center;justify-content:center;gap:6px;min-width:0;min-height:42px;border:1px solid var(--border-main);border-radius:6px;background:var(--bg-card);color:var(--text-main);font-size:var(--font-ui-xs);font-weight:700;cursor:pointer}.gamer-tools button:hover{border-color:var(--accent);color:var(--accent)}.gamer-tools svg{flex-shrink:0}
.gamer-tool-dialog{position:fixed;inset:0;margin:auto;width:min(540px,calc(100vw - 32px));max-height:calc(100dvh - 40px);overflow:auto;padding:0;border:1px solid var(--border-main);border-radius:8px;background:var(--bg-main);color:var(--text-main);box-shadow:0 20px 60px #0005;font-size:14px;letter-spacing:0}.gamer-tool-dialog::backdrop{background:#0009}.gamer-tool-dialog header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 20px;border-bottom:1px solid var(--border-main)}.gamer-tool-dialog h2{margin:0;font-size:20px;font-weight:700}.gamer-tool-dialog button{cursor:pointer;color:var(--text-main)}.gamer-tool-dialog header button{display:grid;place-items:center;padding:5px;background:transparent;border:0}.gamer-tool-body{padding:20px}.gamer-tool-body :deep(.gamer-match-options){padding:0;min-height:0;border:0;background:transparent;box-shadow:none;margin-bottom:20px}.table-heading{display:flex;align-items:center;gap:8px;border-top:1px solid var(--border-main);padding-top:16px}.table-heading h3{margin:0;font-size:15px;font-weight:700}.table-heading span{margin-right:auto;color:var(--text-secondary);font-size:12px}.table-heading button{padding:4px 8px;border:1px solid var(--border-main);border-radius:4px;background:var(--bg-main);font-size:12px}.table-filter{display:block;width:100%;margin:12px 0;padding:8px 10px;background:var(--bg-input);color:var(--text-main);border:1px solid var(--border-main);border-radius:4px}.gamer-table-list{display:grid;gap:2px;max-height:300px;overflow:auto;overscroll-behavior:contain}.gamer-table-list label{display:flex;align-items:center;gap:10px;padding:8px 4px;cursor:pointer;min-height:36px}.gamer-table-list label:hover{background:var(--bg-main)}.gamer-table-list input{accent-color:var(--accent);width:16px;height:16px;flex-shrink:0}.gamer-table-list span{min-width:0;overflow-wrap:anywhere}.gamer-table-list small{margin-left:auto;white-space:nowrap;color:var(--text-secondary);font-size:11px}.tool-note{font-size:13px;line-height:1.6;color:var(--text-secondary);margin:12px 0 0}.tool-note button{display:inline-flex;align-items:center;gap:4px;margin-left:8px;color:var(--accent)}.replay-count{margin:0 0 20px}.replay-count strong{font-size:22px;font-variant-numeric:tabular-nums}.replay-actions{display:flex;gap:12px;margin-top:16px}.replay-actions button{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:10px 14px;border:1px solid var(--border-main);border-radius:5px;background:var(--bg-main)}.replay-link{display:inline-flex;align-items:center;gap:5px;margin-top:18px;color:var(--accent)}.gamer-tool-body textarea{display:block;resize:vertical;width:100%;margin-top:12px;padding:8px;background:var(--bg-main);color:var(--text-main);border:1px solid var(--border-main);font:12px ui-monospace,monospace}.gamer-game-statistics{margin:0}.gamer-game-statistics>div{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 0;border-bottom:1px solid var(--border-main)}.gamer-game-statistics dt{color:var(--text-secondary)}.gamer-game-statistics dd{margin:0;font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>
