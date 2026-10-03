<script setup>
import { computed, onMounted, ref } from 'vue';
import { api } from './api.js';
import { language, t } from './i18n.js';
import { PROJECT_BY_ID } from './projects/catalog.js';
import { projectIconUrl } from '../../shared/projectIcons.js';
import { userFacingError } from './errorMessages.js';
const props=defineProps({modelValue:Object});
const emit=defineEmits(['update:modelValue']);
const en=computed(()=>language.value==='en'), catalog=ref([]), loading=ref(false), error=ref('');
const selected=computed(()=>props.modelValue.projects);
function title(id){const p=catalog.value.find(p=>p.project_ref===id);return t(PROJECT_BY_ID[id]?.shortTitle || p?.display_name || id);}
function update(projects){emit('update:modelValue',{projects});}
function toggle(id){update(selected.value.includes(id)?selected.value.filter(p=>p!==id):[...selected.value,id].slice(0,15));}
function move(i,offset){const p=[...selected.value],j=i+offset;if(j<0||j>=p.length)return;[p[i],p[j]]=[p[j],p[i]];update(p);}
async function load(){loading.value=true;error.value='';try{catalog.value=(await api.duelProjects()).projects;}catch(e){error.value=t(userFacingError(e));}finally{loading.value=false;}}
onMounted(load);
</script>
<template>
  <div class="duel-setup">
    <section><h3>{{en?'Choose projects':'选择项目'}} <small>{{selected.length}} / 15</small></h3>
      <p v-if="error" class="alert" role="alert">{{error}}</p>
      <p v-if="loading" class="muted">{{en?'Loading projects…':'正在加载项目…'}}</p>
      <button v-else-if="!catalog.length" type="button" class="secondary-button" @click="load">{{en?'Reload projects':'重新加载项目'}}</button>
      <div class="duel-project-picker"><button v-for="p in catalog" :key="p.project_ref" type="button" :aria-pressed="selected.includes(p.project_ref)" :disabled="!selected.includes(p.project_ref)&&selected.length>=15" @click="toggle(p.project_ref)"><img :src="projectIconUrl(p.project_ref)" alt=""/><span>{{title(p.project_ref)}}</span><b>{{selected.includes(p.project_ref)?'✓':'+'}}</b></button></div>
    </section>
    <section><h3>{{en?'Play order':'出场顺序'}}</h3><p v-if="!selected.length" class="muted">{{en?'Select at least one project.':'至少选择一个项目。'}}</p>
      <ol class="duel-project-order"><li v-for="(id,i) in selected" :key="id"><span>{{i+1}}. {{title(id)}}</span><button type="button" :disabled="i===0" :aria-label="en?'Move up':'上移'" @click="move(i,-1)">↑</button><button type="button" :disabled="i===selected.length-1" :aria-label="en?'Move down':'下移'" @click="move(i,1)">↓</button><button type="button" :aria-label="en?'Remove project':'移除项目'" @click="toggle(id)">×</button></li></ol>
    </section>
  </div>
</template>
<style scoped>
.duel-setup{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:24px}.duel-setup h3{margin:0 0 14px}.duel-setup small{font-size:13px;color:var(--competition-muted)}.duel-project-picker{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;max-height:400px;overflow:auto;padding:2px}.duel-project-picker button{display:flex;align-items:center;gap:8px;padding:10px;min-width:0;text-align:left;border:1px solid var(--competition-border);border-radius:9px;background:transparent;color:inherit;font:inherit;font-size:14px}.duel-project-picker button[aria-pressed=true]{border-color:#b88b37;background:#b88b3715}.duel-project-picker img{width:30px;height:30px;flex:none}.duel-project-picker span{flex:1;overflow-wrap:anywhere}.duel-project-order{padding:0;list-style:none;margin:0}.duel-project-order li{display:flex;align-items:center;gap:6px;padding:8px 0;border-bottom:1px solid var(--competition-border)}.duel-project-order span{flex:1;min-width:0;overflow-wrap:anywhere}.duel-project-order button{min-height:40px;min-width:34px;border:1px solid var(--competition-border);border-radius:6px;background:transparent;color:inherit}.duel-setup button:disabled{opacity:.45}@media(max-width:780px){.duel-setup{grid-template-columns:1fr}.duel-project-order button{min-height:44px}}
</style>
