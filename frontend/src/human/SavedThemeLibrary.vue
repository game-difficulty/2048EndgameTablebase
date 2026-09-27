<template>
  <section class="saved-theme-library">
    <div class="saved-theme-heading"><span><strong>{{ t('已保存主题') }}</strong><small>{{ themes.length }} / {{ limit }}</small></span><div><button type="button" :disabled="themes.length >= limit" @click="createNew">{{ t('新建') }}</button><button type="button" @click="fileInput?.click()">{{ t('导入 .vth') }}</button><button type="button" :disabled="themes.length >= limit" @click="createFromCurrent">{{ t('保存当前配色') }}</button></div></div>
    <input ref="fileInput" class="visually-hidden" type="file" accept=".vth,application/json" @change="importFile">
    <p v-if="error" class="error-text" role="alert">{{ t(error) }}</p>
    <p v-if="!themes.length && !busy" class="muted">{{ t('还没有保存的 .vth 主题。') }}</p>
    <div v-else class="saved-theme-list">
      <button v-for="item in themes" :key="item.id" type="button" :class="{active:Number(preferences.saved_theme_id)===item.id,selected:selectedId===item.id}" @click="select(item.id)"><span>{{ item.name }}</span><small>{{ Number(preferences.saved_theme_id)===item.id?t('使用中'):t('编辑') }}</small></button>
    </div>
    <div v-if="draft" class="theme-editor">
      <div class="theme-editor-actions"><input v-model.trim="draftName" maxlength="64" :aria-label="t('主题名称')"><button type="button" @click="activate">{{ t('应用') }}</button><button type="button" @click="exportTheme">{{ t('导出') }}</button><button type="button" class="danger-text" @click="remove">{{ t('删除') }}</button></div>
      <div class="settings-options"><button v-for="mode in availableModes" :key="mode" type="button" :class="{active:editMode===mode}" @click="editMode=mode">{{ mode==='light'?t('浅色'):t('深色') }}</button><button v-if="availableModes.length<2" type="button" @click="addMode">+ {{ t('另一模式') }}</button></div>
      <div class="vth-tile-picker"><button v-for="value in tileValues" :key="value" type="button" :class="{active:selectedTile===value}" :style="previewStyle(value)" @click="selectedTile=value">{{ value }}</button></div>
      <div class="vth-fields">
        <label v-for="field in fields" :key="field.key"><span>{{ t(field.label) }}</span><input v-model.trim="currentStyle[field.key]" spellcheck="false"></label>
      </div>
      <button type="button" :disabled="busy" @click="save">{{ t(busy?'保存中…':'保存修改') }}</button>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue';
import themesCatalog from '../../../docs_and_configs/themes.json';
import { createLocalStorageStore } from '../services/storage/localStorageStore.js';
import { saveAccountPreferences } from '../services/preferences/accountPreferences.js';
import { applySavedThemePayload, clearSavedThemeStyles, createSavedTheme, deleteSavedTheme, downloadVth, getSavedTheme, listSavedThemes, normalizeVthTheme, updateSavedTheme, VTH_TILE_VALUES } from '../services/preferences/savedThemes.js';
import { resolveTileColors } from '../utils/tileColors.js';
import { t } from './i18n.js';

const preferencesStore=createLocalStorageStore({key:'user-preferences',version:1,defaultValue:{}});
const preferences=ref(preferencesStore.read());
const themes=ref([]),limit=ref(32),selectedId=ref(0),draft=ref(null),draftName=ref(''),editMode=ref('light'),selectedTile=ref(2),busy=ref(false),error=ref(''),fileInput=ref(null);
const tileValues=VTH_TILE_VALUES;
const fields=[{key:'--tile-color',label:'文字颜色'},{key:'--tile-background',label:'棋块颜色'},{key:'--tile-shadow-color',label:'光晕颜色'},{key:'--tile-outline-color',label:'描边颜色'}];
const availableModes=computed(()=>Object.keys(draft.value||{}));
const currentStyle=computed(()=>draft.value?.[editMode.value]?.[selectedTile.value]||{});
const previewStyle=value=>{const style=draft.value?.[editMode.value]?.[value];return style?{background:style['--tile-background'],color:style['--tile-color'],boxShadow:`0 0 8px ${style['--tile-shadow-color']},inset 0 0 0 1px ${style['--tile-outline-color']}`}:{}};
const clone=value=>JSON.parse(JSON.stringify(value));
function themeFromColors(source){
  const palette=resolveTileColors(VTH_TILE_VALUES.map((_,i)=>source[i]||themesCatalog.Default[i]));
  const mode=Object.fromEntries(VTH_TILE_VALUES.map((value,index)=>[value,{'--tile-color':palette[index].color,'--tile-background':palette[index].background,'--tile-shadow-color':'#00000000','--tile-outline-color':'#00000000'}]));
  return {light:clone(mode),dark:clone(mode)};
}
function generatedTheme(){const stored=preferencesStore.read();return themeFromColors(stored.use_custom_theme?stored.custom_colors:(themesCatalog[stored.theme]||themesCatalog.Default));}
function availableName(base){const names=new Set(themes.value.map(item=>item.name.toLocaleLowerCase()));if(!names.has(base.toLocaleLowerCase()))return base;for(let index=2;;index++){const candidate=`${base} ${index}`;if(!names.has(candidate.toLocaleLowerCase()))return candidate;}}
async function reload(){const result=await listSavedThemes();themes.value=result.themes||[];limit.value=result.limit||32;}
async function select(id){busy.value=true;error.value='';try{const item=await getSavedTheme(id,{force:true});selectedId.value=id;draft.value=clone(item.theme);draftName.value=item.name;editMode.value=item.theme.light?'light':'dark';}catch(e){error.value=e.code||'主题读取失败。';}finally{busy.value=false;}}
async function importFile(event){const file=event.target.files?.[0];event.target.value='';if(!file)return;busy.value=true;error.value='';try{if(file.size>64*1024)throw new Error('theme_too_large');const raw=normalizeVthTheme(JSON.parse(await file.text()));const name=file.name.replace(/\.vth$/i,'').slice(0,64)||t('导入主题');const item=await createSavedTheme(name,raw);await reload();await select(item.id);}catch(e){error.value=e.code||e.message||'主题导入失败。';}finally{busy.value=false;}}
async function createNew(){busy.value=true;error.value='';try{const item=await createSavedTheme(availableName(t('新主题')),themeFromColors(themesCatalog.Default));await reload();await select(item.id);}catch(e){error.value=e.code||'主题保存失败。';}finally{busy.value=false;}}
async function createFromCurrent(){busy.value=true;error.value='';try{const item=await createSavedTheme(availableName(t('当前配色')),generatedTheme());await reload();await select(item.id);}catch(e){error.value=e.code||'主题保存失败。';}finally{busy.value=false;}}
function addMode(){const source=draft.value.light||draft.value.dark;const mode=draft.value.light?'dark':'light';draft.value[mode]=clone(source);editMode.value=mode;}
async function save(){busy.value=true;error.value='';try{const item=await updateSavedTheme(selectedId.value,draftName.value,draft.value);draft.value=clone(item.theme);await reload();if(Number(preferences.value.saved_theme_id)===item.id){clearSavedThemeStyles();applySavedThemePayload(item.theme,preferences.value.dark_mode!==false);}}catch(e){error.value=e.code||'主题保存失败。';}finally{busy.value=false;}}
function activate(){if(!draft.value)return;preferences.value=preferencesStore.update(current=>({...current,saved_theme_id:selectedId.value,use_custom_theme:false}));saveAccountPreferences({saved_theme_id:selectedId.value,use_custom_theme:false});clearSavedThemeStyles();applySavedThemePayload(draft.value,preferences.value.dark_mode!==false);window.dispatchEvent(new Event('human-preferences-changed'));}
function exportTheme(){if(draft.value)downloadVth(draftName.value,draft.value);}
async function remove(){busy.value=true;error.value='';try{await deleteSavedTheme(selectedId.value);if(Number(preferences.value.saved_theme_id)===selectedId.value){preferences.value=preferencesStore.update(current=>({...current,saved_theme_id:0}));saveAccountPreferences({saved_theme_id:0});window.dispatchEvent(new Event('human-preferences-changed'));}draft.value=null;selectedId.value=0;await reload();}catch(e){error.value=e.code||'主题删除失败。';}finally{busy.value=false;}}
onMounted(()=>reload().catch(()=>{error.value='主题列表读取失败。';}));
</script>
