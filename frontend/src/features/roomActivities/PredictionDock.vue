<template>
  <RoomActivityEntry v-if="active && !dismissed.includes(active.id)" :target="target" kind="prediction" :label="t('下注','Predict')" :caption="caption" :dismiss-label="t('收起下注','Dismiss predictions')" @dismiss="dismiss(active.id)" @open="$emit('open')"><img :src="artwork" alt="" draggable="false" /></RoomActivityEntry>
</template>
<script setup>
import {computed,ref,onMounted,onUnmounted} from 'vue';
import RoomActivityEntry from './RoomActivityEntry.vue';
import {useActivityDismissals} from './useActivityDismissals.js';
import {openPredictionMarket} from './activityAvailability.js';
import artwork from './assets/prediction-colored.webp';
const props=defineProps({state:Object,connected:Boolean,online:Boolean,target:String,lang:String});
defineEmits(['open']);
const {dismissed,dismiss}=useActivityDismissals('prediction');
const now=ref(Date.now()/1000);let timer;
const active=computed(()=>openPredictionMarket(props.state,now.value,props.connected,props.online));
const t=(zh,en)=>props.lang==='zh'?zh:en;
const caption=computed(()=>{
  if(active.value?.deadline==null)return t('下注开放','Entries open');
  const seconds=Math.max(0,Math.ceil(active.value.deadline-now.value));
  return `${t('截止','Closes in')} ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;
});
onMounted(()=>timer=setInterval(()=>{now.value=Date.now()/1000},1000));
onUnmounted(()=>clearInterval(timer));
</script>
