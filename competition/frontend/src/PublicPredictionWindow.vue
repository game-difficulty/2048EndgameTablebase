<script setup>
import {computed,onMounted,onBeforeUnmount,ref,watch} from 'vue';
import {language} from './i18n.js';
import {ServerClock} from '../../shared/serverClock.mjs';
const props=defineProps({room:Object});
const en=computed(()=>language.value==='en');
const clock=new ServerClock(),now=ref(clock.now());let timer;
watch(()=>props.room.server_time,value=>clock.observe(value),{immediate:true});
const seconds=computed(()=>Math.max(0,Math.ceil((Date.parse(props.room.prediction_window?.minimum_until)-now.value)/1000)));
onMounted(()=>timer=setInterval(()=>now.value=clock.now(),250));
onBeforeUnmount(()=>clearInterval(timer));
</script>
<template><div v-if="room.prediction_window?.open" class="prediction-window" role="status">
  <strong>{{en?'Spectator entries open':'观众下注开放'}} · {{seconds}} {{en?'seconds':'秒'}}</strong>
  <p>{{en?'Both players have confirmed. Seats are locked; play starts after this window. Match clocks have not started.':'双方已主动准备，席位已锁定。窗口结束后开赛，目前不扣比赛用时。'}}</p>
</div></template>
<style scoped>.prediction-window{padding:16px;border:1px solid #b58f45;border-radius:10px;background:#b58f4510}.prediction-window strong{font-size:22px}.prediction-window p{font-size:14px;line-height:1.6;margin-bottom:0}</style>
