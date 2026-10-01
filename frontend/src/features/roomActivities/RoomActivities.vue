<template>
  <div class="room-activities">
    <LuckyBags v-if="room.capabilities.lucky_bags" ref="lucky" :state="luckyState" v-bind="common" @open="activate('lucky')" @login="emit('login')" @balance="emit('balance')">
      <template #default="slot"><slot v-bind="slot" /></template>
    </LuckyBags>
    <slot v-else :bag="null" :open="() => {}" caption="" />
    <RedEnvelopes v-if="room.capabilities.red_envelopes" ref="red" :state="redState" v-bind="common" @open="activate('red')" @login="emit('login')" @balance="emit('balance')" />
    <component :is="room.content_kind === 'competition-match' ? CompetitionPredictions : Predictions" v-if="room.capabilities.predictions" ref="predictions" :state="predictionState" :online="online" v-bind="common" :entry-target="predictionTarget" @open="activate('predictions')" @login="emit('login')" @balance="emit('balance')" />
  </div>
</template>
<script setup>
import { ref, computed } from 'vue';
import LuckyBags from './LuckyBags.vue';
import RedEnvelopes from './RedEnvelopes.vue';
import Predictions from './Predictions.vue';
import CompetitionPredictions from './CompetitionPredictions.vue';
import { provideActivities } from './context.js';
const props = defineProps({room:Object,transport:Object,user:Object,connected:Boolean,online:Boolean,lang:String,luckyState:Object,redState:Object,predictionState:Object,dockTarget:String,predictionTarget:String});
const emit = defineEmits(['login','balance']);
provideActivities({...props.transport,room:props.room});
const lucky=ref(null),red=ref(null),predictions=ref(null);
const common=computed(()=>({user:props.user,connected:props.connected,lang:props.lang,dockTarget:props.dockTarget}));
function activate(name) { for(const [id,component] of Object.entries({lucky,red,predictions})) if(id!==name)component.value?.close(); }
defineExpose({compose:()=>red.value?.compose(),open:id=>red.value?.open(id)});
</script>
<style scoped>
.room-activities { min-width:0; }
</style>
