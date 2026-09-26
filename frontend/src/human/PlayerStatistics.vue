<template>
  <section class="player-statistics" :aria-busy="loading">
    <div class="statistics-variants">
      <button v-for="variant in variants" :key="variant" :class="{ active: selected === variant }" @click="selected = variant">
        <span>{{ variant.replace('x',' × ') }}</span>
      </button>
    </div>
    <p v-if="error" class="notice danger" role="alert">{{ t(error) }} <button @click="load(true)">{{ t('重试') }}</button></p>
    <div v-if="loading && !current" class="large-empty">{{ t('正在读取统计…') }}</div>
    <div v-else-if="!current" class="large-empty">{{ t('此模式暂无统计记录') }}</div>
    <template v-else>
      <div class="statistics-current">
        <article><span>PB</span><strong>{{ integer(current.pb_score) }}</strong></article>
        <article><span>{{ t('B10 分数线') }}</span><strong>{{ current.b10_score == null ? '—' : integer(current.b10_score) }}</strong></article>
        <article><span>B10 RATING</span><strong>{{ format(current.b10_rating, 1) }}</strong></article>
        <article><span>{{ t('正式记录') }}</span><strong>{{ integer(current.game_count) }}</strong></article>
      </div>

      <div class="statistics-toolbar">
        <div><strong>{{ selected.replace('x',' × ') }}</strong><span>{{ t('成长轨迹') }}</span></div>
        <div class="statistics-toolbar-actions">
          <button :disabled="loading || posterBusy || !Object.keys(featureLabels).length" @click="downloadPoster">{{ posterBusy ? t('正在生成…') : t('下载分享图') }}</button>
          <div class="statistics-axis-switch" role="group" :aria-label="t('横轴')">
            <button :class="{ active: axis === 'games' }" @click="axis = 'games'">{{ t('对局数') }}</button>
            <button :class="{ active: axis === 'time' }" @click="axis = 'time'">{{ t('时间') }}</button>
          </div>
        </div>
      </div>

      <div class="statistics-charts">
        <StatChart :points="series" :axis="axis" metric="score" :title="t('PB 与 B10 分数线')"
          :labels="[t('PB'),t('B10 分数线')]" :colors="['#c7993d','#b86f4c']" />
        <StatChart :points="series" :axis="axis" metric="rating" title="B10 Rating"
          :labels="['B10 Rating']" :colors="['#5d9a91']" />
      </div>

      <div class="statistics-lower">
        <section class="panel achievement-panel">
          <header><div><span class="player-kicker">{{ selected.replace('x',' × ') }}</span><h2>{{ t('终盘成就') }}</h2></div><small>{{ t('包含式计数') }}</small></header>
          <div class="achievement-bars" :style="{ '--bar-count': Object.keys(featureLabels).length }">
            <article v-for="(label,key,index) in featureLabels" :key="key">
              <strong>{{ integer(current.features?.[key] || 0) }}</strong>
              <div class="achievement-column" aria-hidden="true">
                <i :class="{ empty: !(current.features?.[key] || 0) }" :style="{ height: achievementHeight(key) }"></i>
              </div>
              <span>{{ translatedFeature(label) }}</span>
              <small v-if="index">{{ retention(key,index) }}</small>
              <small v-else>{{ t('达成局数') }}</small>
            </article>
          </div>
        </section>
        <StatChart v-if="selected === '4x4'" class="rate-progress-chart" :points="rateSeries" :axis="axis"
          metric="rate32k" :title="t('32K 综率成长')" :labels="[t('32K 综率')]" :colors="['#d2ad5f']"
          :summary="percent(current.rate_32k?.value)"
          :note="`${integer(current.rate_32k?.passed || 0)} / ${integer(current.rate_32k?.total || 0)} ${t('个残局阶段')} · ${integer(current.rate_32k?.candidate_games || 0)} ${language === 'en' ? 'games reached 32K' : '局达到 32K'}`" />
      </div>
    </template>
  </section>
</template>

<script setup>
import { computed, defineComponent, h, onMounted, ref, watch } from 'vue';
import { json } from './client.js';
import { t, language } from './i18n.js';
import { drawStatisticsPoster } from './statisticsPoster.js';

const props = defineProps({ username: { type: String, required: true } });
const variants = ['4x4','3x4','2x4','3x3'];
const selected = ref('4x4'), axis = ref('games'), loading = ref(false), error = ref('');
const posterBusy = ref(false), player = ref(null);
const summaries = ref({}), series = ref([]), rateSeries = ref([]), featureLabels = ref({});
const cache = new Map(); let serial = 0;
const current = computed(() => summaries.value[selected.value] || null);
const integer = value => Number(value || 0).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN');
const format = (value, digits) => value == null ? '—' : Number(value).toLocaleString(undefined,{maximumFractionDigits:digits,minimumFractionDigits:digits});
const percent = value => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
const translatedFeature = label => language.value === 'en' ? ({'满盘':'Full board','二阶满盘':'Second full board','三阶满盘':'Third full board'}[label] || label) : label;
function retention(key,index) {
  const keys = Object.keys(featureLabels.value), previous = current.value.features?.[keys[index - 1]] || 0;
  const value = current.value.features?.[key] || 0;
  return previous ? `${(value / previous * 100).toFixed(1)}%` : '—';
}
function achievementHeight(key) {
  const values = Object.keys(featureLabels.value).map(name => Number(current.value.features?.[name] || 0));
  const maximum = Math.max(...values, 1), value = Number(current.value.features?.[key] || 0);
  return value ? `${Math.max(5, value / maximum * 100)}%` : '0%';
}
async function load(force = false) {
  const key = `${props.username}:${selected.value}`;
  if (!force && cache.has(key)) return apply(cache.get(key));
  const own = ++serial; loading.value = true; error.value = '';
  try {
    const result = await json(`/api/human/users/${encodeURIComponent(props.username)}/statistics?variant=${selected.value}`);
    if (own !== serial) return;
    cache.set(key,result); apply(result);
  } catch { if (own === serial) error.value = '无法读取统计，请稍后重试。'; }
  finally { if (own === serial) loading.value = false; }
}
function apply(result) {
  player.value = result.player || null;
  summaries.value = result.summaries || {};
  series.value = result.series || [];
  rateSeries.value = result.rate_32k_series || [];
  featureLabels.value = result.feature_labels || {};
}
async function downloadPoster() {
  if (!current.value || loading.value || posterBusy.value || !Object.keys(featureLabels.value).length) return;
  posterBusy.value = true; error.value = '';
  try {
    const canvas = await drawStatisticsPoster({ name: player.value?.display_name || props.username,
      variant: selected.value, axis: axis.value, summary: current.value, series: series.value,
      rateSeries: rateSeries.value, featureLabels: featureLabels.value,
      dark: document.documentElement.dataset.theme === 'dark', language: language.value });
    const blob = await new Promise(resolve => canvas.toBlob(resolve,'image/png'));
    if (!blob) throw new Error('poster_blob_unavailable');
    const url=URL.createObjectURL(blob), link=document.createElement('a');
    link.href=url;link.download=`2048-${selected.value}-statistics-${axis.value}.png`;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  } catch { error.value='无法生成分享图。'; }
  finally { posterBusy.value=false; }
}
watch(selected, () => { series.value=[]; rateSeries.value=[]; featureLabels.value={}; load(); });
watch(() => props.username, () => { cache.clear(); summaries.value={}; series.value=[]; load(); });
onMounted(() => load());

const StatChart = defineComponent({
  props: { points:Array, axis:String, metric:String, title:String, labels:Array, colors:Array, summary:String, note:String },
  setup(chartProps) {
    const active = ref(-1), width=640, height=300, left=76, right=20, top=28, bottom=54;
    const rows = computed(() => chartProps.points || []);
    const xValue = point => chartProps.axis === 'time' ? point.ended_at : point.game_index;
    const fields = computed(() => chartProps.metric === 'score' ? ['pb_score','b10_score'] : chartProps.metric === 'rate32k' ? ['value'] : ['b10_rating']);
    const extent = computed(() => {
      const xs=rows.value.map(xValue).filter(Number.isFinite), ys=rows.value.flatMap(point => fields.value.map(field => point[field])).filter(Number.isFinite);
      const dataMinX=xs.length ? Math.min(...xs) : 0, dataMaxX=xs.length ? Math.max(...xs) : 1;
      const xSpan=dataMaxX-dataMinX, xPadding=xSpan ? xSpan*.025 : chartProps.axis==='time' ? 86400 : 1;
      const low=ys.length ? Math.min(...ys) : 0, high=ys.length ? Math.max(...ys) : 1;
      const yPadding=Math.max((high-low)*.1,chartProps.metric==='rating' ? 35 : Math.max(high*.025,1));
      let minY=chartProps.metric==='rating'?low-yPadding:Math.max(0,low-yPadding), maxY=high+yPadding;
      if (chartProps.metric === 'rate32k') {
        const center=(low+high)/2;
        const span=Math.max(.2,(high-low)*1.16);
        minY=Math.max(0,center-span/2); maxY=Math.min(1,center+span/2);
        if(maxY-minY<.2) { if(minY===0) maxY=.2; else minY=maxY-.2; }
      }
      if(maxY===minY) maxY=minY+1;
      return {minX:dataMinX-xPadding,maxX:dataMaxX+xPadding,dataMinX,dataMaxX,minY,maxY};
    });
    const x = value => extent.value.maxX === extent.value.minX ? (left + width - right) / 2 : left + (value-extent.value.minX)/(extent.value.maxX-extent.value.minX)*(width-left-right);
    const y = value => height-bottom-(value-extent.value.minY)/(extent.value.maxY-extent.value.minY)*(height-top-bottom);
    const path = field => {
      let output='', previous=null;
      rows.value.forEach(point => { const value=point[field]; if(!Number.isFinite(value)) return;
        const px=x(xValue(point)), py=y(value); output += previous ? `L${px},${py}` : `M${px},${py}`; previous=point; });
      return output;
    };
    const tickValue = value => chartProps.metric==='score' ? (value>=1e6?`${(value/1e6).toFixed(1)}M`:value>=1000?`${Math.round(value/1000)}k`:Math.round(value)) : chartProps.metric==='rate32k' ? `${Math.round(value*100)}%` : Number(value).toFixed(0);
    const date = value => new Date(value*1000).toLocaleDateString(language.value==='en'?'en-US':'zh-CN',{year:'numeric',month:'short',day:'numeric'});
    const shortDate = value => new Date(value*1000).toLocaleString(language.value==='en'?'en-US':'zh-CN',
      extent.value.dataMaxX-extent.value.dataMinX<172800
        ? {month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}
        : {year:'2-digit',month:'numeric',day:'numeric'});
    const xTicks = computed(() => {
      const {dataMinX:min,dataMaxX:max}=extent.value;
      if (min === max) return [min];
      const fractions=chartProps.axis==='time'&&max-min<172800?[0,.5,1]:[0,.25,.5,.75,1];
      return fractions.map(fraction => min+(max-min)*fraction);
    });
    const xTickValue = value => chartProps.axis==='games' ? `${Math.round(value)}` : shortDate(value);
    function pointer(event) {
      if (!rows.value.length) return;
      const rect=event.currentTarget.getBoundingClientRect(), px=(event.clientX-rect.left)/rect.width*width;
      active.value=rows.value.reduce((best,point,index) => Math.abs(x(xValue(point))-px)<Math.abs(x(xValue(rows.value[best]))-px)?index:best,0);
    }
    return () => h('article',{class:'panel statistic-chart'},[
      h('header',{class:'stat-chart-heading'},[
        h('div',[h('h3',chartProps.title), chartProps.note ? h('small',chartProps.note) : null]),
        chartProps.summary ? h('strong',chartProps.summary) : null
      ]),
      h('div',{class:'stat-chart-legend'},fields.value.map((field,index)=>h('span',[
        h('i',{style:{background:chartProps.colors[index]}}),chartProps.labels[index]
      ]))),
      rows.value.length ? h('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':chartProps.title,onPointermove:pointer,onPointerleave:()=>active.value=-1},[
        ...[0,.25,.5,.75,1].map(fraction=>{const value=extent.value.minY+(extent.value.maxY-extent.value.minY)*fraction, py=y(value);return h('g',{},[h('line',{x1:left,x2:width-right,y1:py,y2:py,class:'stat-grid'}),h('text',{x:left-12,y:py+6,class:'stat-axis','text-anchor':'end'},tickValue(value))]);}),
        ...xTicks.value.map(value=>h('g',{},[
          h('line',{x1:x(value),x2:x(value),y1:top,y2:height-bottom,class:'stat-grid stat-grid-x'}),
          h('text',{x:x(value),y:height-16,class:'stat-axis stat-axis-x','text-anchor':'middle'},xTickValue(value))
        ])),
        ...fields.value.map((field,index)=>h('path',{d:path(field),class:'stat-line',stroke:chartProps.colors[index],fill:'none'})),
        ...(chartProps.metric==='rate32k' ? rows.value.map(point=>h('circle',{cx:x(xValue(point)),cy:y(point.value),r:2.8,class:'stat-point',fill:chartProps.colors[0]})) : []),
        ...rows.value.map((point,index)=>h('circle',{cx:x(xValue(point)),cy:y(point[fields.value[0]]),r:10,class:'stat-hit',tabindex:0,onFocus:()=>active.value=index,onBlur:()=>active.value=-1})),
        active.value>=0 ? h('g',{class:'stat-tooltip'},[
          h('line',{x1:x(xValue(rows.value[active.value])),x2:x(xValue(rows.value[active.value])),y1:top,y2:height-bottom}),
          ...fields.value.filter(field=>Number.isFinite(rows.value[active.value][field])).map((field,index)=>h('circle',{cx:x(xValue(rows.value[active.value])),cy:y(rows.value[active.value][field]),r:5,fill:chartProps.colors[index]}))
        ]) : null
      ]) : h('div',{class:'large-empty'},t('暂无趋势数据')),
      active.value>=0 ? h('div',{class:'stat-chart-tooltip'},[
        h('strong',chartProps.axis==='games'?(language.value==='en'?`Game ${rows.value[active.value].game_index}`:`第 ${rows.value[active.value].game_index} 局`):date(rows.value[active.value].ended_at)),
        ...fields.value.filter(field=>Number.isFinite(rows.value[active.value][field])).map((field,index)=>h('span',{style:{color:chartProps.colors[index]}},`${chartProps.labels[index]} ${field === 'b10_rating' ? format(rows.value[active.value][field], 1) : field === 'value' ? percent(rows.value[active.value][field]) : integer(rows.value[active.value][field])}`))
      ]) : null
    ]);
  }
});
</script>
