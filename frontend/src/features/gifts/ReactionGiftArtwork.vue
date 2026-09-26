<template>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" fill="none" :class="[id, { animated }]" aria-hidden="true">
<g v-if="id === 'bad-four'">
<ellipse cx="64" cy="116" rx="37" ry="5" fill="#162239" opacity=".15"/>
<g class="body">
 <path d="M29 37 22 10q20 6 24 23M84 31q7-17 24-21l-8 30" fill="#de5963" stroke="#923f50" stroke-width="3" stroke-linejoin="round"/>
 <path d="m27 17 5 15 7-1m62-14-8 15 7 1" fill="#ffaea0"/>
 <rect x="20" y="31" width="89" height="81" rx="16" fill="#b1834e" stroke="#806044" stroke-width="3"/>
 <rect x="20" y="25" width="89" height="81" rx="16" fill="#ebc994" stroke="#ac824f" stroke-width="2.5"/>
 <path d="M24 79q39 16 81-6v18q0 12-13 12H36q-12 0-12-12Z" fill="#d9ae76"/>
 <path d="M29 47V39q0-5 6-5h15" stroke="#fff3d9" stroke-width="4" stroke-linecap="round" fill="none"/>
 <path d="m38 45 10 4m31 0 11-5" stroke="#73503b" stroke-width="3" stroke-linecap="round"/>
 <text x="65" y="87" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="55" fill="#725035">4</text>
 <path d="M82 91q7 6 13-2" stroke="#8d423b" stroke-width="3" fill="none" stroke-linecap="round"/>
</g>
<g class="anger" stroke="#d65d65" stroke-width="3.5" fill="none" stroke-linecap="round"><path d="M109 50v7h7m-7 14v-7h7"/></g>
</g>
<g v-else-if="id === 'dealer-fault'">
<ellipse cx="63" cy="118" rx="43" ry="4" fill="#162239" opacity=".15"/>
<g class="card-back" transform="rotate(-15 46 35)"><rect x="26" y="10" width="40" height="55" rx="6" fill="#dce3ee" stroke="#7b8da4" stroke-width="2.5"/><rect x="32" y="16" width="28" height="43" rx="3" fill="#637590"/><path d="m37 27 18 21m0-21L37 48" stroke="#b8d0df" stroke-width="2"/></g>
<g class="dealt-card"><g transform="translate(8 -4) rotate(14 69 41)"><rect x="47" y="9" width="43" height="58" rx="6" fill="#fff0ce" stroke="#bd955b" stroke-width="2.5"/><path d="M53 24v-8h10" fill="none" stroke="#fffaf0" stroke-width="3"/><text x="69" y="51" text-anchor="middle" font-family="Arial,sans-serif" font-weight="900" font-size="37" fill="#b65b47">4</text><path d="m80 57 3-4 3 4-3 4Z" fill="#bd7658"/></g></g>
<g class="hand">
 <path d="m11 87 16-16q9-11 22-12l14-20q4-5 8 0l-8 23 17-8q9-3 10 4l-17 13 20-6q9-1 8 6L79 84 58 90l-24 22Z" fill="#c8d8e2" stroke="#647c90" stroke-width="2.7" stroke-linejoin="round"/>
 <path d="m18 86 14-12q8-10 20-10l14-22-8 24 12 3-20 17-18 11Z" fill="#f3f8f5"/>
 <path d="m68 75 17-13M71 81l19-10M34 75l8 4" fill="none" stroke="#a2bac8" stroke-width="2" stroke-linecap="round"/>
 <path d="m10 86 27 24-13 14L0 99Z" fill="#344a68" stroke="#657d95" stroke-width="2.5"/><path d="m14 86 26 23-5 5L9 91Z" fill="#ebf0ed"/><circle cx="15" cy="106" r="3" fill="#f0c76b"/>
</g>
<g class="verdict"><path d="m102 65 6 5 8 1 1 8 5 6-5 6-1 8-8 1-6 5-6-5-8-1-1-8-5-6 5-6 1-8 8-1Z" fill="#df656a" stroke="#9b4355" stroke-width="2.5"/><circle cx="102" cy="85" r="13" fill="none" stroke="#ffc9b5" stroke-width="1.5"/><path d="M102 76v10" stroke="#fff3dc" stroke-width="5" stroke-linecap="round"/><circle cx="102" cy="93" r="2.6" fill="#fff3dc"/></g>
</g>
<g v-else-if="id === 'cry-loss'">
<ellipse cx="64" cy="118" rx="44" ry="4" fill="#162239" opacity=".15"/>
<g class="face"><path d="M108 61c0 28-17 47-44 47S20 89 20 61s17-44 44-44 44 16 44 44Z" fill="#d79635" stroke="#ac762e" stroke-width="2.5"/><ellipse cx="64" cy="59" rx="44" ry="43" fill="#f4c35d"/><path d="M29 40q8-16 25-17" fill="none" stroke="#ffe4a1" stroke-width="5" stroke-linecap="round"/><ellipse cx="35" cy="70" rx="9" ry="5" fill="#e89464" opacity=".65"/><ellipse cx="93" cy="70" rx="9" ry="5" fill="#e89464" opacity=".65"/>
 <path d="M34 47q8 2 17-6m27 0q9 8 18 6" fill="none" stroke="#805034" stroke-width="3" stroke-linecap="round"/>
 <path d="m36 56 13 5-13 5m56-10-13 5 13 5" fill="none" stroke="#705039" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
 <path d="M34 67h12l1 25q-5 9-15 5Z" fill="#69bfde" stroke="#388da9" stroke-width="2"/><path d="M82 67h12l2 30q-10 5-15-5Z" fill="#69bfde" stroke="#388da9" stroke-width="2"/>
 <path d="M38 70v19m49-19v19" stroke="#c3f1f1" stroke-width="3" stroke-linecap="round"/>
 <path d="M52 90q-1-18 12-18t12 18q-12 10-24 0Z" fill="#794239" stroke="#704232" stroke-width="2"/><path d="M56 90q8-8 16 0" fill="#ed8c80"/>
 <ellipse cx="25" cy="90" rx="11" ry="13" transform="rotate(-22 25 90)" fill="#f3c361" stroke="#b98539" stroke-width="2.5"/><ellipse cx="103" cy="90" rx="11" ry="13" transform="rotate(22 103 90)" fill="#f3c361" stroke="#b98539" stroke-width="2.5"/>
</g>
<g class="tear tear-left"><path d="M35 100q-11 14 0 14t0-14" fill="#70c8e5" stroke="#3e98b4" stroke-width="1.5"/></g>
<g class="tear tear-right"><path d="M94 100q-11 14 0 14t0-14" fill="#70c8e5" stroke="#3e98b4" stroke-width="1.5"/></g>
<g class="lost-chip"><g transform="translate(65 111) rotate(-20)"><ellipse rx="12" ry="6" fill="#ae813c"/><ellipse cy="-3" rx="12" ry="6" fill="#e7bd65" stroke="#fff0b9" stroke-width="1.5"/><path d="M-5-3H5" stroke="#b17c35" stroke-width="2"/></g></g>
</g>
<g v-else-if="id === 'laugh-win'">
<ellipse cx="64" cy="118" rx="44" ry="4" fill="#162239" opacity=".15"/>
<g class="coin coin-left"><circle cx="14" cy="30" r="10" fill="#e0aa40" stroke="#aa782d" stroke-width="2"/><circle cx="14" cy="29" r="6.5" fill="#f9d878" stroke="#fff1b5" stroke-width="1.5"/><path d="M14 25v8" stroke="#b88734" stroke-width="2"/></g>
<g class="coin coin-right"><circle cx="113" cy="22" r="10" fill="#e0aa40" stroke="#aa782d" stroke-width="2"/><circle cx="113" cy="21" r="6.5" fill="#f9d878" stroke="#fff1b5" stroke-width="1.5"/><path d="M113 17v8" stroke="#b88734" stroke-width="2"/></g>
<g class="face"><path d="M108 61c0 28-17 47-44 47S20 89 20 61s17-44 44-44 44 16 44 44Z" fill="#d79635" stroke="#ac762e" stroke-width="2.5"/><ellipse cx="64" cy="59" rx="44" ry="43" fill="#f4c35d"/><path d="M29 40q8-16 25-17" fill="none" stroke="#ffe4a1" stroke-width="5" stroke-linecap="round"/><ellipse cx="35" cy="70" rx="9" ry="5" fill="#e89464" opacity=".65"/><ellipse cx="93" cy="70" rx="9" ry="5" fill="#e89464" opacity=".65"/>
 <path d="M34 47q7-6 15-5m30 0q8-1 15 5" fill="none" stroke="#805034" stroke-width="3" stroke-linecap="round"/>
 <path d="m35 55 13 6-13 5m58-11-13 6 13 5" fill="none" stroke="#705039" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
 <path d="M42 73q22 6 44 0-2 26-22 26T42 73Z" fill="#7b4238" stroke="#704232" stroke-width="2"/><path d="M45 75q19 4 38 0l-4 9H49Z" fill="#fff8dd"/><path d="M53 95q11-14 22 0-11 7-22 0Z" fill="#e58678"/>
 <path d="M29 67q-13 9-7 13 8 4 7-13Z" fill="#77c9e0"/><path d="M100 66q13 9 7 13-8 4-7-13Z" fill="#77c9e0"/>
</g>
<g class="hands"><path d="m24 102 7-11q3-4 6 0l3 8 4-4q5-3 7 2l-1 12-10 9Z" fill="#f7ca6d" stroke="#b98539" stroke-width="2.5"/><path d="m104 102-7-11q-3-4-6 0l-3 8-4-4q-5-3-7 2l1 12 10 9Z" fill="#f7ca6d" stroke="#b98539" stroke-width="2.5"/></g>
<g class="spark" fill="#f4ce74" stroke="#bd873c" stroke-width="1"><path d="m8 56 3 6 6 2-6 2-3 6-2-6-6-2 6-2Z"/><path d="m115 51 2 5 6 2-6 2-2 5-2-5-6-2 6-2Z"/></g>
</g>
</svg>
</template>
<script setup>
defineProps({ id: String, animated: Boolean });
</script>
<style scoped>
svg { width:100%;height:100%;overflow:visible; }
.body,.face,.hand,.hands,.dealt-card,.verdict,.card-back,.anger,.coin,.spark{transform-box:fill-box;transform-origin:center}.animated .body{animation:bad-four 3.5s both}.animated .anger{animation:anger 3.5s both}.animated .dealt-card{animation:deal 3.5s both}.animated .card-back{animation:fan 3.5s both}.animated .hand{animation:hand 3.5s both}.animated .verdict{animation:verdict 3.5s both}.animated.cry-loss .face{animation:sob 3.5s both}.animated .tear{animation:tear 3.5s both}.animated .tear-right{animation-delay:.16s}.animated .lost-chip{animation:lost 3.5s both;transform-box:fill-box;transform-origin:center}.animated.laugh-win .face{animation:laugh 3.5s both}.animated .hands{animation:clap 3.5s both}.animated .coin-left{animation:coin 3.5s both}.animated .coin-right{animation:coin 3.5s .16s both}.animated .spark{animation:spark 3.5s both}
@keyframes bad-four{0%{transform:translateY(-15px) rotate(-8deg);opacity:0}8%{opacity:1;transform:translateY(3px) scale(1.06,.93)}15%{transform:translateY(-2px) scale(.98,1.03)}22%,36%{transform:translateX(-5px) rotate(-5deg)}29%,43%{transform:translateX(5px) rotate(5deg)}55%{transform:rotate(-7deg)}65%,100%{transform:none}}
@keyframes anger{0%,36%{opacity:0;transform:scale(.1)}47%{opacity:1;transform:scale(1.25)}58%,100%{opacity:1;transform:scale(1)}}
@keyframes deal{0%,12%{transform:translate(-9px,13px) rotate(-16deg)}29%{transform:translate(7px,-6px) rotate(9deg)}38%,100%{transform:none}}@keyframes fan{0%,8%{transform:rotate(10deg)}20%,100%{transform:none}}@keyframes hand{0%,10%{transform:rotate(-4deg)}22%{transform:translate(2px,-2px) rotate(4deg)}40%{transform:none}49%{transform:translate(-4px,3px)}60%,100%{transform:none}}@keyframes verdict{0%,32%{opacity:0;transform:scale(1.8) rotate(-18deg)}40%{opacity:1;transform:scale(.88) rotate(5deg)}47%,100%{opacity:1;transform:none}}
@keyframes sob{0%{transform:translateY(-4px)}12%{transform:translateY(3px) scale(1.02,.96)}20%,36%{transform:translate(-2px,2px) rotate(-2deg)}28%,44%{transform:translate(2px,2px) rotate(2deg)}60%,100%{transform:none}}@keyframes tear{0%,11%{opacity:0;transform:translateY(-14px)}19%{opacity:1}29%{opacity:0;transform:translateY(7px)}30%{opacity:0;transform:translateY(-14px)}39%{opacity:1}50%{opacity:0;transform:translateY(7px)}60%,100%{opacity:1;transform:none}}@keyframes lost{0%,28%{opacity:1;transform:translate(0,0) rotate(0deg)}57%,100%{opacity:0;transform:translate(22px,8px) rotate(45deg)}}
@keyframes laugh{0%{transform:translateY(4px) scale(.96)}12%{transform:translateY(-4px) rotate(-8deg)}22%,42%{transform:translateY(-2px) rotate(7deg)}32%,52%{transform:translateY(-3px) rotate(-7deg)}65%,100%{transform:none}}@keyframes clap{0%,12%{transform:none}22%,42%{transform:translateY(-5px) scale(.82,1)}32%,52%{transform:translateY(1px) scale(1.04,1)}64%,100%{transform:none}}@keyframes coin{0%,18%{opacity:0;transform:translateY(50px) scale(.4)}32%{opacity:1;transform:translateY(-6px) rotate(-16deg)}49%,100%{opacity:1;transform:none}}@keyframes spark{0%,27%{opacity:0;transform:scale(.4)}38%{opacity:1;transform:scale(1.3)}52%,100%{opacity:1;transform:none}}

@media(prefers-reduced-motion:reduce) { .animated * { animation:none !important; } }
</style>
