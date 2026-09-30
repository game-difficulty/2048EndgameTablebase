import { ref, watch } from 'vue';
import { messages } from './translations.js';
import { extraMessages } from './translationsExtra.js';
import { projectMessages } from './projectTranslations.js';
import { errorTranslations } from './translationsErrors.js';

const KEY='competition-language';
function savedOverride(){
 try{const value=localStorage.getItem(KEY);return ['en','zh'].includes(value)?value:null;}catch{return null;}
}
export const languageOverride=ref(savedOverride());
let upstreamLanguage=null;
let languageRequest=0;
let preferenceReader=null;
function browserLanguage(){return typeof navigator!=='undefined' && navigator.language?.toLowerCase().startsWith('zh')?'zh':'en';}
function initialLanguage(){
 if(typeof window==='undefined')return 'zh';
 const query=new URLSearchParams(window.location.search).get('lang');
 if(['en','zh'].includes(query))return query;
 return languageOverride.value || browserLanguage();
}
export const language=ref(initialLanguage());
const dictionary={...messages,...extraMessages,...projectMessages,...errorTranslations};
export function t(value){
 if(language.value!=='en' || typeof value!=='string')return value;
 const text=value.trim();
 if(Object.hasOwn(dictionary,text))return value.replace(text,dictionary[text]);
 for(const [pattern,replace] of patterns){if(pattern.test(text))return text.replace(pattern,replace);}
 return value;
}
// Parameterized display messages: identifiers and player/team names stay intact.
const patterns=[
 [/^请求失败（(\d+)）$/,'Request failed ($1)'],
 [/^确定关闭「(.*)」？关闭后无法重新落座或开赛，房间记录仍会保留。$/s,'Close “$1”? Seating and play will be disabled. The room record will be retained.'],
 [/^本机 AI 计算失败，请重试：(.*)$/s,'Local AI calculation failed. Please retry: $1'],
 [/^(.+)获得先手$/,'$1 gets first pick'],
 [/^(.+) 将首先选择项目 A 并 BAN 一个项目$/,'$1 will pick Game A and ban one game first'],
 [/^规则版本 (.+)$/,'Rules version $1'],
 [/^我的剩余 (.+)$/,'Your time remaining: $1'],
 [/^休整 (.+) 秒$/,'Intermission: $1 seconds'],
 [/^用时 (.+) 秒$/,'Time: $1 s'],
 [/^送出 (.+) 块$/,'Delivered: $1'],
 [/^成绩 (.+)$/,'Result: $1'],
 [/^盘面和 (.+)$/,'Tile sum: $1'],
 [/^([\d.,]+) 块$/,'$1 pieces'],
 [/^([\d.,]+) 分$/,'$1 points'],
 [/^(\d+) 个比赛房间$/,'$1 match rooms'],
 [/^赛事方 #(\d+)$/,'Staff #$1'],
 [/^Table 用户 ID (\d+)$/,'Table user ID $1'],
 [/^([黄白])([123])$/ ,(_,side,n)=>`${side==='黄'?'Yellow':'White'} ${n}`],
 [/^(\d+) 号位$/,'Seat $1'],
 [/^(\d+)号 ·$/,'$1 ·'],
 [/^上移 (.+)$/,'Move $1 up'],[/^下移 (.+)$/,'Move $1 down'],
 [/^关闭比赛房间 (.+)$/,'Close match room $1'],
 [/^(.+)获胜$/,'$1 wins'],[/^(.+)胜$/,'$1 wins'],
 [/^(.+)本场数据$/,'$1 game statistics'],
 [/^(.+)真·华容道棋盘$/,'$1 Huarong Dao board'],[/^(.+)越来越大棋盘$/,'$1 Getting Bigger board'],[/^(.+)项目棋盘$/,'$1 game board'],
 [/^([黄白])方局分 (.+)$/,(_,s,n)=>`${s==='黄'?'Yellow':'White'} game score ${n}`],
 [/^第 (\d+) 行有误：.*$/,'Invalid row $1: user ID, team name (optional), guest 0/1, captain 0/1, team position (1/2/3).'],
 [/^(.+)已入选(\d+)局，共需25局$/,'$1: $2 of 25 games selected'],
 [/^(\d+) 号 · (.+)（(已签到|未到场)）$/,(_,n,name,state)=>`Seat ${n} · ${name} (${state==='已签到'?'Checked in':'Absent'})`],
];
export function setLanguage(value){
 if(!['en','zh','auto'].includes(value))return;
 languageOverride.value=value==='auto'?null:value;
 language.value=languageOverride.value || upstreamLanguage || browserLanguage();
 try{if(languageOverride.value)localStorage.setItem(KEY,value);else localStorage.removeItem(KEY);}catch{}
 if(typeof window!=='undefined'){
  const url=new URL(window.location.href);
  if(url.searchParams.has('lang')){if(value==='auto')url.searchParams.delete('lang');else url.searchParams.set('lang',value);window.history.replaceState(window.history.state,'',url);}
 }
 if(value==='auto')void syncAccountLanguage();
}
export async function syncAccountLanguage(reader=preferenceReader){
 if(!reader)return;
 preferenceReader=reader;
 const request=++languageRequest;
 try{
  const result=await reader();
  if(request!==languageRequest)return;
  upstreamLanguage=['zh','en'].includes(result?.language)?result.language:null;
  const explicit=typeof window!=='undefined' && ['zh','en'].includes(new URLSearchParams(window.location.search).get('lang'));
  if(!languageOverride.value && !explicit)language.value=upstreamLanguage || browserLanguage();
 }catch{ /* Keep the current language when preferences cannot be read. */ }
}
if(typeof document!=='undefined')watch(language,value=>{
 document.documentElement.lang=value==='en'?'en':'zh-CN';
 document.title=value==='en'?'2048 Competition Center':'2048 赛事中心';
},{immediate:true});
