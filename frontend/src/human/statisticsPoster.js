export const STATISTICS_POSTER_WIDTH = 1600;
export const STATISTICS_POSTER_HEIGHT = 1600;

const FONT = '"Clear Sans", "Microsoft YaHei", Arial, sans-serif';
const COLORS = {
  light: { page:'#f8f3eb', card:'#fffaf3', line:'#d9cfc3', text:'#30241f', muted:'#776d64', header:'#75685f', headerText:'#fffaf3', grid:'#ded5ca' },
  dark: { page:'#1f1f20', card:'#2d2d2c', line:'#4b4946', text:'#fff8ee', muted:'#c3b9ad', header:'#464340', headerText:'#fff9f0', grid:'#45433f' },
};
const FEATURE_COPY = {
  en: { '满盘':'Full board', '二阶满盘':'Second full board', '三阶满盘':'Third full board' },
};
const COPY = {
  zh: { statistics:'玩家统计', games:'按对局数', time:'按时间', pb:'PB', b10:'B10 分数线', rating:'B10 Rating', records:'正式记录', scoreChart:'PB 与 B10 分数线', ratingChart:'B10 Rating 成长', achievements:'终盘成就', inclusive:'包含式计数', rate:'32K 综率成长', reached:'达成局数', stages:'个残局阶段', footer:'生成于 https://play.2048tables.online/' },
  en: { statistics:'PLAYER STATISTICS', games:'BY GAME COUNT', time:'BY TIME', pb:'PB', b10:'B10 SCORE LINE', rating:'B10 RATING', records:'OFFICIAL GAMES', scoreChart:'PB & B10 SCORE LINE', ratingChart:'B10 RATING PROGRESS', achievements:'FINAL-BOARD ACHIEVEMENTS', inclusive:'INCLUSIVE COUNTS', rate:'32K RATE PROGRESS', reached:'GAMES ACHIEVED', stages:'endgame stages', footer:'Generated at https://play.2048tables.online/' },
};

const number = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString('en-US');
const rating = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});
const percent = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value)*100).toFixed(1)}%`;
const numeric = value => value == null ? Number.NaN : Number(value);

function rounded(ctx,x,y,w,h,r,color) {
  ctx.fillStyle=color; ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x,y,w,h,r);
  else { ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); }
  ctx.fill();
}
function text(ctx,value,x,y,size,color,{weight=700,align='left',maxWidth}={}) {
  ctx.fillStyle=color; ctx.font=`${weight} ${size}px ${FONT}`; ctx.textAlign=align; ctx.textBaseline='middle';
  if(maxWidth) ctx.fillText(String(value),x,y,maxWidth); else ctx.fillText(String(value),x,y);
}
function line(ctx,x1,y1,x2,y2,color,width=1) {
  ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
}

export function statisticsChartExtent(points, fields, axis, metric) {
  const xValue = point => axis === 'time' ? numeric(point.ended_at) : numeric(point.game_index);
  const xs=points.map(xValue).filter(Number.isFinite);
  const ys=points.flatMap(point=>fields.map(field=>numeric(point[field]))).filter(Number.isFinite);
  const dataMinX=xs.length?Math.min(...xs):0, dataMaxX=xs.length?Math.max(...xs):1;
  const xSpan=dataMaxX-dataMinX, xPadding=xSpan?xSpan*.025:axis==='time'?86400:1;
  const low=ys.length?Math.min(...ys):0, high=ys.length?Math.max(...ys):1;
  const yPadding=Math.max((high-low)*.1,metric==='rating'?35:Math.max(high*.025,1));
  let minY=metric==='rating'?low-yPadding:Math.max(0,low-yPadding), maxY=high+yPadding;
  if(metric==='rate') {
    const center=(low+high)/2, span=Math.max(.2,(high-low)*1.16);
    minY=Math.max(0,center-span/2); maxY=Math.min(1,center+span/2);
    if(maxY-minY<.2) { if(minY===0) maxY=.2; else minY=maxY-.2; }
  }
  if(maxY===minY) maxY=minY+1;
  return {minX:dataMinX-xPadding,maxX:dataMaxX+xPadding,dataMinX,dataMaxX,minY,maxY};
}

function shortDate(seconds,lang,withTime=false) {
  const options=withTime?{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}:{year:'2-digit',month:'numeric',day:'numeric'};
  return new Date(Number(seconds)*1000).toLocaleString(lang==='en'?'en-US':'zh-CN',options);
}
function tick(value,metric) {
  if(metric==='score') return value>=1e6?`${(value/1e6).toFixed(1)}M`:value>=1000?`${Math.round(value/1000)}k`:String(Math.round(value));
  if(metric==='rate') return `${Math.round(value*100)}%`;
  return String(Math.round(value));
}
function chart(ctx,{x,y,w,h,title,points,axis,fields,labels,colors,metric,palette,lang,summary,note}) {
  rounded(ctx,x,y,w,h,18,palette.card); ctx.strokeStyle=palette.line; ctx.lineWidth=2; ctx.strokeRect(x+.5,y+.5,w-1,h-1);
  text(ctx,title,x+28,y+39,25,palette.text);
  if(summary) text(ctx,summary,x+w-28,y+39,34,colors[0],{align:'right'});
  if(note) text(ctx,note,x+28,y+70,17,palette.muted,{weight:400,maxWidth:w-56});
  const legendY=note?y+99:y+75;
  let legendX=x+28;
  labels.forEach((label,index)=>{line(ctx,legendX,legendY,legendX+25,legendY,colors[index],5);text(ctx,label,legendX+35,legendY,17,palette.muted,{weight:600});legendX+=ctx.measureText(label).width+78;});
  const plot={left:x+90,right:x+w-28,top:legendY+28,bottom:y+h-48};
  if(!points.length) return;
  const extent=statisticsChartExtent(points,fields,axis,metric);
  const xv=point=>axis==='time'?numeric(point.ended_at):numeric(point.game_index);
  const px=value=>plot.left+(value-extent.minX)/(extent.maxX-extent.minX)*(plot.right-plot.left);
  const py=value=>plot.bottom-(value-extent.minY)/(extent.maxY-extent.minY)*(plot.bottom-plot.top);
  for(let i=0;i<5;i++) {
    const f=i/4, value=extent.minY+(extent.maxY-extent.minY)*f, yy=py(value);
    line(ctx,plot.left,yy,plot.right,yy,palette.grid,1);
    text(ctx,tick(value,metric),plot.left-16,yy,16,palette.muted,{weight:500,align:'right'});
  }
  const xFractions=axis==='time'&&extent.dataMaxX-extent.dataMinX<172800?[0,.5,1]:[0,.25,.5,.75,1];
  xFractions.forEach(f=>{
    const xValue=extent.dataMinX+(extent.dataMaxX-extent.dataMinX)*f,xx=px(xValue);
    text(ctx,axis==='time'?shortDate(xValue,lang,extent.dataMaxX-extent.dataMinX<172800):Math.round(xValue),xx,plot.bottom+27,15,palette.muted,{weight:500,align:'center'});
  });
  ctx.save();ctx.beginPath();ctx.rect(plot.left,plot.top,plot.right-plot.left,plot.bottom-plot.top);ctx.clip();
  fields.forEach((field,index)=>{
    ctx.strokeStyle=colors[index];ctx.lineWidth=5;ctx.lineJoin='round';ctx.lineCap='round';ctx.beginPath();let started=false;
    points.forEach(point=>{const value=numeric(point[field]);if(!Number.isFinite(value))return;const xx=px(xv(point)),yy=py(value);if(started)ctx.lineTo(xx,yy);else{ctx.moveTo(xx,yy);started=true;}});ctx.stroke();
    if(metric==='rate') points.forEach(point=>{const value=numeric(point[field]);if(!Number.isFinite(value))return;ctx.fillStyle=colors[index];ctx.beginPath();ctx.arc(px(xv(point)),py(value),5,0,Math.PI*2);ctx.fill();});
  });
  ctx.restore();
}

function statCard(ctx,x,y,w,title,value,accent,palette) {
  rounded(ctx,x,y,w,132,15,palette.card); ctx.fillStyle=accent;ctx.fillRect(x,y,8,132);
  text(ctx,title,x+28,y+34,16,palette.muted); text(ctx,value,x+28,y+88,34,palette.text,{maxWidth:w-50});
}
function wrappedLabel(ctx,label,x,y,width,size,color) {
  const parts=String(label).split(' + '), lines=[];let row='';
  parts.forEach(part=>{const candidate=row?`${row} + ${part}`:part;if(ctx.measureText(candidate).width<=width||!row)row=candidate;else{lines.push(row);row=part;}});if(row)lines.push(row);
  lines.slice(0,2).forEach((value,index)=>text(ctx,value,x,y+(index-(lines.length-1)/2)*size*1.05,size,color,{align:'center',maxWidth:width}));
}
function achievements(ctx,{x,y,w,h,labels,features,palette,copy,lang}) {
  rounded(ctx,x,y,w,h,18,palette.card);ctx.strokeStyle=palette.line;ctx.lineWidth=2;ctx.strokeRect(x+.5,y+.5,w-1,h-1);
  text(ctx,copy.achievements,x+28,y+42,28,palette.text);text(ctx,copy.inclusive,x+w-28,y+42,16,palette.muted,{weight:500,align:'right'});
  const entries=Object.entries(labels), values=entries.map(([key])=>Number(features?.[key]||0)), maximum=Math.max(...values,1);
  const gap=16, innerW=w-56, columnW=(innerW-gap*(entries.length-1))/entries.length, base=y+h-94, maxHeight=h-180;
  entries.forEach(([key,raw],index)=>{
    const value=values[index], cx=x+28+columnW/2+index*(columnW+gap), bh=value?Math.max(8,value/maximum*maxHeight):0;
    text(ctx,number(value),cx,y+87,22,palette.text,{align:'center'});
    rounded(ctx,cx-columnW*.32,base-bh,columnW*.64,bh,7,['#d6b45e','#c99b4c','#b77f45','#9c6742','#825240','#69423a'][index%6]);
    line(ctx,cx-columnW*.4,base,cx+columnW*.4,base,palette.line,2);
    const label=FEATURE_COPY[lang]?.[raw]||raw;ctx.font=`700 15px ${FONT}`;wrappedLabel(ctx,label,cx,base+32,columnW,15,palette.text);
  });
}

export async function drawStatisticsPoster({canvas=document.createElement('canvas'),name,variant,axis='games',summary,series=[],rateSeries=[],featureLabels={},dark=false,language='zh'}) {
  await document.fonts.ready; canvas.width=STATISTICS_POSTER_WIDTH;canvas.height=STATISTICS_POSTER_HEIGHT;
  const ctx=canvas.getContext('2d');if(!ctx)throw new Error('canvas_unavailable');
  const palette=dark?COLORS.dark:COLORS.light,copy=COPY[language]||COPY.zh;
  ctx.fillStyle=palette.page;ctx.fillRect(0,0,canvas.width,canvas.height);
  rounded(ctx,60,50,1480,220,22,palette.header);
  text(ctx,name||copy.statistics,105,115,50,palette.headerText,{maxWidth:660});
  text(ctx,`${variant.replace('x','×')} · ${copy.statistics}`,105,185,30,palette.headerText,{weight:600});
  rounded(ctx,1180,92,290,95,16,dark?'#343231':'#fff3e3');
  text(ctx,axis==='time'?copy.time:copy.games,1325,140,23,dark?palette.headerText:'#3a2d27',{align:'center'});
  const statY=310,statW=350;
  statCard(ctx,60,statY,statW,copy.pb,number(summary?.pb_score),'#c7993d',palette);
  statCard(ctx,437,statY,statW,copy.b10,number(summary?.b10_score),'#b86f4c',palette);
  statCard(ctx,814,statY,statW,copy.rating,rating(summary?.b10_rating),'#5d9a91',palette);
  statCard(ctx,1191,statY,349,copy.records,number(summary?.game_count),'#8c8178',palette);
  chart(ctx,{x:60,y:475,w:725,h:430,title:copy.scoreChart,points:series,axis,fields:['pb_score','b10_score'],labels:[copy.pb,copy.b10],colors:['#c7993d','#b86f4c'],metric:'score',palette,lang:language});
  chart(ctx,{x:815,y:475,w:725,h:430,title:copy.ratingChart,points:series,axis,fields:['b10_rating'],labels:[copy.rating],colors:['#5d9a91'],metric:'rating',palette,lang:language});
  const achievementHeight=455;
  achievements(ctx,{x:60,y:940,w:variant==='4x4'?725:1480,h:achievementHeight,labels:featureLabels,features:summary?.features,palette,copy,lang:language});
  if(variant==='4x4') {
    const rate=summary?.rate_32k||{};
    chart(ctx,{x:815,y:940,w:725,h:455,title:copy.rate,points:rateSeries,axis,fields:['value'],labels:['32K'],colors:['#d2ad5f'],metric:'rate',palette,lang:language,summary:percent(rate.value),note:`${number(rate.passed||0)} / ${number(rate.total||0)} ${copy.stages}`});
  }
  line(ctx,80,1505,1520,1505,palette.line,2);
  text(ctx,copy.footer,1515,1545,21,palette.muted,{weight:400,align:'right'});
  return canvas;
}
