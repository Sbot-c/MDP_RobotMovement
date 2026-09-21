"""Builds the animated arcade arena as a self-contained HTML block for st.components.v1.html."""
import json

from . import room
from .model import state_name

METHODS = [
    dict(id="mdp", name="MDP", short="MDP"),
    dict(id="hj", name="Hooke-Jeeves", short="HJ"),
    dict(id="adp", name="ADP", short="ADP"),
]


def arena_html(runs, bands, values, exit_bonus):
    data = dict(
        poly=room.POLY, block=room.BLOCK, exit=room.EXIT, actions=room.ACTIONS,
        exitBonus=exit_bonus, methods=METHODS,
        values={k: [round(float(v), 3) for v in vals] for k, vals in values.items()},
        names=[state_name(s) for s in range(60)],
        runs={
            k: dict(
                path=[[round(p["x"], 1), round(p["y"], 1), round(p["h"], 1)] for p in r["path"]],
                acts=r["acts"], bumps=r["bumps"], exit=r["exit"], steps=r["steps"],
                cum=[round(c, 2) for c in r["cum"]], good=r["cum_good"],
                sens=[[round(v, 2) for v in sv] for sv in r["sens"]],
                states=[bands.state_of(sv) for sv in r["sens"]],
            ) for k, r in runs.items()
        },
    )
    return TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600&family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
:root{--font-ui:"Bricolage Grotesque",ui-sans-serif,system-ui,sans-serif;--font-pixel:"Press Start 2P",ui-monospace,monospace;
--bg:#e8e2ff;--stage:#f6f3ff;--panel:#fff;--ink:#241a4b;--muted:#5a5091;--line:#cdc2ff;--frame:#3b2d8f;--wall:#3b2d8f;--wall-edge:#8b7bff;
--floor-a:#f6f3ff;--floor-b:#ebe6ff;--mdp:#2b64ff;--hj:#e07800;--adp:#d42a90;--on-c:#fff;--bay-a:#d9f5e6;--bay-b:#b8ead0}
*{box-sizing:border-box}body{margin:0;background:transparent;color:var(--ink);font-family:var(--font-ui);font-size:14px;line-height:1.45;font-variant-numeric:tabular-nums}
button{font:inherit;color:inherit}:focus-visible{outline:3px solid var(--mdp);outline-offset:2px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px;align-items:start;padding:4px 10px 12px 4px}
@media (max-width:760px){.layout{grid-template-columns:1fr}}
.board{position:relative;border:4px solid var(--ink);border-bottom:0;border-radius:6px 6px 0 0;background:var(--stage);overflow:hidden}
svg{display:block;width:100%;height:auto}
.banner{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);max-width:88%;padding:10px 14px;border:3px solid var(--ink);box-shadow:4px 4px 0 var(--ink);background:var(--panel);font-family:var(--font-pixel);font-size:10px;line-height:1.8;text-align:center;opacity:0;pointer-events:none;transition:opacity .25s}
.banner.on{opacity:1}
.tray{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:12px;background:var(--frame);border:4px solid var(--ink);border-top:0;border-radius:0 0 6px 6px}
.lab{color:#fff;font-size:13px}.group{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.sp{flex:1}
.method{--c:var(--ink);display:inline-flex;align-items:center;gap:7px;padding:6px 9px 6px 6px;border:3px solid var(--ink);border-radius:2px;background:var(--panel);font-family:var(--font-pixel);font-size:9px;cursor:pointer}
.method .cap{width:18px;height:18px;border:2px solid var(--ink);border-radius:2px;background:var(--c)}
.method .cap.tri{background:linear-gradient(90deg,var(--mdp) 0 34%,var(--hj) 34% 67%,var(--adp) 67%)}
.method[aria-pressed="true"]{transform:translate(-2px,-2px);box-shadow:4px 4px 0 var(--ink);background:var(--c);color:var(--on-c)}
.method[data-m="all"][aria-pressed="true"]{background:var(--panel);color:var(--ink)}
.ctl{display:inline-flex;align-items:center;gap:6px;padding:7px 10px;border:3px solid var(--ink);border-radius:2px;background:var(--panel);cursor:pointer;font-family:var(--font-pixel);font-size:9px}
.ctl svg{width:12px;height:12px;fill:currentColor}.ctl[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
.card{border:4px solid var(--ink);border-radius:4px;background:var(--panel);padding:14px 16px;box-shadow:6px 6px 0 var(--ink)}
h2{font-family:var(--font-pixel);font-weight:400;font-size:11px;line-height:1.6;margin:0}
.who{margin:4px 0 8px;color:var(--muted)}
.sens{display:grid;grid-template-columns:42px 1fr 52px;gap:8px;align-items:center;margin:5px 0}.sens .val{text-align:right}
.track{height:9px;border-radius:2px;background:var(--line);overflow:hidden}.fill{display:block;height:100%;width:0;background:var(--fc,var(--mdp))}
.stateTag{margin:10px 0 0;font-size:13px;color:var(--muted)}.stateTag b{color:var(--ink)}
.act{display:inline-block;margin:8px 0 6px;padding:7px 9px;border-radius:2px;background:var(--fc,var(--mdp));color:var(--on-c);font-family:var(--font-pixel);font-size:9px}
.log{margin:0;padding:0;list-style:none;font-size:12px;color:var(--muted);min-height:80px}
.log li{display:flex;justify-content:space-between;padding:1px 0;border-bottom:1px dashed var(--line)}.log li:first-child{color:var(--ink)}
.kpis{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;margin:12px 0 0;padding-top:10px;border-top:2px solid var(--line)}
.kpis dt{color:var(--muted);font-size:12px}.kpis dd{margin:0;font-family:var(--font-pixel);font-size:13px;padding-top:3px}.kpis small{font-family:var(--font-ui);font-size:11px;color:var(--muted)}
.wall{fill:none;stroke-linejoin:round}.wa{stroke:var(--wall);stroke-width:20}.wb{stroke:var(--wall-edge);stroke-width:4}
.blockr{fill:var(--wall);stroke:var(--wall-edge);stroke-width:4}.gate{fill:var(--floor-a)}.pix{font-family:var(--font-pixel);font-size:14px;fill:var(--ink)}
.ghost{fill:none;stroke:var(--c);stroke-width:3;opacity:.22;stroke-linejoin:round;stroke-linecap:round;stroke-dasharray:2 8}
.trace{fill:none;stroke:var(--c);stroke-linejoin:round;stroke-linecap:round;stroke-width:6;filter:drop-shadow(0 0 5px var(--c))}
.bot{filter:drop-shadow(0 0 6px var(--c))}.bot .body{fill:var(--c);stroke:var(--ink);stroke-width:3}.bot .wheel{fill:var(--ink)}
.bot .tread{stroke:var(--stage);stroke-width:1.6;stroke-dasharray:2 3;fill:none}.bot .visor{fill:var(--ink)}.bot .eye{fill:#fff}.bot .pupil{fill:#101418}
.bot .ant{stroke:var(--ink);stroke-width:1.6}.bot .led{fill:var(--c);stroke:var(--ink);stroke-width:1}
.bot .nameplate rect{fill:var(--panel);stroke:var(--c);stroke-width:2}.bot .nameplate text{font:400 9px var(--font-pixel);fill:var(--ink);text-anchor:middle}
.sprite{transform-box:fill-box;transform-origin:center;animation:bob .5s ease-in-out infinite alternate}
.moving .tread{animation:tread .3s linear infinite}.led{animation:blink 1s steps(2,jump-none) infinite}
@keyframes bob{to{transform:scale(1.06)}}@keyframes tread{to{stroke-dashoffset:-5}}@keyframes blink{50%{opacity:.25}}
.spark line{stroke:var(--c);stroke-width:3;stroke-linecap:round}.spark{opacity:0;transform-box:fill-box;transform-origin:center}
.spark.pop{animation:pop .6s ease-out}@keyframes pop{0%{opacity:1;transform:scale(.3)}100%{opacity:0;transform:scale(1.8)}}
.portal circle{fill:none;stroke:var(--wall-edge);stroke-width:3;transform-box:fill-box;transform-origin:center;animation:pulse 1.6s ease-in-out infinite}
.portal circle:nth-child(2){animation-delay:.4s}@keyframes pulse{50%{transform:scale(1.25);opacity:.5}}
@media (prefers-reduced-motion:reduce){.sprite,.moving .tread,.led,.portal circle,.spark.pop{animation:none}}
</style></head><body>
<div class="layout">
 <section aria-label="Robot run">
  <div class="board">
   <svg id="map" viewBox="-140 -50 1040 620" role="img" aria-label="Room map with the entrance on the left and the exit bay at the bottom right">
    <defs>
     <pattern id="tiles" width="40" height="40" patternUnits="userSpaceOnUse"><rect width="20" height="20" style="fill:var(--floor-a)"/><rect x="20" width="20" height="20" style="fill:var(--floor-b)"/><rect y="20" width="20" height="20" style="fill:var(--floor-b)"/><rect x="20" y="20" width="20" height="20" style="fill:var(--floor-a)"/></pattern>
     <pattern id="bay" width="20" height="20" patternUnits="userSpaceOnUse"><rect width="10" height="10" style="fill:var(--bay-a)"/><rect x="10" width="10" height="10" style="fill:var(--bay-b)"/><rect y="10" width="10" height="10" style="fill:var(--bay-b)"/><rect x="10" y="10" width="10" height="10" style="fill:var(--bay-a)"/></pattern>
     <clipPath id="roomClip"><path id="roomClipPath"/></clipPath>
    </defs>
    <rect x="-140" y="-50" width="1040" height="620" style="fill:var(--stage)"/>
    <path id="floorTiles" style="fill:url(#tiles);fill-rule:evenodd"/>
    <rect id="bayRect" clip-path="url(#roomClip)" style="fill:url(#bay)"/>
    <path class="wall wa" id="wallA"/><path class="wall wb" id="wallB"/>
    <rect class="blockr" id="block" rx="10"/>
    <rect class="gate" x="-12" y="131" width="24" height="38"/><rect class="gate" x="773" y="463" width="24" height="38"/>
    <g class="portal"><circle cx="-4" cy="150" r="17"/><circle cx="-4" cy="150" r="9"/></g>
    <text class="pix" x="-136" y="156">START</text>
    <path d="M806 505V455" style="stroke:var(--ink);stroke-width:4" fill="none"/>
    <path d="M808 455h34v10h-34zM808 475h34v10h-34z" style="fill:var(--ink)"/>
    <path d="M808 465h34v10h-34zM808 485h34v10h-34z" style="fill:var(--panel);stroke:var(--ink);stroke-width:1.5"/>
    <text class="pix" x="806" y="524">GOAL</text>
    <g id="ghosts"></g><g id="traces"></g><g id="bots"></g><g id="sparks"></g>
   </svg>
   <div class="banner" id="banner" role="status" aria-live="polite"></div>
  </div>
  <div class="tray">
   <span class="lab">Method</span><div class="group" id="methodBtns" role="group" aria-label="Method filter"></div>
   <span class="sp"></span>
   <div class="group">
    <button type="button" class="ctl" id="playBtn"><svg viewBox="0 0 14 14" aria-hidden="true"><path id="playIcon" d="M2 1h4v12H2zM8 1h4v12H8z"/></svg><span id="playTxt">Pause</span></button>
    <button type="button" class="ctl" id="restartBtn"><svg viewBox="0 0 14 14" aria-hidden="true"><path d="M7 1.5a5.5 5.5 0 1 0 5.2 3.7l-1.9.6A3.5 3.5 0 1 1 7 3.5V6l4-3.2L7 0z"/></svg>Restart</button>
    <span class="group" role="group" aria-label="Speed" id="speedBtns"></span>
   </div>
  </div>
 </section>
 <aside class="card" aria-label="Live readout">
  <h2>Live readout</h2><p class="who" id="who"></p><div id="sensors"></div>
  <p class="stateTag" id="stateTag"></p><span class="act" id="actNow"></span>
  <ul class="log" id="log"></ul>
  <dl class="kpis"><div><dt>Steps</dt><dd id="kSteps">0</dd></div><div><dt>Wall bumps</dt><dd id="kBumps">0</dd></div>
  <div><dt>On-track share</dt><dd id="kWall">0%</dd></div><div><dt>Return so far</dt><dd id="kRet">0</dd></div></dl>
 </aside>
</div>
<script>
(function(){
var D=__DATA__;
var NS='http://www.w3.org/2000/svg',M=D.methods,ICON=['\u2191','\u2197','\u2192','\u2196'],RATE=9;
function $(i){return document.getElementById(i)}
function el(t,a,p){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function dp(pts){return 'M'+pts.map(function(p){return p[0]+' '+p[1]}).join('L')+'Z'}
var B=D.block,bp=[[B.x,B.y],[B.x+B.w,B.y],[B.x+B.w,B.y+B.h],[B.x,B.y+B.h]];
$('floorTiles').setAttribute('d',dp(D.poly)+dp(bp));$('roomClipPath').setAttribute('d',dp(D.poly));
$('wallA').setAttribute('d',dp(D.poly));$('wallB').setAttribute('d',dp(D.poly));
var bk=$('block');bk.setAttribute('x',B.x);bk.setAttribute('y',B.y);bk.setAttribute('width',B.w);bk.setAttribute('height',B.h);
var X=D.exit,br=$('bayRect');br.setAttribute('x',X.x);br.setAttribute('y',X.y);br.setAttribute('width',X.w);br.setAttribute('height',X.h);
function pts(path,n){var o=[];for(var i=0;i<n;i++)o.push(path[i][0]+','+path[i][1]);return o.join(' ')}
M.forEach(function(m){
  var id=m.id,c='var(--'+id+')',r=D.runs[id];
  var g=el('polyline',{'class':'ghost',id:'ghost-'+id,points:pts(r.path,r.path.length)},$('ghosts'));g.style.setProperty('--c',c);
  var t=el('polyline',{'class':'trace',id:'trace-'+id},$('traces'));t.style.setProperty('--c',c);
  var bot=el('g',{'class':'bot',id:'bot-'+id},$('bots'));bot.style.setProperty('--c',c);
  bot.innerHTML='<g><g class="sprite"><rect class="wheel" x="-10" y="-18" width="20" height="7" rx="3"/><rect class="wheel" x="-10" y="11" width="20" height="7" rx="3"/>'+
   '<path class="tread" d="M-8 -14.5H8M-8 14.5H8"/><circle class="body" r="13"/><rect class="visor" x="1" y="-8.5" width="11" height="17" rx="5.5"/>'+
   '<circle class="eye" cx="7.5" cy="-3.4" r="2.4"/><circle class="eye" cx="7.5" cy="3.4" r="2.4"/><circle class="pupil" cx="8.4" cy="-3.4" r="1.1"/><circle class="pupil" cx="8.4" cy="3.4" r="1.1"/>'+
   '<line class="ant" x1="-8" y1="0" x2="-15" y2="0"/><circle class="led" cx="-16.5" cy="0" r="2.4"/></g></g>'+
   '<g class="nameplate" transform="translate(0,'+(id==='adp'?40:-40)+')"><rect x="-27" y="-13" width="54" height="21" rx="2"/><text y="2">'+m.short+'</text></g>';
  var so=el('g',{id:'spark-'+id},$('sparks')),s=el('g',{'class':'spark'},so);s.style.setProperty('--c',c);
  for(var k=0;k<8;k++){var a=k*Math.PI/4;el('line',{x1:Math.cos(a)*10,y1:Math.sin(a)*10,x2:Math.cos(a)*22,y2:Math.sin(a)*22},s)}
});
var mode='all',speed=1,reduce=window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches,playing=!reduce,t=0,last=performance.now(),st={},fin=[];
function reset(){M.forEach(function(m){st[m.id]={k:0,done:false,lastK:0}});t=0;fin=[];$('banner').classList.remove('on')}
function vis(){return mode==='all'?M:M.filter(function(m){return m.id===mode})}
M.concat([{id:'all',name:'All three'}]).forEach(function(m){
  var b=document.createElement('button');b.type='button';b.className='method';b.dataset.m=m.id;
  b.style.setProperty('--c',m.id==='all'?'var(--ink)':'var(--'+m.id+')');
  b.innerHTML='<span class="cap'+(m.id==='all'?' tri':'')+'"></span>'+m.name;
  b.onclick=function(){mode=m.id;reset();playing=true;sync()};$('methodBtns').appendChild(b);
});
[1,2,4,8].forEach(function(s){var b=document.createElement('button');b.type='button';b.className='ctl';b.textContent=s+'x';b.dataset.s=s;b.onclick=function(){speed=s;sync()};$('speedBtns').appendChild(b)});
[['Front'],['Left'],['Right'],['Back']].forEach(function(p,i){var d=document.createElement('div');d.className='sens';d.id='s'+i;d.innerHTML='<b>'+p[0]+'</b><span class="track"><span class="fill"></span></span><span class="val"></span>';$('sensors').appendChild(d)});
function sync(){
  document.querySelectorAll('.method').forEach(function(b){b.setAttribute('aria-pressed',String(b.dataset.m===mode))});
  document.querySelectorAll('#speedBtns .ctl').forEach(function(b){b.setAttribute('aria-pressed',String(+b.dataset.s===speed))});
  M.forEach(function(m){var on=mode==='all'||mode===m.id;['ghost-','trace-','bot-'].forEach(function(p){$(p+m.id).style.display=on?'':'none'})});
  $('playTxt').textContent=playing?'Pause':'Play';$('playIcon').setAttribute('d',playing?'M2 1h4v12H2zM8 1h4v12H8z':'M3 1l9 6-9 6z');
}
$('playBtn').onclick=function(){if(vis().every(function(m){return st[m.id].done})){reset();playing=true}else playing=!playing;sync()};
$('restartBtn').onclick=function(){reset();playing=true;sync()};
function an(d){return ((d+540)%360)-180}
function pop(id,x,y){var o=$('spark-'+id),s=o.firstChild;o.setAttribute('transform','translate('+x+' '+y+')');s.classList.remove('pop');void s.getBoundingClientRect();s.classList.add('pop')}
function tick(ts){
  requestAnimationFrame(tick);
  var dt=Math.max(0,Math.min(0.1,(ts-last)/1000));last=ts;if(playing)t+=dt*RATE*speed;
  var V=vis(),lead=null,lf=-1,moving=false;
  V.forEach(function(m){
    var id=m.id,r=D.runs[id],s=st[id],N=r.steps,k=Math.max(0,Math.min(t,N)),i=Math.min(Math.floor(k),Math.max(0,N-1)),f=N?k-i:0;s.k=k;
    var a=r.path[i],b=r.path[Math.min(i+1,r.path.length-1)],x=a[0]+(b[0]-a[0])*f,y=a[1]+(b[1]-a[1])*f,ang=a[2]+an(b[2]-a[2])*f;
    var bot=$('bot-'+id);bot.setAttribute('transform','translate('+x.toFixed(1)+' '+y.toFixed(1)+')');
    bot.firstChild.setAttribute('transform','rotate('+ang.toFixed(1)+') scale(1.4)');bot.classList.toggle('moving',playing&&k<N);
    $('trace-'+id).setAttribute('points',pts(r.path,i+1)+' '+x.toFixed(1)+','+y.toFixed(1));
    r.bumps.forEach(function(bi){if(s.lastK<bi&&k>=bi)pop(id,r.path[bi][0],r.path[bi][1])});
    if(!s.done&&k>=N){s.done=true;fin.push(id);if(r.exit)pop(id,x,y)}
    s.lastK=k;var fr=N?k/N:1;if(!s.done&&fr>lf){lf=fr;lead=m}if(!s.done)moving=true;
  });
  if(!lead)lead=fin.length?M.filter(function(m){return m.id===fin[0]})[0]:V[0];
  readout(lead);
  var bn=$('banner');
  if(!moving&&V.every(function(m){return st[m.id].done})){
    if(!bn.classList.contains('on')){
      var ok=V.filter(function(m){return D.runs[m.id].exit}).sort(function(a,b){return D.runs[a.id].steps-D.runs[b.id].steps}),msg;
      if(!ok.length)msg='GAME OVER. No robot reached the exit in 600 steps.';
      else{var w=ok[0],r=D.runs[w.id],bb=r.bumps.length;
        msg='STAGE CLEAR! '+w.name+(V.length>1?' reached the exit first, in ':' reached the exit in ')+r.steps+' steps'+(bb?' with '+bb+(bb>1?' bumps.':' bump.'):' with no bumps.');
        var miss=V.filter(function(m){return !D.runs[m.id].exit});if(miss.length)msg+=' '+miss.map(function(m){return m.name}).join(' and ')+' did not make it.'}
      bn.textContent=msg;bn.classList.add('on');
    }
    if(playing){playing=false;sync()}
  }else bn.classList.remove('on');
}
var logKey='';
function readout(m){
  var id=m.id,r=D.runs[id],s=st[id],N=r.steps,k=s.k,i=Math.min(Math.floor(k),Math.max(0,N-1)),si=Math.min(Math.round(k),r.sens.length-1);
  var sv=r.sens[si],c='var(--'+id+')',sid=r.states[si],nm=D.names[sid];
  $('who').textContent=(mode==='all'?'Following the leader: ':'Following ')+m.name;
  sv.forEach(function(v,j){var row=$('s'+j),f=row.querySelector('.fill');f.style.width=Math.min(100,v/3*100).toFixed(0)+'%';f.style.setProperty('--fc',c);row.querySelector('.val').textContent=v.toFixed(2)+' m'});
  var vt=D.values[id]?' <b>'+(id==='mdp'?'V*(s)':'V\u0302(s)')+' = '+D.values[id][sid].toFixed(2)+'</b>':'';
  $('stateTag').innerHTML='State <b>#'+sid+'</b>: front '+nm[0]+', left '+nm[1]+', right '+nm[2]+', back '+nm[3]+'.'+vt;
  var a=r.acts[Math.min(i,r.acts.length-1)],ac=$('actNow');ac.textContent=ICON[a]+' '+D.actions[a];ac.style.setProperty('--fc',c);
  var key=id+':'+i;if(key!==logKey){logKey=key;var h='';for(var j=i;j>i-5&&j>=0;j--)h+='<li><span>'+D.actions[r.acts[j]]+'</span><span>step '+(j+1)+'</span></li>';$('log').innerHTML=h}
  var up=Math.floor(k),ret=r.cum[up],bumps=r.bumps.filter(function(b){return k>=b}).length;if(s.done&&r.exit)ret+=D.exitBonus;
  $('kSteps').innerHTML=up+'<small> / '+N+'</small>';$('kBumps').textContent=bumps;
  $('kWall').textContent=(up?(100*r.good[up]/up).toFixed(0):0)+'%';$('kRet').textContent=ret.toFixed(1);
}
reset();sync();requestAnimationFrame(tick);
})();
</script></body></html>"""
