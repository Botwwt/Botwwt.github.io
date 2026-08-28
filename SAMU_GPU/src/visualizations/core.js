import {activateWhenVisible, AnimationController, bindTransport, css, fitCanvas, lerp, roundedRect} from "../animation-controller.js";

const line = (ctx,a,b,color,width=2) => { ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.stroke(); };
const dot = (ctx,x,y,r,color) => { ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); };
const label = (ctx,text,x,y,color=css("--ink"),size=12,align="center") => { ctx.fillStyle=color; ctx.font=`500 ${size}px IBM Plex Mono, monospace`; ctx.textAlign=align; ctx.fillText(text,x,y); };
const box = (ctx,x,y,w,h,text,color,active=false) => { roundedRect(ctx,x,y,w,h,6); ctx.fillStyle=active?`${color}22`:css("--paper"); ctx.fill(); ctx.strokeStyle=active?color:css("--line-strong"); ctx.lineWidth=active?2:1; ctx.stroke(); label(ctx,text,x+w/2,y+h/2+4,active?color:css("--ink"),11); };

export function initArchitecture() {
  const canvas = document.querySelector("#architecture-canvas");
  if (!canvas) return;
  const stages = ["token uₜ","controller c,d","write GEMM","mode transition","state zₜ₊₁","readout"];
  const controller = new AnimationController({steps:stages.length,duration:1250,onFrame:draw});
  function draw(step,p) {
    const {ctx,width:w,height:h}=fitCanvas(canvas); ctx.clearRect(0,0,w,h);
    const xs=[.1,.28,.47,.66,.84,.94].map(v=>v*w), y=h*.5;
    for(let i=0;i<xs.length-1;i++) line(ctx,[xs[i]+36,y],[xs[i+1]-36,y],css("--line-strong"),1.5);
    stages.forEach((s,i)=>box(ctx,xs[i]-34,y-34,68,68,s,i<2?css("--dynamic"):i===2?css("--write"):css("--state"),i===step));
    const next=(step+1)%stages.length, x=lerp(xs[step]+34,xs[next]-34,p); dot(ctx,x,y,6,css("--motion"));
    const M=16; for(let j=0;j<M;j++){const a=j/M*Math.PI*2+performance.now()/2400; const r=52+12*Math.sin(j); const cx=xs[3],cy=y; dot(ctx,cx+Math.cos(a)*r,cy+Math.sin(a)*r,2.3,css("--state"));}
    label(ctx,"dense parallel",xs[2],y-58,css("--muted"),10); label(ctx,"recurrent",xs[3],y+72,css("--muted"),10);
    label(ctx,stages[step],w*.5,h-20,css("--dynamic"),12);
  }
  bindTransport(document.querySelector('[data-controller="architecture"]'),controller);
  activateWhenVisible(canvas.closest(".hero-figure"),controller);
}

const stepInfo = [
  ["uₜ arrives",[["uₜ","[B,d]","BF16","dynamic · token-wise"]]],
  ["controller gives cₜ,dₜ",[["cₜ,dₜ","[B,2]","FP32/BF16","dynamic · token-wise"]]],
  ["write projection computes B uₜ",[["writeₜ","[B,M complex]","BF16","dynamic · mode-wise"]]],
  ["gₜ = exp(cₜ)",[["gₜ","[B,1]","FP32","dynamic · token-wise"]]],
  ["each mode computes ρⱼ",[["ν","[M]","FP32","static · mode-wise"],["ρ","[B,M]","FP32","derived"]]],
  ["phase rotation reconstructed",[["θ","[M]","FP32","static · mode-wise"],["φ","[B,M]","FP32","derived"]]],
  ["old zₜ read",[["stateₜ","[B,M complex]","BF16/FP32","recurrent · mode-wise"]]],
  ["rotation",[["rotated z","[B,M complex]","FP32","lane-local"]]],
  ["decay",[["ρ·R(φ)z","[B,M complex]","FP32","lane-local"]]],
  ["write added",[["writeₜ","[B,M complex]","BF16","dynamic · mode-wise"]]],
  ["zₜ₊₁ produced",[["stateₜ₊₁","[B,M complex]","BF16/FP32","recurrent · mode-wise"]]]
];

export function initStep(root) {
  const canvas=root.querySelector("[data-step-canvas]"), badges=root.querySelector("[data-step-badges]"), status=root.querySelector("[data-step-status]");
  let modes=32, view="full";
  const controller=new AnimationController({steps:11,duration:900,onFrame:draw,onStep:update});
  function update(step){status.textContent=`STEP ${step+1} / 11 · ${stepInfo[step][0]}`; badges.innerHTML=stepInfo[step][1].map(x=>`<span class="data-badge"><b>${x[0]}</b> · ${x[1]} · ${x[2]} · ${x[3]}</span>`).join("");}
  function draw(step,p){
    const {ctx,width:w,height:h}=fitCanvas(canvas);ctx.clearRect(0,0,w,h);
    const dynamic=css("--dynamic"),state=css("--state"),write=css("--write"),muted=css("--muted"),ink=css("--ink");
    const center={x:w*.54,y:h*.53}, R=Math.min(w,h)*.18;
    for(let j=0;j<modes;j++){const a=j/modes*Math.PI*2-.3;const rr=R*(.5+.5*(j+1)/modes);const x=center.x+Math.cos(a)*rr,y=center.y+Math.sin(a)*rr;dot(ctx,x,y,modes===32?2.4:4,state);}
    const nodes=[{x:w*.08,y:h*.25,t:"uₜ",c:dynamic},{x:w*.28,y:h*.25,t:"cₜ,dₜ",c:dynamic},{x:w*.28,y:h*.75,t:"B uₜ",c:write},{x:w*.54,y:h*.18,t:"ρ,φ",c:dynamic},{x:w*.54,y:h*.53,t:"zₜ",c:state},{x:w*.78,y:h*.53,t:"R·ρ·z",c:state},{x:w*.92,y:h*.53,t:"zₜ₊₁",c:state}];
    const links=[[0,1],[0,2],[1,3],[3,4],[4,5],[2,5],[5,6]];links.forEach(([a,b])=>line(ctx,[nodes[a].x,nodes[a].y],[nodes[b].x,nodes[b].y],css("--line-strong"),1));
    nodes.forEach((n,i)=>box(ctx,n.x-31,n.y-21,62,42,n.t,n.c,[0,1,2,3,4,5,5,5,5,5,6][step]===i));
    const activeLink=Math.min(links.length-1,[0,0,1,2,3,3,3,4,4,5,6][step]);const [a,b]=links[activeLink];dot(ctx,lerp(nodes[a].x,nodes[b].x,p),lerp(nodes[a].y,nodes[b].y,p),6,step===2||step===9?write:dynamic);
    ctx.save();ctx.translate(center.x,center.y);const angle=(step>=7?(step-6+p):0)*.32;if(view!=="retention")ctx.rotate(angle);const decay=view==="phase"?1:1-.025*Math.max(0,step-7+p);ctx.scale(decay,decay);ctx.strokeStyle=state;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(R*.82,0);ctx.stroke();dot(ctx,R*.82,0,7,state);ctx.restore();
    if(view==="write"||view==="full"){const alpha=step>=9?1:p*.25;ctx.globalAlpha=alpha;dot(ctx,center.x+R*.55,center.y-R*.35,7,write);ctx.globalAlpha=1;}
    label(ctx,view==="full"?"full update":`${view} isolated`,w-18,20,muted,10,"right"); label(ctx,`${modes} complex modes = ${2*modes} real state scalars`,18,h-18,ink,10,"left");
  }
  root.querySelector("[data-step-modes]").addEventListener("change",e=>modes=+e.target.value);
  root.querySelector("[data-step-view]").addEventListener("change",e=>view=e.target.value);
  bindTransport(root,controller); update(0);
  activateWhenVisible(root,controller);
}

const memoryPaths={
  gemm:{logical:"U + W + write tensor（按 shape 计算）",path:[[.12,.38],[.36,.38],[.82,.26],[.12,.66]]},
  serial:{logical:"write load + state boundary load/store",path:[[.12,.65],[.36,.58],[.57,.66],[.84,.67],[.57,.66]]},
  scan:{logical:"token summaries + prefix intermediates + q",path:[[.12,.58],[.36,.48],[.57,.28],[.57,.66],[.84,.67],[.12,.7]]}
};
export function initMemory(root){
  const stage=root.querySelector("[data-memory-stage]"),select=root.querySelector("[data-memory-track]"),readout=root.querySelector("[data-logical-traffic]");let kind=select.value;
  const particles=Array.from({length:9},()=>{const e=document.createElement("i");e.className="particle";stage.append(e);return e;});
  const controller=new AnimationController({steps:1,duration:1800,onFrame:(_,p)=>{
    const rect=stage.getBoundingClientRect(),path=memoryPaths[kind].path;
    particles.forEach((el,i)=>{const q=(p+i/particles.length)%1,scaled=q*(path.length-1),idx=Math.min(path.length-2,Math.floor(scaled)),t=scaled-idx,a=path[idx],b=path[idx+1];el.style.transform=`translate(${lerp(a[0],b[0],t)*rect.width}px,${lerp(a[1],b[1],t)*rect.height}px)`;});
  }});
  function setKind(){kind=select.value;readout.textContent=memoryPaths[kind].logical;controller.reset();}
  select.addEventListener("change",setKind);root.addEventListener("click",e=>{const a=e.target.dataset.memoryAction;if(a==="toggle"){controller.toggle();e.target.textContent=controller.playing?"Pause":"Play";}if(a==="step")controller.progress=(controller.progress+.15)%1;if(a==="reset")controller.reset();});setKind();
  activateWhenVisible(root,controller);
}

export function initWarp(root){
  const grid=root.querySelector("[data-lane-grid]"),detail=root.querySelector("[data-lane-detail]"),cInput=root.querySelector("[data-warp-c]"),dInput=root.querySelector("[data-warp-d]"),control=root.querySelector("[data-warp-control]");let active=7;
  const values=Array.from({length:32},(_,j)=>({nu:.025+j*.0125,theta:-Math.PI+j*Math.PI*2/32,x:Math.cos(j*.41),y:Math.sin(j*.29),wr:.08*Math.sin(j),wi:.08*Math.cos(j*.7)}));
  grid.innerHTML=values.map((_,j)=>`<button class="lane ${j===active?"active":""}" data-lane="${j}">lane ${j}<br>mode ${j}</button>`).join("");
  function render(){const c=+cInput.value,d=+dInput.value,v=values[active],g=Math.exp(c),rho=Math.exp(-v.nu*g),phi=v.theta+d,co=Math.cos(phi),si=Math.sin(phi),xp=rho*(v.x*co-v.y*si)+v.wr,yp=rho*(v.x*si+v.y*co)+v.wi;control.textContent=`c=${c.toFixed(2)} · d=${d.toFixed(2)} · broadcast → 32 lanes`;detail.innerHTML=`<b>lane ${active} → mode ${active}</b><br><span style="color:var(--static)">ν=${v.nu.toFixed(4)} · θ=${v.theta.toFixed(3)}</span><br><span style="color:var(--state)">x=${v.x.toFixed(3)} · y=${v.y.toFixed(3)}</span><br><span style="color:var(--write)">write_R=${v.wr.toFixed(3)} · write_I=${v.wi.toFixed(3)}</span><hr>g=exp(c)=${g.toFixed(3)}<br>ρ=${rho.toFixed(4)}<br>cosφ=${co.toFixed(4)} · sinφ=${si.toFixed(4)}<hr><b>x′=${xp.toFixed(4)}<br>y′=${yp.toFixed(4)}</b>`;}
  grid.addEventListener("click",e=>{const b=e.target.closest("[data-lane]");if(!b)return;active=+b.dataset.lane;grid.querySelectorAll(".lane").forEach(x=>x.classList.toggle("active",+x.dataset.lane===active));render();document.dispatchEvent(new CustomEvent("samu:context",{detail:{title:`Warp lane ${active}`,lines:[`ν=${values[active].nu.toFixed(4)}`,`ρ=${Math.exp(-values[active].nu*Math.exp(+cInput.value)).toFixed(4)}`]}}));});
  cInput.addEventListener("input",render);dInput.addEventListener("input",render);render();
}

export function initCoherent(root){
  const canvas=root.querySelector("[data-coherent-canvas]"),cInput=root.querySelector("[data-coherent-c]"),dInput=root.querySelector("[data-coherent-d]");let mode="samu";
  const independent=Array.from({length:32},(_,j)=>.18+.74*(.5+.5*Math.sin(j*1.93)));
  function draw(){const {ctx,width:w,height:h}=fitCanvas(canvas);ctx.clearRect(0,0,w,h);const c=+cInput.value,d=+dInput.value,base=h*.78,plotH=h*.55,g=Math.exp(c);
    label(ctx,"retention ρⱼ",20,25,css("--muted"),11,"left");line(ctx,[42,base],[w-25,base],css("--line-strong"));
    for(let j=0;j<32;j++){const nu=.02+j*.013,rho=mode==="samu"?Math.exp(-nu*g):Math.max(.03,Math.min(.99,independent[j]+.11*Math.sin(d+j*.77))),x=50+j*(w-90)/31,y=base-rho*plotH;line(ctx,[x,base],[x,y],mode==="samu"?css("--state"):css("--rglru"),5);dot(ctx,x,y,4,css("--dynamic"));if(j%4===0)label(ctx,`m${j}`,x,base+18,css("--muted"),9);}
    ctx.strokeStyle=css("--motion");ctx.lineWidth=2;ctx.beginPath();for(let j=0;j<32;j++){const x=50+j*(w-90)/31,y=h*.2+Math.sin((-Math.PI+j*Math.PI*2/32)+d)*24;j?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();label(ctx,mode==="samu"?`shared c=${c.toFixed(2)}, d=${d.toFixed(2)}`:"32 conceptual independent gates",w-24,25,css("--dynamic"),11,"right");
  }
  root.addEventListener("click",e=>{if(!e.target.dataset.coherentMode)return;mode=e.target.dataset.coherentMode;root.querySelectorAll("[data-coherent-mode]").forEach(b=>b.classList.toggle("primary",b===e.target));draw();});cInput.addEventListener("input",draw);dInput.addEventListener("input",draw);draw();
}

export function initCoreFigure(root){const type=root.dataset.figure;if(type==="step")initStep(root);if(type==="memory")initMemory(root);if(type==="warp")initWarp(root);if(type==="coherent")initCoherent(root);}
