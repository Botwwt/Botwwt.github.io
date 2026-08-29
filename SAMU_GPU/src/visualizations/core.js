import {activateWhenVisible, AnimationController, bindTransport, css, fitCanvas, lerp, roundedRect} from "../animation-controller.js";

const line = (ctx,a,b,color,width=2) => { ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.stroke(); };
const dot = (ctx,x,y,r,color) => { ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); };
const label = (ctx,text,x,y,color=css("--ink"),size=12,align="center") => { ctx.fillStyle=color; ctx.font=`500 ${size}px IBM Plex Mono, monospace`; ctx.textAlign=align; ctx.fillText(text,x,y); };
const box = (ctx,x,y,w,h,text,color,active=false) => { roundedRect(ctx,x,y,w,h,6); ctx.fillStyle=active?`${color}22`:css("--paper"); ctx.fill(); ctx.strokeStyle=active?color:css("--line-strong"); ctx.lineWidth=active?2:1; ctx.stroke(); label(ctx,text,x+w/2,y+h/2+4,active?color:css("--ink"),12); };
const arrow = (ctx,a,b,color,width=1.5) => { line(ctx,a,b,color,width); const angle=Math.atan2(b[1]-a[1],b[0]-a[0]),size=7; ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0]-Math.cos(angle-.5)*size,b[1]-Math.sin(angle-.5)*size);ctx.lineTo(b[0]-Math.cos(angle+.5)*size,b[1]-Math.sin(angle+.5)*size);ctx.closePath();ctx.fillStyle=color;ctx.fill(); };
const card = (ctx,x,y,w,h,title,subtitle,color,active=false) => { roundedRect(ctx,x,y,w,h,8);ctx.fillStyle=active?`${color}18`:css("--paper");ctx.fill();ctx.strokeStyle=active?color:css("--line-strong");ctx.lineWidth=active?2:1;ctx.stroke();label(ctx,title,x+w/2,y+h*.43,active?color:css("--ink"),13);label(ctx,subtitle,x+w/2,y+h*.69,css("--muted"),10); };

export function initArchitecture() {
  const canvas = document.querySelector("#architecture-canvas");
  if (!canvas) return;
  const explainer=document.querySelector("#architecture-explainer");
  const stages = [
    "当前 token 的向量 uₜ 进入同一次打包投影。",
    "投影分成两条支路：2 个共享控制量，以及 2M 个实数构成的复数写入。",
    "静态参数 νⱼ、θⱼ 把共享 cₜ、dₜ 转换成各模态不同的衰减 ρⱼ,ₜ 和相位 φⱼ,ₜ。",
    "各模态并行更新；同一模态内部仍必须按时间顺序读取旧状态 zₜ。",
    "得到新状态 zₜ₊₁；解码时它会作为下一步的固定大小缓存。"
  ];
  const controller = new AnimationController({steps:stages.length,duration:1800,onFrame:draw,onStep:update});
  function update(step){if(explainer)explainer.textContent=stages[step];}
  function draw(step,p) {
    const {ctx,width:w,height:h}=fitCanvas(canvas); ctx.clearRect(0,0,w,h);
    const ink=css("--ink"),muted=css("--muted"),lineColor=css("--line-strong"),dynamic=css("--dynamic"),write=css("--write"),state=css("--state"),motion=css("--motion");
    const input={x:w*.04,y:h*.40,w:w*.14,h:h*.20};
    const controls={x:w*.26,y:h*.16,w:w*.20,h:h*.20};
    const writeNode={x:w*.26,y:h*.64,w:w*.20,h:h*.20};
    const transition={x:w*.56,y:h*.26,w:w*.25,h:h*.48};
    const output={x:w*.86,y:h*.40,w:w*.11,h:h*.20};
    arrow(ctx,[input.x+input.w,input.y+input.h*.42],[controls.x,controls.y+controls.h*.55],lineColor);
    arrow(ctx,[input.x+input.w,input.y+input.h*.58],[writeNode.x,writeNode.y+writeNode.h*.45],lineColor);
    arrow(ctx,[controls.x+controls.w,controls.y+controls.h*.55],[transition.x,transition.y+transition.h*.28],lineColor);
    arrow(ctx,[writeNode.x+writeNode.w,writeNode.y+writeNode.h*.45],[transition.x,transition.y+transition.h*.72],lineColor);
    arrow(ctx,[transition.x+transition.w,transition.y+transition.h*.5],[output.x,output.y+output.h*.5],lineColor);
    arrow(ctx,[transition.x+transition.w*.5,h*.93],[transition.x+transition.w*.5,transition.y+transition.h],state,1.5);
    arrow(ctx,[transition.x+transition.w*.5,h*.08],[transition.x+transition.w*.5,transition.y],css("--static"),1.5);
    card(ctx,input.x,input.y,input.w,input.h,"输入 uₜ","[B,d] · BF16",dynamic,step===0);
    card(ctx,controls.x,controls.y,controls.w,controls.h,"共享控制 cₜ, dₜ","每个 token 只有 2 个",dynamic,step===1||step===2);
    card(ctx,writeNode.x,writeNode.y,writeNode.w,writeNode.h,"复数写入 wₜ","[B,M complex]",write,step===1);
    card(ctx,output.x,output.y,output.w,output.h,"新状态","zₜ₊₁",state,step===4);
    roundedRect(ctx,transition.x,transition.y,transition.w,transition.h,9);ctx.fillStyle=step===3?`${state}16`:css("--paper");ctx.fill();ctx.strokeStyle=step===2?dynamic:step===3?state:lineColor;ctx.lineWidth=step===2||step===3?2:1;ctx.stroke();
    label(ctx,"并行模态更新",transition.x+transition.w*.5,transition.y+24,ink,14);
    label(ctx,"ρⱼ,ₜ · R(φⱼ,ₜ) · zⱼ,ₜ + wⱼ,ₜ",transition.x+transition.w*.5,transition.y+44,muted,10);
    const rows=5,rowLeft=transition.x+16,rowRight=transition.x+transition.w-16;
    for(let j=0;j<rows;j++){const y=transition.y+68+j*(transition.h-88)/(rows-1);line(ctx,[rowLeft,y],[rowRight,y],css("--line"),1);dot(ctx,rowLeft+16,y,3.2,dynamic);dot(ctx,lerp(rowLeft+24,rowRight-15,(p+j*.17)%1),y,4,step===3?state:motion);label(ctx,`模态 ${j}`,rowLeft+26,y-7,muted,9,"left");}
    label(ctx,"静态谱 νⱼ, θⱼ",transition.x+transition.w*.5,h*.055,css("--static"),11);
    label(ctx,"旧状态 zₜ（固定大小缓存）",transition.x+transition.w*.5,h*.975,state,11);
    const activePath=[[[input.x+input.w,input.y+input.h*.5],[controls.x,controls.y+controls.h*.55]],[[input.x+input.w,input.y+input.h*.5],[writeNode.x,writeNode.y+writeNode.h*.45]],[[controls.x+controls.w,controls.y+controls.h*.55],[transition.x,transition.y+transition.h*.28]],[[transition.x+12,transition.y+transition.h*.5],[transition.x+transition.w-12,transition.y+transition.h*.5]],[[transition.x+transition.w,transition.y+transition.h*.5],[output.x,output.y+output.h*.5]]][step];
    dot(ctx,lerp(activePath[0][0],activePath[1][0],p),lerp(activePath[0][1],activePath[1][1],p),5,step===1?write:step===3||step===4?state:motion);
  }
  bindTransport(document.querySelector('[data-controller="architecture"]'),controller,{play:"播放",pause:"暂停"});
  update(0);
  activateWhenVisible(canvas.closest(".hero-figure"),controller);
}

const stepInfo = [
  ["读取当前输入 uₜ",[["uₜ","[B,d]","BF16","每个 token 一组"]]],
  ["控制器生成 cₜ、dₜ",[["cₜ,dₜ","[B,2]","FP32/BF16","每个 token 共享"]]],
  ["写入投影计算 B uₜ",[["写入 wₜ","[B,M 个复数]","BF16","随模态变化"]]],
  ["计算 gₜ = exp(cₜ)",[["gₜ","[B,1]","FP32","每个 token 共享"]]],
  ["各模态计算保留率 ρⱼ",[["ν","[M]","FP32","静态模态参数"],["ρ","[B,M]","FP32","由公式得到"]]],
  ["重建各模态的旋转相位",[["θ","[M]","FP32","静态模态参数"],["φ","[B,M]","FP32","由公式得到"]]],
  ["读取旧状态 zₜ",[["旧状态 zₜ","[B,M 个复数]","BF16/FP32","随模态变化"]]],
  ["旋转旧状态",[["旋转后的 z","[B,M 个复数]","FP32","线程局部值"]]],
  ["乘以保留率",[["ρ·R(φ)z","[B,M 个复数]","FP32","线程局部值"]]],
  ["加入当前写入",[["写入 wₜ","[B,M 个复数]","BF16","随模态变化"]]],
  ["得到新状态 zₜ₊₁",[["新状态 zₜ₊₁","[B,M 个复数]","BF16/FP32","随模态变化"]]]
];

export function initStep(root) {
  const canvas=root.querySelector("[data-step-canvas]"), badges=root.querySelector("[data-step-badges]"), status=root.querySelector("[data-step-status]");
  let modes=32, view="full";
  const controller=new AnimationController({steps:11,duration:900,onFrame:draw,onStep:update});
  function update(step){status.textContent=`第 ${step+1} / 11 步 · ${stepInfo[step][0]}`; badges.innerHTML=stepInfo[step][1].map(x=>`<span class="data-badge"><b>${x[0]}</b> · ${x[1]} · ${x[2]} · ${x[3]}</span>`).join("");}
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
    const viewLabel={full:"完整更新",retention:"只看保留率",phase:"只看相位",write:"只看写入"}[view];
    label(ctx,viewLabel,w-18,20,muted,10,"right"); label(ctx,`${modes} 个复数模态 = ${2*modes} 个实数状态量`,18,h-18,ink,10,"left");
  }
  root.querySelector("[data-step-modes]").addEventListener("change",e=>modes=+e.target.value);
  root.querySelector("[data-step-view]").addEventListener("change",e=>view=e.target.value);
  bindTransport(root,controller); update(0);
  activateWhenVisible(root,controller);
}

const memoryPaths={
  gemm:{logical:"读取输入 U 与权重 W，写出当前写入向量（按张量形状计算）",path:[[.12,.38],[.36,.38],[.82,.26],[.12,.66]]},
  serial:{logical:"读取写入向量，并在递推区间边界读写状态",path:[[.12,.65],[.36,.58],[.57,.66],[.84,.67],[.57,.66]]},
  scan:{logical:"读写分块摘要、前缀中间量和零状态输出 q",path:[[.12,.58],[.36,.48],[.57,.28],[.57,.66],[.84,.67],[.12,.7]]}
};
export function initMemory(root){
  const stage=root.querySelector("[data-memory-stage]"),select=root.querySelector("[data-memory-track]"),readout=root.querySelector("[data-logical-traffic]");let kind=select.value;
  const particles=Array.from({length:9},()=>{const e=document.createElement("i");e.className="particle";stage.append(e);return e;});
  const controller=new AnimationController({steps:1,duration:1800,onFrame:(_,p)=>{
    const rect=stage.getBoundingClientRect(),path=memoryPaths[kind].path;
    particles.forEach((el,i)=>{const q=(p+i/particles.length)%1,scaled=q*(path.length-1),idx=Math.min(path.length-2,Math.floor(scaled)),t=scaled-idx,a=path[idx],b=path[idx+1];el.style.transform=`translate(${lerp(a[0],b[0],t)*rect.width}px,${lerp(a[1],b[1],t)*rect.height}px)`;});
  }});
  function setKind(){kind=select.value;readout.textContent=memoryPaths[kind].logical;controller.reset();}
  select.addEventListener("change",setKind);root.addEventListener("click",e=>{const a=e.target.dataset.memoryAction;if(a==="toggle"){controller.toggle();e.target.textContent=controller.playing?"暂停":"播放";}if(a==="step")controller.progress=(controller.progress+.15)%1;if(a==="reset")controller.reset();});setKind();
  activateWhenVisible(root,controller);
}

export function initWarp(root){
  const grid=root.querySelector("[data-lane-grid]"),detail=root.querySelector("[data-lane-detail]"),cInput=root.querySelector("[data-warp-c]"),dInput=root.querySelector("[data-warp-d]"),control=root.querySelector("[data-warp-control]");let active=7;
  const values=Array.from({length:32},(_,j)=>({nu:.025+j*.0125,theta:-Math.PI+j*Math.PI*2/32,x:Math.cos(j*.41),y:Math.sin(j*.29),wr:.08*Math.sin(j),wi:.08*Math.cos(j*.7)}));
  grid.innerHTML=values.map((_,j)=>`<button class="lane ${j===active?"active":""}" data-lane="${j}">线程 ${j}<br>模态 ${j}</button>`).join("");
  function render(){const c=+cInput.value,d=+dInput.value,v=values[active],g=Math.exp(c),rho=Math.exp(-v.nu*g),phi=v.theta+d,co=Math.cos(phi),si=Math.sin(phi),xp=rho*(v.x*co-v.y*si)+v.wr,yp=rho*(v.x*si+v.y*co)+v.wi;control.textContent=`c=${c.toFixed(2)} · d=${d.toFixed(2)} · 广播到 32 个线程`;detail.innerHTML=`<b>线程 ${active} → 模态 ${active}</b><br><span style="color:var(--static)">ν=${v.nu.toFixed(4)} · θ=${v.theta.toFixed(3)}</span><br><span style="color:var(--state)">旧状态：x=${v.x.toFixed(3)} · y=${v.y.toFixed(3)}</span><br><span style="color:var(--write)">写入：wᴿ=${v.wr.toFixed(3)} · wᴵ=${v.wi.toFixed(3)}</span><hr>g=exp(c)=${g.toFixed(3)}<br>ρ=${rho.toFixed(4)}<br>cosφ=${co.toFixed(4)} · sinφ=${si.toFixed(4)}<hr><b>新状态：x′=${xp.toFixed(4)}<br>y′=${yp.toFixed(4)}</b>`;}
  grid.addEventListener("click",e=>{const b=e.target.closest("[data-lane]");if(!b)return;active=+b.dataset.lane;grid.querySelectorAll(".lane").forEach(x=>x.classList.toggle("active",+x.dataset.lane===active));render();document.dispatchEvent(new CustomEvent("samu:context",{detail:{title:`线程束中的线程 ${active}`,lines:[`ν=${values[active].nu.toFixed(4)}`,`ρ=${Math.exp(-values[active].nu*Math.exp(+cInput.value)).toFixed(4)}`]}}));});
  cInput.addEventListener("input",render);dInput.addEventListener("input",render);render();
}

export function initCoherent(root){
  const canvas=root.querySelector("[data-coherent-canvas]"),cInput=root.querySelector("[data-coherent-c]"),dInput=root.querySelector("[data-coherent-d]"),readout=root.querySelector("[data-coherent-readout]");
  const modes=Array.from({length:12},(_,j)=>({nu:.025+j*.028,theta:-Math.PI+j*Math.PI*2/12}));
  function draw(){
    const {ctx,width:w,height:h}=fitCanvas(canvas);ctx.clearRect(0,0,w,h);const c=+cInput.value,d=+dInput.value,g=Math.exp(c),ink=css("--ink"),muted=css("--muted"),grid=css("--line"),dynamic=css("--dynamic"),state=css("--state"),motion=css("--motion");
    const gap=Math.max(28,w*.035),left={x:28,y:70,w:w*.56-gap,h:h-100},right={x:w*.56+16,y:70,w:w*.44-44,h:h-100};
    label(ctx,"同一个 cₜ 广播给 12 个模态",left.x,left.y-38,ink,13,"left");label(ctx,`cₜ=${c.toFixed(2)}，gₜ=exp(cₜ)=${g.toFixed(2)}`,left.x,left.y-17,dynamic,10,"left");
    const plot={x:left.x+45,y:left.y+18,w:left.w-58,h:left.h-62};
    for(let tick=0;tick<=4;tick++){const yy=plot.y+tick*plot.h/4;line(ctx,[plot.x,yy],[plot.x+plot.w,yy],grid,1);label(ctx,(1-tick/4).toFixed(2),plot.x-9,yy+4,muted,8,"right");}
    label(ctx,"保留率 ρ（1 表示完全保留）",plot.x,plot.y-8,muted,9,"left");
    const values=modes.map((m,j)=>{const rho=Math.exp(-m.nu*g),baseRho=Math.exp(-m.nu),x=plot.x+j*plot.w/(modes.length-1),y=plot.y+(1-rho)*plot.h,baseY=plot.y+(1-baseRho)*plot.h;line(ctx,[x,baseY],[x,y],motion,1.2);dot(ctx,x,baseY,3,css("--line-strong"));dot(ctx,x,y,5,state);label(ctx,String(j),x,plot.y+plot.h+17,muted,8);return {...m,rho,phi:m.theta+d};});
    label(ctx,"灰点：c=0 的基准；绿色：当前结果",plot.x,plot.y+plot.h+37,muted,8,"left");
    label(ctx,"同一个 dₜ 整体平移相位谱",right.x,right.y-38,ink,13,"left");label(ctx,`dₜ=${d.toFixed(2)} rad（弧度）`,right.x,right.y-17,dynamic,10,"left");
    const radius=Math.min(right.w,right.h)*.31,cx=right.x+right.w*.5,cy=right.y+right.h*.47;ctx.beginPath();ctx.arc(cx,cy,radius,0,Math.PI*2);ctx.strokeStyle=grid;ctx.lineWidth=1;ctx.stroke();line(ctx,[cx-radius-8,cy],[cx+radius+8,cy],grid,1);line(ctx,[cx,cy-radius-8],[cx,cy+radius+8],grid,1);
    values.forEach((m,j)=>{const bx=cx+Math.cos(m.theta)*radius,by=cy+Math.sin(m.theta)*radius,px=cx+Math.cos(m.phi)*radius,py=cy+Math.sin(m.phi)*radius;dot(ctx,bx,by,3,css("--line-strong"));arrow(ctx,[bx,by],[px,py],`${motion}`,1);dot(ctx,px,py,4.5,dynamic);if(j%3===0)label(ctx,`m${j}`,px+(px>=cx?8:-8),py+(py>=cy?12:-6),muted,8,px>=cx?"left":"right");});
    label(ctx,"灰点：基础相位 θⱼ",right.x+right.w*.5,right.y+right.h-28,muted,8);label(ctx,"橙点：当前相位 φⱼ=θⱼ+dₜ",right.x+right.w*.5,right.y+right.h-10,dynamic,8);
    const picks=[0,3,6,9];readout.innerHTML=`<table><thead><tr><th>模态 j</th><th>静态 νⱼ</th><th>当前保留率 ρⱼ,ₜ</th><th>静态 θⱼ</th><th>当前相位 φⱼ,ₜ</th></tr></thead><tbody>${picks.map(j=>`<tr><td>${j}</td><td>${values[j].nu.toFixed(3)}</td><td>${values[j].rho.toFixed(3)}</td><td>${values[j].theta.toFixed(2)}</td><td>${values[j].phi.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
  }
  cInput.addEventListener("input",draw);dInput.addEventListener("input",draw);draw();
}

export function initCoreFigure(root){const type=root.dataset.figure;if(type==="step")initStep(root);if(type==="memory")initMemory(root);if(type==="warp")initWarp(root);if(type==="coherent")initCoherent(root);}
