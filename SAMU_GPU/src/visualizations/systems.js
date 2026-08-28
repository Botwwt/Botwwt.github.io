import {activateWhenVisible, AnimationController, bindTransport, css, fitCanvas} from "../animation-controller.js";

export function initScan(root){
  const track=root.querySelector("[data-scan-track]"),eq=root.querySelector("[data-scan-equation]");let mode="tokens";
  const maxStep={tokens:3,compressed:3,prefix:4};
  const controller=new AnimationController({steps:4,duration:1200,onStep:render,onFrame:()=>{}});
  function tokenView(s){
    if(s===0)return {nodes:["T₀(z)=a₀z+w₀","T₁(z)=a₁z+w₁","T₂(z)=a₂z+w₂","T₃(z)=a₃z+w₃"],active:[0,1,2,3],eq:"四个 token 各自是 affine map。点击 Next 开始按时间顺序组合。"};
    if(s===1)return {nodes:["T₁∘T₀ → (P₀₁,q₀₁)","T₃∘T₂ → (P₂₃,q₂₃)"],active:[0,1],eq:"q₀₁ = a₁w₀+w₁；两组 compose 可以同时完成。"};
    if(s===2)return {nodes:["(P₂₃,q₂₃) ∘ (P₀₁,q₀₁)"],active:[0],eq:"P=P₂₃P₀₁，q=P₂₃q₀₁+q₂₃。"};
    return {nodes:["T₀:₃(z)=Pz+q"],active:[0],eq:"最终 summary 对任意输入 z 都精确；顺序没有被交换。"};
  }
  function compressedView(s){const items=s<2?["P₀ vector","P₁ vector","P₂ vector","P₃ vector"]:s===2?["fold transition structure"]:["C · G · D","q[ M complex ]"];return {nodes:items,active:items.map((_,i)=>i),eq:s<2?"普通对角 P 看起来是 M-dimensional vector。":"SAMU 把 transition P 精确折叠为 C,G,D；但 q 仍是 state-sized O(M)。"};}
  function prefixView(s){const offset=[0,1,2,4][s];return {nodes:Array.from({length:8},(_,i)=>offset&&i>=offset?`S${i-offset}:${i}`:`S${i}`),active:Array.from({length:8},(_,i)=>offset&&i>=offset?i:-1).filter(i=>i>=0),eq:s===0?"8 个 chunk summaries：尚未传播 prefix。":`Round ${s}: offset ${offset}。只有 index ≥ ${offset} 的节点组合左侧 prefix。`};}
  function render(){controller.steps=maxStep[mode]+1;controller.stepIndex=Math.min(controller.stepIndex,maxStep[mode]);const view=mode==="tokens"?tokenView(controller.stepIndex):mode==="compressed"?compressedView(controller.stepIndex):prefixView(controller.stepIndex);track.innerHTML=view.nodes.map((n,i)=>`<div class="scan-node ${view.active.includes(i)?"active":""} ${controller.stepIndex>0?"composed":""}" style="grid-column:span ${Math.max(1,Math.floor(8/view.nodes.length))}">${n}</div>`).join("");eq.innerHTML=`<b>${mode==="prefix"?`prefix round ${controller.stepIndex}`:`compose step ${controller.stepIndex}`}</b><br>${view.eq}`;}
  root.addEventListener("click",e=>{if(e.target.dataset.scanMode){mode=e.target.dataset.scanMode;controller.reset();root.querySelectorAll("[data-scan-mode]").forEach(b=>b.classList.toggle("primary",b===e.target));render();}});bindTransport(root,controller);activateWhenVisible(root,controller);render();
}

export function initTimeMode(root){
  const canvas=root.querySelector("[data-tm-canvas]"),stats=root.querySelector("[data-tm-stats]"),tile=root.querySelector("[data-tm-tile]");let kind="serial",phase=0;
  function draw(){const {ctx,width:w,height:h}=fitCanvas(canvas);ctx.clearRect(0,0,w,h);const n=32,gap=1,cell=Math.min((w-70)/n,(h-55)/n),ox=50,oy=20;phase=(phase+.12)%32;
    for(let t=0;t<n;t++)for(let m=0;m<n;m++){let active=false;if(kind==="serial")active=t<=phase&&t>phase-5;else{const [tt,mm]=tile.value.split("x").map(Number);active=(Math.floor(t/Math.min(n,tt)) + Math.floor(m/mm) + Math.floor(phase/4))%3===0;}ctx.fillStyle=active?css("--dynamic"):((t%8===0||m%8===0)?css("--paper-2"):`${css("--state")}16`);ctx.fillRect(ox+m*cell,oy+t*cell,cell-gap,cell-gap);}
    ctx.fillStyle=css("--muted");ctx.font="10px IBM Plex Mono";ctx.fillText("time ↓",3,30);ctx.fillText("mode →",w-75,h-8);if(!matchMedia("(prefers-reduced-motion: reduce)").matches)requestAnimationFrame(draw);
  }
  function update(){const [tt,mm]=tile.value.split("x").map(Number),ctas=Math.ceil(32/Math.min(32,tt))*Math.ceil(32/mm),warps=Math.ceil(mm/8);stats.innerHTML=`<p class="aside-title">Logical schedule</p><dl><dt>matrix</dt><dd>32 × 32</dd><dt>CTA tile</dt><dd>${tt} × ${mm}</dd><dt>number of CTAs</dt><dd>${ctas}</dd><dt>warps / CTA</dt><dd>≈ ${warps}</dd><dt>parallel jobs</dt><dd>${kind==="serial"?32:ctas}</dd><dt>serial depth</dt><dd>${kind==="serial"?32:Math.min(32,tt)}</dd></dl><span class="evidence hypothesis">Needs benchmark</span>`;}
  root.addEventListener("click",e=>{if(!e.target.dataset.tmKind)return;kind=e.target.dataset.tmKind;root.querySelectorAll("[data-tm-kind]").forEach(b=>b.classList.toggle("primary",b===e.target));update();});tile.addEventListener("change",update);update();draw();
}

const models={
  samu:{title:"SAMU",count:"2",subtitle:"token coordinates",list:["dynamic: cₜ,dₜ","static: νⱼ,θⱼ","state: M complex","ρⱼ,φⱼ reconstructed"] ,formula:"z′ⱼ = exp(−νⱼexp(c)) · exp(i(θⱼ+d)) · zⱼ + wⱼ"},
  rglru:{title:"RG-LRU",count:"M",subtitle:"conceptual channel gates",list:["dynamic: channel-wise inputs/gates","static: recurrent parameters","state: real channels","optimized traffic is implementation-dependent"],formula:"x′ = aₜ ⊙ x + gateₜ ⊙ inputₜ"},
  mamba3:{title:"Mamba-3",count:"structured",subtitle:"official SSM controls",list:["dynamic SSM quantities","chunked hardware-aware path","SISO/MIMO variants","official fused Triton baseline"],formula:"official structured SSM update; representation follows current implementation"}
};
export function initCompare(root){const host=root.querySelector("[data-compare-cards]"),formula=root.querySelector("[data-compare-formula]");let active="samu";host.innerHTML=Object.entries(models).map(([k,m])=>`<div class="recurrence-card ${k===active?"active":""}" tabindex="0" data-model="${k}"><h3>${m.title}</h3><span class="dynamic-count">${m.count}</span><p>${m.subtitle}</p><ul>${m.list.map(x=>`<li>${x}</li>`).join("")}</ul></div>`).join("");function choose(k){active=k;host.querySelectorAll("[data-model]").forEach(e=>e.classList.toggle("active",e.dataset.model===k));formula.innerHTML=`<b>${models[k].title}</b><br>${models[k].formula}`;}host.addEventListener("click",e=>{const c=e.target.closest("[data-model]");if(c)choose(c.dataset.model);});host.addEventListener("keydown",e=>{if((e.key==="Enter"||e.key===" ")&&e.target.dataset.model)choose(e.target.dataset.model);});choose(active);}

const transitionData={
 token:{rglru:["channel-wise dynamic update quantities","state-sized response"],mamba3:["official per-token SSM quantities","structured state transition"],samu:["dynamic: c,d","static ν,θ reconstruct Pⱼ","write q=wₜ: O(M)"]},
 tokens:{rglru:["32 channel-wise dynamic steps","fusion may avoid HBM materialization"],mamba3:["32 structured SSM steps","hardware-aware chunk algebra"],samu:["32 pairs cₜ,dₜ","composable affine steps","q prefixes remain state-sized"]},
 chunk:{rglru:["chunk summary follows actual recurrence algebra","no simplification asserted here"],mamba3:["official chunked SSD/SSM representation","matmul-oriented implementation"],samu:["transition: C,G,D","zero-state output q: O(M)","whole summary is not O(1)"]}
};
export function initTransition(root){const view=root.querySelector("[data-transition-view]");let span="token";function render(){view.innerHTML=["rglru","mamba3","samu"].map(k=>`<div class="transition-model ${k}"><h3>${k==="mamba3"?"Mamba-3":k==="rglru"?"RG-LRU":"SAMU"}</h3><ul>${transitionData[span][k].map(x=>`<li>${x}</li>`).join("")}</ul></div>`).join("");}root.addEventListener("click",e=>{if(!e.target.dataset.transitionSpan)return;span=e.target.dataset.transitionSpan;root.querySelectorAll("[data-transition-span]").forEach(b=>b.classList.toggle("primary",b===e.target));render();});render();}

export function initSystemFigure(root){const type=root.dataset.figure;if(type==="scan")initScan(root);if(type==="timemode")initTimeMode(root);if(type==="compare")initCompare(root);if(type==="transition")initTransition(root);}
