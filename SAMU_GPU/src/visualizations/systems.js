import {activateWhenVisible, AnimationController, bindTransport, css, fitCanvas} from "../animation-controller.js";
import {renderMath} from "../math.js";

export function initScan(root){
  const track=root.querySelector("[data-scan-track]"),eq=root.querySelector("[data-scan-equation]");let mode="tokens";
  const maxStep={tokens:3,compressed:3,prefix:4};
  const controller=new AnimationController({steps:4,duration:1200,onStep:render,onFrame:()=>{}});
  function tokenView(s){
    if(s===0)return {nodes:["T₀(z)=a₀z+w₀","T₁(z)=a₁z+w₁","T₂(z)=a₂z+w₂","T₃(z)=a₃z+w₃"],active:[0,1,2,3],eq:"四个 token 各自对应一个仿射映射。点击“下一轮”后按时间顺序组合。"};
    if(s===1)return {nodes:["T₁∘T₀ → (P₀₁,q₀₁)","T₃∘T₂ → (P₂₃,q₂₃)"],active:[0,1],eq:"q₀₁ = a₁w₀+w₁；两组 compose 可以同时完成。"};
    if(s===2)return {nodes:["(P₂₃,q₂₃) ∘ (P₀₁,q₀₁)"],active:[0],eq:"P=P₂₃P₀₁，q=P₂₃q₀₁+q₂₃。"};
    return {nodes:["T₀:₃(z)=Pz+q"],active:[0],eq:"最终摘要对任意输入 z 都精确；时间顺序没有被交换。"};
  }
  function compressedView(s){const items=s<2?["P₀ 向量","P₁ 向量","P₂ 向量","P₃ 向量"]:s===2?["折叠状态转移结构"]:["C · G · D","q（M 个复数）"];return {nodes:items,active:items.map((_,i)=>i),eq:s<2?"普通对角 P 是 M 维向量。":"SAMU 把状态转移 P 精确折叠为 C,G,D；但 q 仍与状态同宽，大小为 O(M)。"};}
  function prefixView(s){const offset=[0,1,2,4][s];return {nodes:Array.from({length:8},(_,i)=>offset&&i>=offset?`S${i-offset}:${i}`:`S${i}`),active:Array.from({length:8},(_,i)=>offset&&i>=offset?i:-1).filter(i=>i>=0),eq:s===0?"8 个分块摘要：尚未传播前缀。":`第 ${s} 轮：间隔 ${offset}。只有编号不小于 ${offset} 的节点组合左侧前缀。`};}
  function render(){controller.steps=maxStep[mode]+1;controller.stepIndex=Math.min(controller.stepIndex,maxStep[mode]);const view=mode==="tokens"?tokenView(controller.stepIndex):mode==="compressed"?compressedView(controller.stepIndex):prefixView(controller.stepIndex);track.innerHTML=view.nodes.map((n,i)=>`<div class="scan-node ${view.active.includes(i)?"active":""} ${controller.stepIndex>0?"composed":""}" style="grid-column:span ${Math.max(1,Math.floor(8/view.nodes.length))}">${n}</div>`).join("");eq.innerHTML=`<b>${mode==="prefix"?`前缀第 ${controller.stepIndex} 轮`:`组合第 ${controller.stepIndex} 步`}</b><br>${view.eq}`;}
  root.addEventListener("click",e=>{if(e.target.dataset.scanMode){mode=e.target.dataset.scanMode;controller.reset();root.querySelectorAll("[data-scan-mode]").forEach(b=>b.classList.toggle("primary",b===e.target));render();}});bindTransport(root,controller);activateWhenVisible(root,controller);render();
}

export function initTimeMode(root){
  const canvas=root.querySelector("[data-tm-canvas]"),stats=root.querySelector("[data-tm-stats]"),tile=root.querySelector("[data-tm-tile]");let kind="serial",phase=0;
  function draw(){const {ctx,width:w,height:h}=fitCanvas(canvas);ctx.clearRect(0,0,w,h);const n=32,gap=1,cell=Math.min((w-70)/n,(h-55)/n),ox=50,oy=20;phase=(phase+.12)%32;
    for(let t=0;t<n;t++)for(let m=0;m<n;m++){let active=false;if(kind==="serial")active=t<=phase&&t>phase-5;else{const [tt,mm]=tile.value.split("x").map(Number);active=(Math.floor(t/Math.min(n,tt)) + Math.floor(m/mm) + Math.floor(phase/4))%3===0;}ctx.fillStyle=active?css("--dynamic"):((t%8===0||m%8===0)?css("--paper-2"):`${css("--state")}16`);ctx.fillRect(ox+m*cell,oy+t*cell,cell-gap,cell-gap);}
    ctx.fillStyle=css("--muted");ctx.font="10px IBM Plex Mono";ctx.fillText("时间 ↓",3,30);ctx.fillText("模态 →",w-75,h-8);if(!matchMedia("(prefers-reduced-motion: reduce)").matches)requestAnimationFrame(draw);
  }
  function update(){const [tt,mm]=tile.value.split("x").map(Number),ctas=Math.ceil(32/Math.min(32,tt))*Math.ceil(32/mm),warps=Math.ceil(mm/8);stats.innerHTML=`<p class="aside-title">逻辑调度</p><dl><dt>矩阵</dt><dd>32 × 32</dd><dt>线程块分片</dt><dd>${tt} × ${mm}</dd><dt>线程块数量</dt><dd>${ctas}</dd><dt>每块线程束</dt><dd>≈ ${warps}</dd><dt>并行任务</dt><dd>${kind==="serial"?32:ctas}</dd><dt>串行深度</dt><dd>${kind==="serial"?32:Math.min(32,tt)}</dd></dl><span class="evidence hypothesis">需要实测</span>`;}
  root.addEventListener("click",e=>{if(!e.target.dataset.tmKind)return;kind=e.target.dataset.tmKind;root.querySelectorAll("[data-tm-kind]").forEach(b=>b.classList.toggle("primary",b===e.target));update();});tile.addEventListener("change",update);update();draw();
}

const models={
  samu:{title:"SAMU",count:"2 个共享控制量",subtitle:"每个 token 产生 cₜ,dₜ",list:["动态量：cₜ,dₜ","静态量：νⱼ,θⱼ","状态：M 个复数","现场重建 ρⱼ,φⱼ"] ,formula:String.raw`z_{j,t+1}=e^{-\nu_j e^{c_t}}e^{i(\theta_j+d_t)}z_{j,t}+w_{j,t}`},
  rglru:{title:"RG-LRU",count:"2d 个动态门值",subtitle:"每个通道有输入门和递推门",list:["iₜ = sigmoid(Wₓx+bₓ)","rₜ = sigmoid(Wₐx+bₐ)","aₜ = exp(−8rₜ·softplus(a_param))","状态：d 个 FP32 实数通道"],formula:String.raw`h_t=a_t\odot h_{t-1}+\sqrt{1-a_t^2}\odot(i_t\odot x_t)`}
};
export function initCompare(root){const host=root.querySelector("[data-compare-cards]"),formula=root.querySelector("[data-compare-formula]");let active="samu";host.innerHTML=Object.entries(models).map(([k,m])=>`<div class="recurrence-card ${k===active?"active":""}" tabindex="0" data-model="${k}"><h3>${m.title}</h3><span class="dynamic-count">${m.count}</span><p>${m.subtitle}</p><ul>${m.list.map(x=>`<li>${x}</li>`).join("")}</ul></div>`).join("");function choose(k){active=k;host.querySelectorAll("[data-model]").forEach(e=>e.classList.toggle("active",e.dataset.model===k));formula.innerHTML=`<span class="formula-label">${models[k].title}</span><div class="math-display" data-katex>${models[k].formula}</div>`;renderMath(formula);}host.addEventListener("click",e=>{const c=e.target.closest("[data-model]");if(c)choose(c.dataset.model);});host.addEventListener("keydown",e=>{if((e.key==="Enter"||e.key===" ")&&e.target.dataset.model)choose(e.target.dataset.model);});choose(active);}

const transitionData={
 token:{rglru:["2d 个动态门输出","每个通道独立计算 a 和输入门","响应与状态同宽"],samu:["动态量：c,d","静态 ν,θ 重建 Pⱼ","写入 q=wₜ，大小 O(M)"]},
  tokens:{rglru:["32 个逐通道动态步骤","已实现摘要/前缀/回放 Triton 路径","融合后不写出完整门张量"],samu:["32 对 cₜ,dₜ","可组合的仿射步骤","q 前缀仍与状态同宽"]},
 chunk:{rglru:["摘要遵循 RG‑LRU 递推代数","不主张额外结构压缩"],samu:["状态转移可写成 C,G,D","零状态输出 q 的大小为 O(M)","完整摘要不是 O(1)"]}
};
export function initTransition(root){const view=root.querySelector("[data-transition-view]");let span="token";function render(){view.innerHTML=["rglru","samu"].map(k=>`<div class="transition-model ${k}"><h3>${k==="rglru"?"RG-LRU":"SAMU"}</h3><ul>${transitionData[span][k].map(x=>`<li>${x}</li>`).join("")}</ul></div>`).join("");}root.addEventListener("click",e=>{if(!e.target.dataset.transitionSpan)return;span=e.target.dataset.transitionSpan;root.querySelectorAll("[data-transition-span]").forEach(b=>b.classList.toggle("primary",b===e.target));render();});render();}

export function initSystemFigure(root){const type=root.dataset.figure;if(type==="scan")initScan(root);if(type==="timemode")initTimeMode(root);if(type==="compare")initCompare(root);if(type==="transition")initTransition(root);}
