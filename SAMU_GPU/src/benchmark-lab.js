const $ = s => document.querySelector(s);
const fmt = (v,d=2) => Number.isFinite(v)?Number(v).toLocaleString(undefined,{maximumFractionDigits:d}):"N/A";
const esc = s => String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const ns="http://www.w3.org/2000/svg";
const el=(tag,attrs={},text="")=>{const x=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>x.setAttribute(k,v));x.textContent=text;return x;};
const colors={"SAMU · tree":"var(--state)","SAMU · chunk":"var(--write)","Mamba-3 · official fused":"var(--mamba)","RG-LRU · official-source reference":"var(--rglru)"};

let rows=[],environment={},profiler={},tab="status";

function measured(filter={}) { return rows.filter(r=>r.status==="measured"&&Object.entries(filter).every(([k,v])=>String(r[k])===String(v))); }
function best(list){return list.reduce((a,b)=>!a||b.median_ms<a.median_ms?b:a,null);}
function matchingBase(r){return r.d_model===128&&r.modes===64;}

function renderHeadline(){
  $("#headline-env").textContent=`${environment.gpu || "GPU unknown"} · CUDA ${environment.torch_cuda||"?"} · PyTorch ${environment.torch||"?"}`;
  const base=measured({workload:"forward",dtype:"bf16",batch:1,length:2048}).filter(matchingBase);
  const samu=best(base.filter(r=>r.model==="samu")),rg=best(base.filter(r=>r.model==="rglru"));
  const sweep=measured({workload:"forward",dtype:"bf16",batch:1}).filter(matchingBase),lengths=[...new Set(sweep.map(r=>r.length))];
  const pairs=lengths.map(length=>({length,samu:best(sweep.filter(r=>r.length===length&&r.model==="samu")),mamba:best(sweep.filter(r=>r.length===length&&r.model==="mamba3"))})).filter(x=>x.samu&&x.mamba);
  const samuWin=pairs.filter(x=>x.samu.median_ms<x.mamba.median_ms).sort((a,b)=>(b.mamba.median_ms/b.samu.median_ms)-(a.mamba.median_ms/a.samu.median_ms))[0];
  const reference=pairs.find(x=>x.length===2048)||pairs[0],mamba=reference?.mamba;
  const chunks=measured({model:"samu",backend:"chunked_compressed",workload:"forward",dtype:"bf16",batch:1,length:2048}).filter(matchingBase),chunk=best(chunks);
  const cards=[
    {label:"SAMU vs official-source RG-LRU reference",metric:samu&&rg?`${fmt(rg.median_ms/samu.median_ms,1)}× lower latency`:"Not measured",body:samu&&rg?`${esc(samu.backend)} ${fmt(samu.median_ms)} ms vs serial reference ${fmt(rg.median_ms)} ms。实现结论，不是 RG-LRU 架构胜负。`:"缺少匹配点。"},
    samuWin?{label:"A measured long-sequence crossover",metric:`${fmt(samuWin.mamba.median_ms/samuWin.samu.median_ms,1)}× SAMU lower latency`,body:`L=${samuWin.length}：SAMU ${fmt(samuWin.samu.median_ms)} ms vs official Mamba-3 ${fmt(samuWin.mamba.median_ms)} ms。不同实现路线，仍是 wall-clock 事实。`}:{label:"Official fused Mamba-3 vs current SAMU",metric:reference&&mamba?`${fmt(reference.samu.median_ms/mamba.median_ms,1)}× Mamba-3 faster`:"Not measured",body:reference&&mamba?`L=${reference.length}：${fmt(mamba.median_ms)} ms vs ${fmt(reference.samu.median_ms)} ms。Track A 实现成熟度差异是结果的一部分。`:"缺少匹配点。"},
    {label:"Best measured SAMU chunk prototype",metric:chunk?`C = ${chunk.chunk_size}`:"Not measured",body:chunk?`${fmt(chunk.median_ms)} ms median at L=2048；tested chunk sizes 中最低，不外推到 custom kernel。`:"未运行 chunk sweep。"}
  ];
  $("#headline-results").innerHTML=cards.map(c=>`<div class="headline-result"><span class="metric-label">${c.label}</span><strong class="metric">${c.metric}</strong><p>${c.body}</p></div>`).join("");
}

function seriesName(r){if(r.model==="mamba3")return "Mamba-3 · official fused";if(r.model==="rglru")return "RG-LRU · official-source reference";if(r.backend==="chunked_compressed")return "SAMU · chunk";if(r.backend?.includes("tree"))return "SAMU · tree";return null;}
function currentRows(){const workload=$("#filter-workload").value,dtype=$("#filter-dtype").value,batch=+$("#filter-batch").value;const candidates=measured({workload,dtype,batch}).filter(matchingBase).filter(r=>seriesName(r));const by=new Map();for(const r of candidates){const key=`${seriesName(r)}:${r.length}`;if(!by.has(key)||r.median_ms<by.get(key).median_ms)by.set(key,r);}return [...by.values()];}

function renderChart(){
  const svg=$("#benchmark-chart"),data=currentRows(),metric=$("#filter-metric").value;svg.replaceChildren();
  if(!data.length){svg.append(el("text",{x:480,y:250,"text-anchor":"middle",fill:"var(--muted)"},"No measured points for this filter"));return;}
  const W=960,H=520,m={l:82,r:28,t:32,b:72},lengths=[...new Set(data.map(d=>d.length))].sort((a,b)=>a-b),vals=data.map(d=>+d[metric]).filter(Number.isFinite),lo=Math.min(...vals),hi=Math.max(...vals),logY=hi/Math.max(lo,1e-12)>25;
  const x=v=>m.l+(lengths.length===1?.5:lengths.indexOf(v)/(lengths.length-1))*(W-m.l-m.r),norm=v=>logY?(Math.log(v)-Math.log(lo))/(Math.log(hi)-Math.log(lo)||1):(v-lo)/(hi-lo||1),y=v=>H-m.b-norm(v)*(H-m.t-m.b);
  for(let i=0;i<=5;i++){const yy=m.t+i*(H-m.t-m.b)/5;svg.append(el("line",{x1:m.l,y1:yy,x2:W-m.r,y2:yy,stroke:"var(--line)","stroke-width":1}));const value=logY?Math.exp(Math.log(hi)+(Math.log(lo)-Math.log(hi))*i/5):hi+(lo-hi)*i/5;svg.append(el("text",{x:m.l-12,y:yy+4,"text-anchor":"end",fill:"var(--muted)","font-size":11,"font-family":"IBM Plex Mono"},formatMetric(value,metric)));}
  lengths.forEach(L=>{const xx=x(L);svg.append(el("text",{x:xx,y:H-m.b+28,"text-anchor":"middle",fill:"var(--muted)","font-size":11,"font-family":"IBM Plex Mono"},String(L)));});
  const names=[...new Set(data.map(seriesName))];names.forEach((name,idx)=>{const s=data.filter(r=>seriesName(r)===name).sort((a,b)=>a.length-b.length),path=s.map((d,i)=>`${i?"L":"M"}${x(d.length)},${y(+d[metric])}`).join(" ");svg.append(el("path",{d:path,fill:"none",stroke:colors[name],"stroke-width":2.5}));s.forEach(d=>{const c=el("circle",{cx:x(d.length),cy:y(+d[metric]),r:6,fill:colors[name],stroke:"var(--paper)","stroke-width":2,tabindex:0,"data-id":d.id});c.addEventListener("click",()=>renderDetail(d,metric));c.addEventListener("focus",()=>renderDetail(d,metric));svg.append(c);});const lx=m.l+idx*220;svg.append(el("line",{x1:lx,y1:H-18,x2:lx+22,y2:H-18,stroke:colors[name],"stroke-width":3}));svg.append(el("text",{x:lx+28,y:H-14,fill:"var(--muted)","font-size":10,"font-family":"IBM Plex Mono"},name));});
  svg.append(el("text",{x:(m.l+W-m.r)/2,y:H-37,"text-anchor":"middle",fill:"var(--muted)","font-size":11,"font-family":"IBM Plex Mono"},"sequence length L"));
  $("#chart-caption").textContent=`${data.length} 个实测点 · ${metric} · ${logY?"log":"linear"} y-axis · best row per implementation/length. 点击点查看 raw samples。`;
  if(data[0])renderDetail(data[0],metric);
}
function formatMetric(v,metric){if(metric==="peak_allocated_bytes")return `${fmt(v/1048576,1)} MiB`;if(metric==="tokens_per_second")return `${fmt(v/1000,1)}k tok/s`;return `${fmt(v,2)} ms`;}
function renderDetail(r,metric){const value=formatMetric(+r[metric],metric),samples=r.raw_samples_ms||[];$("#benchmark-detail").innerHTML=`<p class="aside-title">Selected point</p><span class="evidence measured">Measured</span><strong class="big">${value}</strong><p>${esc(seriesName(r))}</p><dl><dt>shape</dt><dd>B${r.batch} · L${r.length} · d${r.d_model} · M${r.modes}</dd><dt>backend</dt><dd>${esc(r.backend)}</dd><dt>track</dt><dd>${esc(r.track)}</dd><dt>protocol</dt><dd>${esc(r.matching_protocol)}</dd><dt>median</dt><dd>${fmt(r.median_ms,4)} ms</dd><dt>P10 / P95</dt><dd>${fmt(r.p10_ms,4)} / ${fmt(r.p95_ms,4)}</dd><dt>compile / setup</dt><dd>${fmt(r.compile_setup_seconds,2)} s (excluded)</dd><dt>peak allocated</dt><dd>${fmt(r.peak_allocated_bytes/1048576,1)} MiB</dd><dt>samples</dt><dd>${samples.length}</dd></dl><p class="code-label">raw CUDA-event samples (ms)</p><p class="mono">${samples.map(x=>fmt(x,4)).join(" · ")||"N/A"}</p><a href="benchmark_results/raw/${encodeURIComponent(r.id)}.json">Open raw JSON →</a>`;}

function renderLab(){const panel=$("#lab-panel"),counts=rows.reduce((a,r)=>(a[r.status]=(a[r.status]||0)+1,a),{});if(tab==="status"){const unsupported=rows.filter(r=>r.status==="unsupported");panel.innerHTML=`<div class="coverage-grid"><div><span class="context-label">Measured</span><strong>${counts.measured||0}</strong><p>完成 warmup 和 synchronized CUDA-event timing 的配置。</p></div><div><span class="context-label">Unsupported</span><strong>${counts.unsupported||0}</strong><p>官方 CuTe decode、Mamba-2 缺失 fused dependency 与尚不存在的 SAMU custom kernels 被保留，不补数。</p></div><div><span class="context-label">Failed / OOM</span><strong>${(counts.failed||0)+(counts.oom||0)}</strong><p>失败组合不会从 raw directory 静默删除。</p></div></div><details><summary>查看 unsupported 配置与原始原因</summary><ul>${unsupported.map(r=>`<li><b>${esc(r.model)} · ${esc(r.backend)}</b> — ${esc(r.error)}</li>`).join("")}</ul></details>`;}
  if(tab==="environment")panel.innerHTML=`<dl class="environment-grid">${Object.entries({GPU:environment.gpu,"Board / driver":environment.driver_and_board,"Compute capability":environment.compute_capability,"SM count":environment.sm_count,"CUDA runtime":environment.torch_cuda,"NVCC":environment.nvcc,"PyTorch":environment.torch,"Triton":environment.triton,"Nsight Compute":environment.ncu}).map(([k,v])=>`<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>`;
  if(tab==="profiler")panel.innerHTML=`<p class="fairness-inline">torch.profiler 事件时间是实测；DRAM/L2/occupancy/register/Tensor Core/SFU/SM counters 因 NCU 不可用而保持 N/A。</p><div class="profiler-grid">${(profiler.profiles||[]).map(p=>`<div><span class="context-label">${esc(p.name)}</span><h3>${fmt(p.self_cuda_total_us,0)} μs · ${p.kernel_event_count} events</h3><ol>${(p.events||[]).slice(0,4).map(e=>`<li>${esc(e.name.length>44?e.name.slice(0,44)+"…":e.name)} · ${fmt(100*e.self_cuda_share,1)}%</li>`).join("")}</ol></div>`).join("")}</div>`;
  if(tab==="raw")panel.innerHTML=`<p><b>Source of truth:</b> ${rows.length} aggregated rows generated from one JSON file per configuration. Aggregation does not invent missing samples.</p><p><a class="button" href="benchmark_results/summary.json">Open summary.json</a> <a class="button" href="benchmark_results/summary.csv">Download summary.csv</a> <a class="button" href="benchmark_results/run_manifest.json">Open run manifest</a></p>`;
}

export async function initBenchmarkLab(){
  try{const [s,e,p]=await Promise.all([fetch("benchmark_results/summary.json").then(r=>r.json()),fetch("benchmark_results/environment.json").then(r=>r.json()),fetch("benchmark_results/profiler/representatives.json").then(r=>r.json())]);rows=s.rows||[];environment=e;profiler=p;renderHeadline();renderChart();renderLab();
    $("#benchmark-controls").addEventListener("change",renderChart);document.querySelector(".lab-tabs").addEventListener("click",ev=>{const b=ev.target.closest("[data-lab-tab]");if(!b)return;tab=b.dataset.labTab;document.querySelectorAll("[data-lab-tab]").forEach(x=>x.classList.toggle("active",x===b));renderLab();});
  }catch(err){$("#headline-results").innerHTML=`<div class="headline-result"><strong>Benchmark data unavailable</strong><p>${esc(err.message)}。请通过 HTTP server 打开本页，file:// 会阻止 fetch。</p></div>`;$("#chart-caption").textContent=`Load error: ${err.message}`;}
}
