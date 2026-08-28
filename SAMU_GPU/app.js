(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
  const number = (id, fallback = 0) => {
    const value = Number($(id)?.value);
    return Number.isFinite(value) ? value : fallback;
  };

  function formatBytes(value) {
    if (!Number.isFinite(value)) return "—";
    const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let index = 0;
    while (Math.abs(value) >= 1024 && index < units.length - 1) {
      value /= 1024;
      index += 1;
    }
    return `${value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${units[index]}`;
  }

  function renderMath(root = document.body) {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(root, {
      delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
      ],
      throwOnError: false
    });
  }

  function updateScroll() {
    const height = document.documentElement.scrollHeight - window.innerHeight;
    const progress = height > 0 ? window.scrollY / height : 0;
    $("#scrollProgress").style.width = `${clamp(progress, 0, 1) * 100}%`;
  }

  function setupSectionObserver() {
    const links = $$(".toc nav a");
    const byId = new Map(links.map(link => [link.getAttribute("href").slice(1), link]));
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach(link => link.classList.remove("is-active"));
      byId.get(visible.target.id)?.classList.add("is-active");
    }, { rootMargin: "-18% 0px -68% 0px", threshold: [0, .1, .3] });
    $$(".report-section").forEach(section => observer.observe(section));
  }

  function setupBackendToggle() {
    document.body.dataset.backend = "decode";
    $$(".backend-button").forEach(button => button.addEventListener("click", () => {
      $$(".backend-button").forEach(item => item.classList.toggle("is-active", item === button));
      document.body.dataset.backend = button.dataset.backend;
    }));
  }

  let warpMode = "mode";
  function renderWarp() {
    const grid = $("#warpGrid");
    grid.innerHTML = "";
    for (let lane = 0; lane < 32; lane += 1) {
      const cell = document.createElement("div");
      cell.className = "warp-lane";
      cell.innerHTML = `<b>${String(lane).padStart(2, "0")}</b><span>${warpMode === "mode" ? `mode ${lane}` : `time t+${lane}`}</span>`;
      cell.title = warpMode === "mode"
        ? `lane ${lane} 持有 complex mode ${lane} 的 ν, θ, x, y`
        : `lane ${lane} 代表 timestep t+${lane}，适合 warp prefix scan`;
      grid.appendChild(cell);
    }
    $("#warpExplanation").innerHTML = warpMode === "mode"
      ? "<b>Mode mapping：</b>同一个 lane 沿 time 迭代自己的 state；32 modes 独立并行，适合 decode / serial recurrence baseline。"
      : "<b>Time mapping：</b>同一个 mode 的 32 个 timesteps 分散到 lanes；需要用 associative affine composition 和 warp shuffle 传播 prefix。";
  }

  function setupWarp() {
    $$(".warp-map-button").forEach(button => button.addEventListener("click", () => {
      $$(".warp-map-button").forEach(item => item.classList.toggle("is-active", item === button));
      warpMode = button.dataset.map;
      renderWarp();
    }));
    renderWarp();
  }

  function drawTimeModeGrid() {
    const canvas = $("#timeModeCanvas");
    if (!canvas) return;
    const M = clamp(Math.round(number("#gridM", 512)), 32, 8192);
    const L = clamp(Math.round(number("#gridL", 2048)), 32, 65536);
    const C = clamp(Math.round(number("#gridC", 256)), 1, L);
    const tile = clamp(Math.round(number("#gridTile", 128)), 1, M);
    const chunks = Math.ceil(L / C);
    const modeTiles = Math.ceil(M / tile);
    const jobs = chunks * modeTiles;
    const warps = Math.ceil(M / 32);
    const threads = M;
    const rect = canvas.getBoundingClientRect();
    const cssWidth = Math.max(320, rect.width || 900);
    const cssHeight = Math.max(220, Math.min(390, cssWidth * .4));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = "#07101c";
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const left = 64, top = 35, right = 20, bottom = 42;
    const width = cssWidth - left - right;
    const height = cssHeight - top - bottom;
    const shownX = Math.min(chunks, 24);
    const shownY = Math.min(modeTiles, 12);
    const gap = 3;
    const cellW = (width - gap * (shownX - 1)) / shownX;
    const cellH = (height - gap * (shownY - 1)) / shownY;
    for (let y = 0; y < shownY; y += 1) {
      for (let x = 0; x < shownX; x += 1) {
        const hue = 186 + (x / Math.max(shownX - 1, 1)) * 28;
        const alpha = .22 + .5 * (1 - y / Math.max(shownY, 1));
        ctx.fillStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
        ctx.fillRect(left + x * (cellW + gap), top + y * (cellH + gap), Math.max(1, cellW), Math.max(1, cellH));
      }
    }
    ctx.strokeStyle = "#526b84";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, top - 8); ctx.lineTo(left + width, top - 8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(left - 8, top); ctx.lineTo(left - 8, top + height); ctx.stroke();
    ctx.fillStyle = "#91a7bb";
    ctx.font = "10px Consolas, monospace";
    ctx.fillText(`TIME → ${chunks} chunk${chunks === 1 ? "" : "s"}${chunks > shownX ? " (sampled)" : ""}`, left, 15);
    ctx.save(); ctx.translate(15, top + height); ctx.rotate(-Math.PI / 2); ctx.fillText(`MODE → ${modeTiles} tile${modeTiles === 1 ? "" : "s"}${modeTiles > shownY ? " (sampled)" : ""}`, 0, 0); ctx.restore();
    ctx.fillStyle = "#5ee6d4";
    ctx.fillText(`one cell = ≤ ${C} timesteps × ≤ ${tile} modes`, left, cssHeight - 14);

    const stats = [
      ["THREADS / SAMPLE", threads.toLocaleString()],
      ["WARPS / SAMPLE", warps.toLocaleString()],
      ["MODE TILES", modeTiles.toLocaleString()],
      ["TIME CHUNKS", chunks.toLocaleString()],
      ["LOCAL JOBS / SAMPLE", jobs.toLocaleString()]
    ];
    $("#gridStats").innerHTML = stats.map(([label, value]) => `<div><span>${label}</span><b>${value}</b></div>`).join("");
  }

  function setupTimeModeGrid() {
    ["#gridM", "#gridL", "#gridC", "#gridTile"].forEach(id => $(id).addEventListener("input", drawTimeModeGrid));
    window.addEventListener("resize", drawTimeModeGrid, { passive: true });
    drawTimeModeGrid();
  }

  const toyDefaults = [
    { c: -.30, d: .10, wr: .80, wi: .10 },
    { c: .20, d: -.05, wr: -.20, wi: .35 },
    { c: -.10, d: .18, wr: .45, wi: -.15 },
    { c: .35, d: .04, wr: .10, wi: .25 }
  ];
  const complex = (re, im) => ({ re, im });
  const cmul = (a, b) => complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
  const cadd = (a, b) => complex(a.re + b.re, a.im + b.im);
  const cfmt = z => `${z.re.toFixed(3)} ${z.im < 0 ? "−" : "+"} ${Math.abs(z.im).toFixed(3)}i`;

  function toyTransition(c, d, nu = .25, theta = .40) {
    const rho = Math.exp(-nu * Math.exp(c));
    const phase = theta + d;
    return complex(rho * Math.cos(phase), rho * Math.sin(phase));
  }

  function renderToy() {
    const tokens = toyDefaults.map((_, index) => ({
      c: number(`#toy-c-${index}`), d: number(`#toy-d-${index}`),
      wr: number(`#toy-wr-${index}`), wi: number(`#toy-wi-${index}`)
    }));
    let P = complex(1, 0);
    let q = complex(0, 0);
    let G = 0, D = 0;
    const rows = [];
    tokens.forEach((token, index) => {
      const a = toyTransition(token.c, token.d);
      P = cmul(a, P);
      q = cadd(cmul(a, q), complex(token.wr, token.wi));
      G += Math.exp(token.c);
      D += token.d;
      rows.push(`<div class="toy-step-row"><b>T${index}</b><span>P=${cfmt(P)}</span><span>q=${cfmt(q)}</span></div>`);
    });
    $("#toySteps").innerHTML = rows.join("");
    $("#toySummary").innerHTML = `
      <span>GENERIC SUMMARY</span><b>(P=${cfmt(P)}, q=${cfmt(q)})</b>
      <span>SAMU COMPRESSED</span><b>(C=4, G=${G.toFixed(3)}, D=${D.toFixed(3)}, q)</b>
      <span>FIXED TOY SPECTRUM</span><b>ν=.25, θ=.40</b>`;
  }

  function setupToy() {
    const root = $("#toyInputs");
    root.innerHTML = toyDefaults.map((token, index) => `<div class="toy-token"><b>T${index}</b>
      <label>c<input id="toy-c-${index}" type="number" step="0.05" value="${token.c}"></label>
      <label>d<input id="toy-d-${index}" type="number" step="0.05" value="${token.d}"></label>
      <label>wR<input id="toy-wr-${index}" type="number" step="0.05" value="${token.wr}"></label>
      <label>wI<input id="toy-wi-${index}" type="number" step="0.05" value="${token.wi}"></label>
    </div>`).join("");
    $$("input", root).forEach(input => input.addEventListener("input", renderToy));
    renderToy();
  }

  let scanSize = 4;
  let scanStep = 0;
  function buildScanLayers(size) {
    const base = Array.from({ length: size }, (_, i) => `S${i}`);
    const layers = [{ label: "Local summaries", nodes: base, combined: [] }];
    for (let offset = 1; offset < size; offset *= 2) {
      const nodes = base.map((_, index) => {
        const start = Math.max(0, index - (offset * 2 - 1));
        const end = index;
        if ((index + 1) % (offset * 2) === 0) return `S${end}∘…∘S${start}`;
        return index % (offset * 2) < offset ? `S${index}` : `carry ${index}`;
      });
      const combined = nodes.map((_, index) => (index + 1) % (offset * 2) === 0 ? index : -1).filter(i => i >= 0);
      layers.push({ label: `offset ${offset} · pair / combine`, nodes, combined });
    }
    const prefixes = base.map((_, i) => i === 0 ? "I" : i === 1 ? "T₀" : `T${i - 1}∘…∘T₀`);
    layers.push({ label: "exclusive prefix", nodes: prefixes, combined: prefixes.map((_, i) => i) });
    return layers;
  }

  function renderScan() {
    const layers = buildScanLayers(scanSize);
    scanStep = clamp(scanStep, 0, layers.length - 1);
    $("#scanVisualizer").innerHTML = layers.map((layer, layerIndex) => `<div class="scan-layer ${layerIndex <= scanStep ? "is-visible" : ""}" style="grid-template-columns:repeat(${scanSize},minmax(72px,1fr))" aria-label="${layer.label}">${layer.nodes.map((node, i) => `<div class="scan-node ${layer.combined.includes(i) ? "is-combined" : ""}">${node}</div>`).join("")}</div>`).join("");
    const current = layers[scanStep];
    $("#scanCaption").innerHTML = `<b>Layer ${scanStep}</b> · ${current.label}${scanStep === layers.length - 1 ? "。每个 chunk 现在拿到进入自己之前的真实 prefix summary。" : "。结合律允许这一层的 pairs 并行。"}`;
    $("#scanPlay").textContent = scanStep === layers.length - 1 ? "从头播放" : "播放一层";
  }

  function setupScan() {
    $("#scanSizeButton").addEventListener("click", () => {
      scanSize = scanSize === 4 ? 8 : 4;
      scanStep = 0;
      $("#scanSizeButton").textContent = `${scanSize} chunks`;
      renderScan();
    });
    $("#scanPlay").addEventListener("click", () => {
      const last = buildScanLayers(scanSize).length - 1;
      scanStep = scanStep >= last ? 0 : scanStep + 1;
      renderScan();
    });
    $("#scanReset").addEventListener("click", () => { scanStep = 0; renderScan(); });
    renderScan();
  }

  function updatePerformance() {
    const B = Math.max(1, Math.round(number("#perfB", 4)));
    const L = Math.max(1, Math.round(number("#perfL", 8192)));
    const M = Math.max(1, Math.round(number("#perfM", 2048)));
    const d = Math.max(1, Math.round(number("#perfD", 2048)));
    const tile = Math.max(1, Math.round(number("#perfTile", 128)));
    const C = Math.max(1, Math.round(number("#perfC", 256)));
    const bytes = Math.max(1, number("#perfType", 2));
    const chunksPerSequence = Math.ceil(L / C);
    const chunks = B * chunksPerSequence;
    const modeTiles = Math.ceil(M / tile);
    const write = B * L * 2 * M * bytes;
    const state = B * 2 * M * bytes;
    const q = chunks * 2 * M * bytes;
    const control = B * L * 2 * bytes;
    const logicalWriteTraffic = 2 * write;
    const generic = chunks * 4 * M * bytes;
    const samu = chunks * (2 * M * bytes + 2 * bytes + 4);
    const saved = Math.max(0, generic - samu);
    const gemmFlops = 2 * B * L * d * (2 * M);
    const jobs = chunks * modeTiles;
    const results = [
      ["WRITE TENSOR", formatBytes(write)], ["WRITE + READ", formatBytes(logicalWriteTraffic)],
      ["RECURRENT STATE", formatBytes(state)], ["COMPACT CONTROLS", formatBytes(control)],
      ["NUMBER OF CHUNKS", chunks.toLocaleString()], ["LOCAL JOBS", jobs.toLocaleString()],
      ["q SUMMARY", formatBytes(q)], ["P STORAGE AVOIDED", formatBytes(saved)],
      ["GENERIC (P,q)", formatBytes(generic)], ["SAMU (C,G,D,q)", formatBytes(samu)],
      ["WRITE GEMM FLOPs", gemmFlops.toExponential(3)], ["CHUNKS / SEQUENCE", chunksPerSequence.toLocaleString()]
    ];
    $("#perfResults").innerHTML = results.map(([label, value]) => `<div class="perf-result"><span>${label}</span><b>${value}</b></div>`).join("");
    const ratio = generic > 0 ? samu / generic : 1;
    $("#genericBar").style.width = "100%";
    $("#samuBar").style.width = `${Math.max(2, ratio * 100)}%`;
    $("#genericLabel").textContent = formatBytes(generic);
    $("#samuLabel").textContent = formatBytes(samu);
  }

  function setupPerformance() {
    ["#perfB", "#perfL", "#perfM", "#perfD", "#perfTile", "#perfC", "#perfType"].forEach(id => $(id).addEventListener("input", updatePerformance));
    updatePerformance();
  }

  const lessons = [
    ["SAMU 一个 timestep", "上一时刻的 complex state，怎样变成下一时刻？", "先只看一个 mode：旧状态乘 transition，再加 write。", "GPU lane 可以持有这个 mode 的 x,y。", "z′ = A·z + write", "先记住：乘法支路决定记忆如何保留，加法支路写入新信息。"],
    ["write 是什么", "新 token 怎样进入所有 spectral modes？", "一个 real 输入 u 分别投影到 real / imag 两组列。", "两次投影可以拼成一次宽 real GEMM。", "writeⱼ = γⱼ(uᵀwᴿⱼ + i·uᵀwᴵⱼ)", "write 是 dense projection，不是 recurrence 本身。"],
    ["为什么 write 可以 GEMM", "单步的 matrix-vector 为什么会变成 matrix-matrix？", "把 B×L 个 tokens 堆成 N 行，每行用相同权重。", "Tensor Core 重用 weight tiles，获得更高 arithmetic intensity。", "[N,d] · [d,2M] → [N,2M]", "序列长度提供 GEMM 的 rows。"],
    ["global memory 问题", "GEMM 结束后 write 去了哪里？", "独立 kernel 会把 [B,L,2M] 写出，recurrence 再读入。", "这是 logical traffic；L2 可能命中，但必须由 profiler 确认。", "traffic ≈ 2·B·L·2M·dtype_bytes", "先量化中间张量，再讨论 fusion。"],
    ["register 是什么", "为什么 state 值得留在离 ALU 最近的位置？", "同一 mode 的 x,y 会被下一个 timestep 立刻使用。", "multi-step kernel 内让 x,y 留在 registers，避免为 recurrence 反复 round-trip。", "(x,y)ₜ → (x,y)ₜ₊₁ → …", "register residency 只在 kernel 生命周期内成立。"],
    ["one thread one mode", "最简单的并行映射是什么？", "lane j 负责 complex mode j，持有 ν,θ,x,y。", "time 在 lane 内循环，mode 在 lanes 间并行。", "lane j ↔ mode j", "这是 decode 与 V1 recurrence 的清晰 baseline。"],
    ["warp 是什么", "32 个 lanes 如何一起工作？", "一个 warp 同步执行 32 个 mode updates。", "c,d 是 warp 共享 token 控制，可广播或只算一次。", "1 warp = 32 lanes ≈ 32 complex modes", "coherent control 正好提供跨 lanes 的重用。"],
    ["coherent c,d", "两个标量怎么产生 M 个不同 transition？", "同一 c,d 作用到不同 νⱼ,θⱼ。", "dynamic metadata 小，static spectral diversity 大。", "Aⱼₜ = exp(−νⱼexp(cₜ))·exp(i(θⱼ+dₜ))", "共享控制不等于共享 transition。"],
    ["CTA / multiple warps", "M 大于 32 时怎么办？", "多个 warps 组成 CTA，覆盖一个 mode tile。", "control 可在 CTA shared/register path 中 staging；state 仍 mode-local。", "CTA tile = warps × 32 modes", "tile 大小要平衡 occupancy 与 reuse。"],
    ["为什么长序列还不够快", "mode 全并行后，time 仍要走 L 步。", "长度 8192 的序列仍有 8192-deep dependency chain。", "训练已知整段序列，需要把 affine 结合律用于 time parallelism。", "parallel depth: O(L) → chunk + O(log #chunks)", "这就是 chunking 出现的原因。"],
    ["chunk 是什么", "如何把长 time 轴切成并行工作单元？", "每个 chunk 独立从 zero state 跑出 local q。", "batch × chunks × mode tiles 形成大量 GPU jobs。", "Sₖ=[kC,(k+1)C)", "chunk 先解决局部工作，再解决跨 chunk 边界。"],
    ["affine operator", "为什么 recurrence 可以组合？", "每步是 T(z)=az+w；函数复合仍是 affine。", "组合是 associative，所以可以 tree scan。", "(a₂,b₂)∘(a₁,b₁)=(a₂a₁,a₂b₁+b₂)", "结合律给并行；不交换保证时间方向。"],
    ["q 是什么", "chunk summary 里的 q 有什么直觉？", "输入 state 设为 zero，跑完整个 chunk，最后状态就是 q。", "q 是 M 个 complex modes 的 additive contribution。", "q = T_chunk(0)", "q 没被压缩，仍是 O(M)。"],
    ["(C,G,D,q)", "SAMU 比 generic (P,q) 多用了什么结构？", "把所有 exp(c) 求和为 G，把 d 求和为 D，C 记录长度。", "每个 mode 的 P 可由 C,G,D 与 ν,θ 重建。", "Pⱼ=e^(−νⱼG)·e^{i(Cθⱼ+D)}", "被压缩的是 P 的动态表示。"],
    ["prefix scan", "每个 chunk 如何得到真正的输入 state？", "对更早 chunks 做 exclusive prefix composition。", "tree 层数是 logarithmic；每层可并行 combine。", "prefix[k]=Tₖ₋₁∘…∘T₀", "z₀=0 时，chunk input 就是 prefix.q。"],
    ["warp time scan", "chunk 内还可以更并行吗？", "把同一 mode 的 timesteps 分给 32 lanes。", "warp shuffle 传播 affine prefixes；布局比 one-lane-one-mode 更复杂。", "lane ℓ ↔ time t+ℓ", "这是 advanced path，不是第一个 correctness kernel。"],
    ["time × mode tile", "如何同时利用两个维度？", "CTA 覆盖 C timesteps × T modes。", "需要共同设计 registers、shared memory、warps 与 output layout。", "tile = [C time] × [T modes]", "二维 tile 是吞吐优化空间，不是固定唯一解。"],
    ["decode", "为什么 decode 不需要 time scan？", "当前只有一个新 token，没有未来 timesteps。", "并行 mode、广播 control、更新 state cache。", "one token × M modes", "decode 优化 latency，而不是整段 throughput。"],
    ["training / prefill", "整段序列已知时应该做什么？", "先 GEMM 全部 writes，再做 chunked recurrence / scan。", "并行 batch×time×mode，必要时 materialize all outputs。", "GEMM → local summaries → prefix → outputs", "同一数学需要 throughput-first backend。"],
    ["RG-LRU", "SAMU 的 compact control 与 RG-LRU 有何不同？", "RG-LRU 通常产生 mode-wise real retention gates。", "其 gates 可 fuse/recompute；不能把数学 O(LM) 直接等同 HBM traffic。", "shared c,d vs mode-wise gates", "对比必须匹配 2M real state scalars。"],
    ["Mamba-3", "为什么它是必须认真对待的 baseline？", "Mamba-3 有 complex state、MIMO 与成熟 hardware-aware implementation。", "更多核心训练工作可组织成 matmul/SSD；SAMU compact metadata 不保证更快。", "structure ≠ benchmark result", "公平实验才决定 Pareto frontier。"],
    ["Final architecture", "最终应怎样拆解 SAMU GPU backend？", "dense write、compact control、decode recurrence、training scan 四块。", "V0 correctness → V1 baseline → V2 control → V3 scan → profiling-gated V4。", "GEMM + coherent transition + dual backend", "先建立可证伪、可 profiling 的系统路线。"]
  ];

  let lessonIndex = 0;
  function renderLesson() {
    const [title, question, example, gpu, formula, takeaway] = lessons[lessonIndex];
    $("#lessonProgress").textContent = `${String(lessonIndex + 1).padStart(2, "0")} / ${lessons.length}`;
    $("#lessonTrack").style.width = `${((lessonIndex + 1) / lessons.length) * 100}%`;
    $("#lessonContent").innerHTML = `
      <span class="lesson-index">STEP ${String(lessonIndex + 1).padStart(2, "0")}</span>
      <h2 id="lessonTitle">${title}</h2>
      <div class="lesson-question"><b>问题 →</b> ${question}</div>
      <div class="lesson-mini"><div class="lesson-example"><small>极小例子 / 直觉</small><p>${example}</p></div><div class="lesson-gpu"><small>GPU 视角</small><p>${gpu}</p></div></div>
      <div class="lesson-formula">${formula}</div>
      <p class="lesson-takeaway">核心结论：${takeaway}</p>`;
    $("#lessonPrev").disabled = lessonIndex === 0;
    $("#lessonNext").textContent = lessonIndex === lessons.length - 1 ? "完成并返回总览" : "下一步 →";
    $("#lessonDots").innerHTML = lessons.map((_, index) => `<i class="${index === lessonIndex ? "is-active" : ""}"></i>`).join("");
  }

  function openLearning() {
    $("#learningOverlay").hidden = false;
    document.body.style.overflow = "hidden";
    $$(".mode-button").forEach(button => button.classList.toggle("is-active", button.dataset.mode === "learn"));
    renderLesson();
    $("#closeLearning").focus();
  }

  function closeLearning() {
    $("#learningOverlay").hidden = true;
    document.body.style.overflow = "";
    $$(".mode-button").forEach(button => button.classList.toggle("is-active", button.dataset.mode === "overview"));
    $(".mode-button[data-mode='learn']").focus();
  }

  function setupLearning() {
    $$(".mode-button").forEach(button => button.addEventListener("click", () => button.dataset.mode === "learn" ? openLearning() : closeLearning()));
    $("#closeLearning").addEventListener("click", closeLearning);
    $("#lessonPrev").addEventListener("click", () => { lessonIndex = Math.max(0, lessonIndex - 1); renderLesson(); });
    $("#lessonNext").addEventListener("click", () => {
      if (lessonIndex === lessons.length - 1) closeLearning();
      else { lessonIndex += 1; renderLesson(); }
    });
    $("#learningOverlay").addEventListener("click", event => { if (event.target === $("#learningOverlay")) closeLearning(); });
    document.addEventListener("keydown", event => {
      if ($("#learningOverlay").hidden) return;
      if (event.key === "Escape") closeLearning();
      if (event.key === "ArrowRight") { event.preventDefault(); $("#lessonNext").click(); }
      if (event.key === "ArrowLeft") { event.preventDefault(); $("#lessonPrev").click(); }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderMath();
    setupSectionObserver();
    setupBackendToggle();
    setupWarp();
    setupTimeModeGrid();
    setupToy();
    setupScan();
    setupPerformance();
    setupLearning();
    updateScroll();
  });
  window.addEventListener("scroll", updateScroll, { passive: true });
})();
