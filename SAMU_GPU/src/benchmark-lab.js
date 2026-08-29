const $ = selector => document.querySelector(selector);
const DATA_ROOT = "benchmark_results_equal_kernel";
const ns = "http://www.w3.org/2000/svg";

const colors = {
  "SAMU Triton": "var(--state)",
  "RG-LRU Triton": "var(--rglru)",
  "Official RG-LRU": "var(--motion)",
  "Official Mamba-3": "var(--dynamic)",
  "SAMU / RG ratio": "var(--state)",
  "SAMU / Mamba ratio": "var(--dynamic)",
  "SAMU C8": "#b48927",
  "SAMU C16": "var(--write)",
  "SAMU C32": "var(--state)",
  "RG-LRU C8": "#8c6b58",
  "RG-LRU C16": "var(--motion)",
  "RG-LRU C32": "var(--rglru)"
};

let summary = {};
let rows = [];
let audit = {};
let environment = {};
let profileData = {};
let activeTab = "protocol";

const fmt = (value, digits = 2) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value))
  ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
  : "N/A";
const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[char]));
const svgEl = (tag, attrs = {}, text = "") => {
  const node = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  return node;
};
const measured = predicate => rows.filter(row => row.status === "measured" && predicate(row));
const one = predicate => measured(predicate)[0] || null;
const clonePoint = (row, series, x) => row ? { ...row, _series: series, _x: Number(x) } : null;

function pairedPoint(workload, length, batch = 1) {
  const base = row => row.track === "A_equal_triton" && row.workload === workload
    && row.length === length && row.batch === batch && row.d_model === 128 && row.modes === 64;
  return {
    samu: one(row => base(row) && row.model === "samu" && row.backend === (workload === "decode" ? "triton_fused_decode" : "triton_auto")),
    rglru: one(row => base(row) && row.model === "rglru" && row.backend === (workload === "decode" ? "triton_fused_decode" : "triton_auto"))
  };
}

function ratioCard(label, samu, other, otherLabel, caveat = "") {
  if (!samu || !other) return `<div class="headline-result"><span class="metric-label">${esc(label)}</span><strong class="metric">N/A</strong><p>缺少匹配的实测点。</p></div>`;
  const ratio = other.median_ms / samu.median_ms;
  const tied = Math.abs(other.median_ms - samu.median_ms) < 1e-12;
  const samuWins = ratio >= 1;
  return `<div class="headline-result ${tied ? "" : samuWins ? "winner-samu" : "winner-other"}">
    <span class="metric-label">${esc(label)}</span>
    <strong class="metric">${tied ? "TIE" : `${fmt(samuWins ? ratio : 1 / ratio, 2)}×`}</strong>
    <p>${tied ? "事件分辨率内持平" : samuWins ? "SAMU 更快" : `${esc(otherLabel)} 更快`}<br>SAMU ${fmt(samu.median_ms, 4)} ms · ${esc(otherLabel)} ${fmt(other.median_ms, 4)} ms${caveat ? `<br><small>${esc(caveat)}</small>` : ""}</p>
  </div>`;
}

function renderHeadline() {
  $("#headline-env").textContent = `${environment.gpu || "GPU unknown"} · CUDA ${environment.torch_cuda || "?"} · Triton ${environment.triton || "?"}`;
  const long = pairedPoint("prefill", 65536);
  const decode = pairedPoint("decode", 1);
  const samuLong = long.samu;
  const mambaLong = one(row => row.track === "B_official" && row.model === "mamba3" && row.workload === "prefill" && row.batch === 1 && row.length === 65536);
  $("#headline-results").innerHTML = [
    ratioCard("Equal Triton · L=65,536", long.samu, long.rglru, "RG-LRU"),
    ratioCard("Equal Triton · decode B=1", decode.samu, decode.rglru, "RG-LRU"),
    ratioCard("Official Mamba-3 · L=65,536", samuLong, mambaLong, "Mamba-3", "同宽参考；参数/state 不匹配")
  ].join("");
}

function currentView() {
  const experiment = $("#filter-experiment").value;
  const common = row => row.batch === 1 && row.d_model === 128 && row.modes === 64;

  if (experiment === "equal-length") {
    const data = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.backend === "triton_auto" && common(row))
      .map(row => clonePoint(row, row.model === "samu" ? "SAMU Triton" : "RG-LRU Triton", row.length));
    return { data, title: "Equal Triton prefill · length sweep", xLabel: "sequence length L", protocol: "d=128 · 16,772 vs 16,768 params · 512 B state", ratioModel: "rglru" };
  }
  if (experiment === "official-length") {
    const samu = measured(row => row.track === "A_equal_triton" && row.model === "samu" && row.workload === "prefill" && row.backend === "triton_auto" && common(row))
      .map(row => clonePoint(row, "SAMU Triton", row.length));
    const official = measured(row => row.track === "B_official" && row.workload === "prefill" && common(row))
      .map(row => clonePoint(row, row.model === "mamba3" ? "Official Mamba-3" : "Official RG-LRU", row.length));
    return { data: [...samu, ...official], title: "Official implementation context · length sweep", xLabel: "sequence length L", protocol: "same width only · Mamba params/state are not matched", ratioModel: "mamba3" };
  }
  if (experiment === "batch") {
    const data = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.length === 512 && row.backend === "triton_auto" && row.d_model === 128 && row.modes === 64)
      .map(row => clonePoint(row, row.model === "samu" ? "SAMU Triton" : "RG-LRU Triton", row.batch));
    return { data, title: "Equal Triton prefill · batch scaling at L=512", xLabel: "batch B", protocol: "same kernel class · same recurrent-state bytes", ratioModel: "rglru" };
  }
  if (experiment === "decode") {
    const data = measured(row => row.track === "A_equal_triton" && row.workload === "decode" && row.backend === "triton_fused_decode" && row.d_model === 128 && row.modes === 64)
      .map(row => clonePoint(row, row.model === "samu" ? "SAMU Triton" : "RG-LRU Triton", row.batch));
    return { data, title: "Equal Triton fused decode · batch scaling", xLabel: "decode batch B", protocol: "one CUDA launch on both sides", ratioModel: "rglru" };
  }
  const data = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.batch === 1 && row.length === 65536 && row.backend.startsWith("triton_chunk_c"))
    .map(row => clonePoint(row, row.model === "samu" ? "SAMU Triton" : "RG-LRU Triton", row.chunk_size));
  return { data, title: "Chunk-size sweep · B1 L=65,536", xLabel: "chunk size C", protocol: "explicit C8 / C16 / C32 paths", ratioModel: "rglru" };
}

function speedupPoints(data, ratioModel) {
  const grouped = new Map();
  data.forEach(row => {
    const group = grouped.get(row._x) || {};
    if (row.model === "samu") group.samu = group.samu?.median_ms < row.median_ms ? group.samu : row;
    if (row.model === ratioModel) group.other = group.other?.median_ms < row.median_ms ? group.other : row;
    grouped.set(row._x, group);
  });
  return [...grouped.entries()].filter(([, group]) => group.samu && group.other).map(([x, group]) => ({
    id: `ratio-${ratioModel}-${x}`,
    status: "derived",
    _series: ratioModel === "mamba3" ? "SAMU / Mamba ratio" : "SAMU / RG ratio",
    _x: Number(x),
    speedup: group.other.median_ms / group.samu.median_ms,
    samu: group.samu,
    other: group.other,
    otherLabel: ratioModel === "mamba3" ? "Official Mamba-3" : "RG-LRU Triton"
  }));
}

const metricValue = (row, metric) => metric === "speedup" ? Number(row.speedup) : Number(row[metric]);
function metricLabel(value, metric) {
  if (metric === "tokens_per_second") return `${fmt(value / 1e6, 2)}M tok/s`;
  if (metric === "speedup") return `${fmt(value, 2)}×`;
  return `${fmt(value, value < 1 ? 4 : 2)} ms`;
}

function renderChart() {
  const svg = $("#benchmark-chart");
  const view = currentView();
  let data = view.data;
  let metric = $("#filter-metric").value;
  if (metric === "speedup") data = speedupPoints(data, view.ratioModel);

  $("#chart-title").textContent = view.title;
  $("#chart-protocol").textContent = view.protocol;
  svg.replaceChildren();
  if (!data.length) {
    svg.append(svgEl("text", { x: 480, y: 250, "text-anchor": "middle", fill: "var(--muted)" }, "No matching measured points"));
    $("#chart-caption").textContent = "该组合没有可比实测点。";
    return;
  }

  const W = 960, H = 520, margin = { l: 92, r: 30, t: 34, b: 82 };
  const xValues = [...new Set(data.map(row => row._x))].sort((a, b) => a - b);
  const values = data.map(row => metricValue(row, metric)).filter(value => Number.isFinite(value) && value > 0);
  const lo = Math.min(...values), hi = Math.max(...values);
  const logY = hi / Math.max(lo, 1e-12) > 20;
  const x = value => margin.l + (xValues.length === 1 ? .5 : xValues.indexOf(value) / (xValues.length - 1)) * (W - margin.l - margin.r);
  const scale = value => logY
    ? (Math.log(value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo) || 1)
    : (value - lo) / (hi - lo || 1);
  const y = value => H - margin.b - scale(value) * (H - margin.t - margin.b);

  for (let tick = 0; tick <= 5; tick++) {
    const yy = margin.t + tick * (H - margin.t - margin.b) / 5;
    const value = logY
      ? Math.exp(Math.log(hi) + (Math.log(lo) - Math.log(hi)) * tick / 5)
      : hi + (lo - hi) * tick / 5;
    svg.append(svgEl("line", { x1: margin.l, y1: yy, x2: W - margin.r, y2: yy, stroke: "var(--line)", "stroke-width": 1 }));
    svg.append(svgEl("text", { x: margin.l - 12, y: yy + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono" }, metricLabel(value, metric)));
  }
  xValues.forEach(value => svg.append(svgEl("text", { x: x(value), y: H - margin.b + 27, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono" }, fmt(value, 0))));

  const seriesNames = [...new Set(data.map(row => row._series))];
  seriesNames.forEach((name, index) => {
    const series = data.filter(row => row._series === name).sort((a, b) => a._x - b._x);
    const stroke = colors[name] || "var(--ink)";
    const path = series.map((row, point) => `${point ? "L" : "M"}${x(row._x)},${y(metricValue(row, metric))}`).join(" ");
    svg.append(svgEl("path", { d: path, fill: "none", stroke, "stroke-width": 2.7 }));
    series.forEach(row => {
      const circle = svgEl("circle", { cx: x(row._x), cy: y(metricValue(row, metric)), r: 6, fill: stroke, stroke: "var(--paper)", "stroke-width": 2, tabindex: 0, role: "button" });
      circle.addEventListener("click", () => renderDetail(row, metric));
      circle.addEventListener("focus", () => renderDetail(row, metric));
      svg.append(circle);
    });
    const legendX = margin.l + index * Math.min(205, (W - margin.l - margin.r) / Math.max(seriesNames.length, 1));
    svg.append(svgEl("line", { x1: legendX, y1: H - 18, x2: legendX + 22, y2: H - 18, stroke, "stroke-width": 3 }));
    svg.append(svgEl("text", { x: legendX + 28, y: H - 14, fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono" }, name));
  });
  svg.append(svgEl("text", { x: (margin.l + W - margin.r) / 2, y: H - 42, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono" }, view.xLabel));
  $("#chart-caption").textContent = `${data.length} 个点 · ${metric} · ${logY ? "log" : "linear"} y-axis。速度比定义为对手 median / SAMU median；大于 1 表示 SAMU 更快。`;
  renderDetail(data[0], metric);
}

function renderDetail(row, metric) {
  if (row.status === "derived") {
    const tied = Math.abs(row.speedup - 1) < 1e-12;
    const winner = row.speedup >= 1 ? "SAMU" : row.otherLabel;
    $("#benchmark-detail").innerHTML = `<p class="aside-title">Pairwise ratio</p><span class="evidence derived">Derived</span><strong class="big">${metricLabel(row.speedup, "speedup")}</strong><p>${tied ? "事件分辨率内持平" : `${esc(winner)} 更快`}；比值为 ${esc(row.otherLabel)} median / SAMU median。</p><dl><dt>SAMU</dt><dd>${fmt(row.samu.median_ms, 4)} ms</dd><dt>${esc(row.otherLabel)}</dt><dd>${fmt(row.other.median_ms, 4)} ms</dd></dl>`;
    return;
  }
  const sfu = row.sfu_static_counts || {};
  const raw = row.raw_samples_ms || [];
  $("#benchmark-detail").innerHTML = `<p class="aside-title">Selected point</p><span class="evidence measured">Measured</span><strong class="big">${metricLabel(metricValue(row, metric), metric)}</strong><p>${esc(row._series)}</p><dl>
    <dt>track</dt><dd>${esc(row.track)}</dd><dt>shape</dt><dd>B${row.batch} · L${row.length} · d${row.d_model}</dd>
    <dt>backend</dt><dd>${esc(row.backend)}</dd><dt>parameters</dt><dd>${fmt(row.parameter_count, 0)}</dd><dt>FP32 state / B</dt><dd>${fmt(row.state_bytes_per_batch, 0)} B</dd>
    <dt>median</dt><dd>${fmt(row.median_ms, 4)} ms</dd><dt>P10 / P95</dt><dd>${fmt(row.p10_ms, 4)} / ${fmt(row.p95_ms, 4)}</dd>
    <dt>logical lower bound</dt><dd>${fmt(row.logical_bytes_lower_bound / 1048576, 2)} MiB</dd><dt>logical effective rate</dt><dd>${fmt(row.logical_effective_gbps, 2)} GB/s*</dd>
    <dt>static exp/sigmoid</dt><dd>${fmt(sfu.exp_or_sigmoid_per_token, 0)} / token</dd><dt>static sqrt</dt><dd>${fmt(sfu.sqrt_per_token, 0)} / token</dd><dt>expected launches</dt><dd>${fmt(row.expected_cuda_launches, 0)}</dd>
  </dl><p class="code-label">raw CUDA-event samples (ms)</p><p class="mono raw-samples">${raw.map(value => fmt(value, 4)).join(" · ")}</p><a href="${DATA_ROOT}/raw/${encodeURIComponent(row.id)}.json">Open raw JSON →</a><p class="fairness-inline">*逻辑字节 / latency，不是硬件 DRAM counter。</p>`;
}

function compiledKernelSummary(profile) {
  if (profile.name.startsWith("mamba3_")) return `N/A* (${profile.triton_compiled_kernels?.length || 0} cached candidates)`;
  return (profile.triton_compiled_kernels || []).map(kernel => `${kernel.name.replace(/^_/, "")}: ${kernel.registers_per_thread}r · ${fmt(kernel.derived_occupancy * 100, 1)}% occ · ${kernel.spills_per_thread}s`).join("<br>") || "N/A";
}

function renderProfile() {
  const profiles = profileData.profiles || [];
  const equalRows = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.batch === 1 && row.length === 2048 && row.backend === "triton_auto");
  const samu = equalRows.find(row => row.model === "samu");
  const rg = equalRows.find(row => row.model === "rglru");
  return `<div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>Operation</th><th>actual CUDA launches</th><th>kernel sum</th><th>register / occupancy / spill</th></tr></thead><tbody>${profiles.map(profile => `<tr><td>${esc(profile.name)}</td><td>${profile.cuda_kernel_launch_count}</td><td>${fmt(profile.self_cuda_total_us, 0)} μs</td><td class="mono">${compiledKernelSummary(profile)}</td></tr>`).join("")}</tbody></table></div>
    <div class="coverage-grid metric-audit"><div><span class="context-label">Static SFU work · C32 L2048</span><strong>${fmt(samu?.sfu_static_counts?.exp_or_sigmoid_per_token, 0)} vs ${fmt(rg?.sfu_static_counts?.exp_or_sigmoid_per_token, 0)}</strong><p>SAMU vs RG-LRU exp/sigmoid per token；RG-LRU 另有 ${fmt(rg?.sfu_static_counts?.sqrt_per_token, 0)} sqrt。是源码静态计数，不是 SFU utilization。</p></div><div><span class="context-label">Logical effective rate</span><strong>${fmt(samu?.logical_effective_gbps, 2)} vs ${fmt(rg?.logical_effective_gbps, 2)} GB/s</strong><p>逻辑流量下界除以 latency；不等于实测 HBM/DRAM 带宽。</p></div><div><span class="context-label">Hardware counters</span><strong>N/A</strong><p>镜像无 Nsight Compute；DRAM bytes、带宽、SFU/SM utilization 保持 null，不用推算值冒充实测。</p></div></div>
    <p class="fairness-inline">Occupancy 是根据 compiler metadata 与 RTX 3090 SM 资源上限推导的估计值，忽略寄存器分配粒度和 scheduler 限制。Mamba 的 cache 同时存在 37 个 autotune candidates，torch.profiler 不能把实际事件唯一关联到某个候选，因此 register/occupancy 明确标 N/A*。</p>`;
}

function renderLab() {
  const panel = $("#lab-panel");
  if (activeTab === "protocol") {
    panel.innerHTML = `<div class="coverage-grid"><div><span class="context-label">Official RG-LRU equations</span><strong>Eq. 1–4</strong><p>r=σ(Wₐx+bₐ), i=σ(Wₓx+bₓ), a=a_base^(8r), h=a·h_prev+√(1−a²)·(i·x)。reset 位置强制 a=0、write multiplier=1。</p></div><div><span class="context-label">Equal track</span><strong>16,772 vs 16,768</strong><p>d=128；参数只差 4，recurrent state 均为 128 FP32 scalars = 512 B / batch。</p></div><div><span class="context-label">Timing</span><strong>Triton driver events</strong><p>每个 timed call 前清 L2；compile、packing、shape dispatch 排除，raw event samples 全部保留。</p></div></div><p class="fairness-inline">Track A 才用于 SAMU↔RG-LRU 架构结论。Track B 的官方 Mamba-3 是同宽 best-native 背景线：118,920 参数、66,816 B native cache，并不与 SAMU 参数或 state 匹配。</p>`;
  } else if (activeTab === "profile") {
    panel.innerHTML = renderProfile();
  } else if (activeTab === "correctness") {
    panel.innerHTML = `<div class="coverage-grid"><div><span class="context-label">RG-LRU serial</span><strong>${fmt(audit.rglru_serial_max_abs, 8)}</strong><p>BF16 output max abs；FP32 cache max abs ${fmt(audit.rglru_serial_cache_max_abs, 9)}。</p></div><div><span class="context-label">RG-LRU C16</span><strong>${fmt(audit.rglru_chunk16_max_abs, 7)}</strong><p>与固定 commit 官方实现逐值对照；cache max abs ${fmt(audit.rglru_chunk16_cache_max_abs, 9)}。</p></div><div><span class="context-label">RG-LRU fused decode</span><strong>${fmt(audit.rglru_adversarial_decode_max_abs, 8)}</strong><p>对抗尺度 sweep 的 output max abs；cache max abs ${fmt(audit.rglru_adversarial_decode_cache_max_abs, 9)}。</p></div></div><p class="fairness-inline">SAMU auto-short output max abs ${fmt(audit.samu_auto_short_max_abs, 8)}，cache ${fmt(audit.samu_auto_short_cache_max_abs, 9)}；|d|≤${fmt(audit.samu_phase_delta_bound_rad, 8)} rad 内 sin/cos polynomial max abs 为 ${fmt(audit.samu_sin_polynomial_max_abs, 10)} / ${fmt(audit.samu_cos_polynomial_max_abs, 10)}。</p>`;
  } else if (activeTab === "environment") {
    const fields = { GPU: environment.gpu, "Board / driver": environment.driver_and_board, "Compute capability": environment.compute_capability, "SM count": environment.sm_count, "CUDA runtime": environment.torch_cuda, PyTorch: environment.torch, Triton: environment.triton, Python: environment.python, "RecurrentGemma commit": environment.source_commits?.recurrentgemma, "Mamba commit": environment.source_commits?.mamba, Timestamp: environment.timestamp_utc };
    panel.innerHTML = `<dl class="environment-grid">${Object.entries(fields).map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl>`;
  } else {
    const measuredCount = rows.filter(row => row.status === "measured").length;
    const unsupportedCount = rows.filter(row => row.status === "unsupported").length;
    panel.innerHTML = `<p><b>Source of truth:</b> ${rows.length} rows · ${measuredCount} measured · ${unsupportedCount} unsupported · ${rows.length - measuredCount - unsupportedCount} failed。聚合文件不补数、不平滑、不外推。</p><p><a class="button" href="${DATA_ROOT}/summary.json">Open summary.json</a> <a class="button" href="${DATA_ROOT}/summary.csv">Download summary.csv</a> <a class="button" href="${DATA_ROOT}/profiles.json">Open profiles.json</a> <a class="button" href="${DATA_ROOT}/run_manifest.json">Run manifest</a></p>`;
  }
}

export async function initBenchmarkLab() {
  try {
    [summary, profileData] = await Promise.all([
      fetch(`${DATA_ROOT}/summary.json`).then(response => {
        if (!response.ok) throw new Error(`summary.json HTTP ${response.status}`);
        return response.json();
      }),
      fetch(`${DATA_ROOT}/profiles.json`).then(response => {
        if (!response.ok) throw new Error(`profiles.json HTTP ${response.status}`);
        return response.json();
      })
    ]);
    rows = summary.rows || [];
    environment = summary.environment || {};
    audit = summary.equation_audit || {};
    renderHeadline();
    renderChart();
    renderLab();
    $("#benchmark-controls").addEventListener("change", renderChart);
    document.querySelector(".lab-tabs").addEventListener("click", event => {
      const button = event.target.closest("[data-lab-tab]");
      if (!button) return;
      activeTab = button.dataset.labTab;
      document.querySelectorAll("[data-lab-tab]").forEach(item => item.classList.toggle("active", item === button));
      renderLab();
    });
  } catch (error) {
    $("#headline-results").innerHTML = `<div class="headline-result"><strong>Benchmark data unavailable</strong><p>${esc(error.message)}。请通过 HTTP server 打开页面。</p></div>`;
    $("#chart-caption").textContent = `Load error: ${error.message}`;
  }
}
