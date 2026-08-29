const $ = selector => document.querySelector(selector);
const fmt = (value, digits = 2) => Number.isFinite(value)
  ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits }) : "N/A";
const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[char]));
const ns = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}, text = "") => {
  const node = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  return node;
};

const DATA_ROOT = "benchmark_results_samu_rg";
const colors = {
  "SAMU · fastest measured": "var(--state)",
  "SAMU · auto": "var(--state)",
  "SAMU · serial": "var(--motion)",
  "SAMU · C8": "#b48927",
  "SAMU · C16": "var(--write)",
  "SAMU · C32": "var(--dynamic)",
  "Official-source RG-LRU": "var(--rglru)",
  "Measured speedup": "var(--state)"
};

let rows = [];
let environment = {};
let correctness = {};
let tab = "protocol";

const measured = predicate => rows.filter(row => row.status === "measured" && predicate(row));
const best = values => values.reduce((winner, row) => !winner || row.median_ms < winner.median_ms ? row : winner, null);
const isBase = row => row.d_model === 128 && row.modes === 64;
const cloneSeries = (row, series, xKey) => row ? { ...row, _series: series, _x: row[xKey] } : null;

function pairAt(predicate) {
  const candidates = measured(predicate);
  return {
    samu: best(candidates.filter(row => row.model === "samu")),
    rglru: best(candidates.filter(row => row.model === "rglru"))
  };
}

function renderHeadline() {
  $("#headline-env").textContent = `${environment.gpu || "GPU unknown"} · CUDA ${environment.torch_cuda || "?"} · PyTorch ${environment.torch || "?"}`;
  const points = [
    { label: "Short prefill · L=128", pair: pairAt(row => row.workload === "forward" && row.batch === 1 && row.length === 128 && isBase(row)) },
    { label: "Long prefill · L=65,536", pair: pairAt(row => row.workload === "forward" && row.batch === 1 && row.length === 65536 && isBase(row)) },
    { label: "Fused decode · B=1", pair: pairAt(row => row.workload === "decode" && row.batch === 1 && isBase(row)) }
  ];
  $("#headline-results").innerHTML = points.map(({ label, pair }) => {
    if (!pair.samu || !pair.rglru) return `<div class="headline-result"><span class="metric-label">${label}</span><strong class="metric">N/A</strong><p>缺少匹配的实测点。</p></div>`;
    const ratio = pair.rglru.median_ms / pair.samu.median_ms;
    return `<div class="headline-result"><span class="metric-label">${label}</span><strong class="metric">${fmt(ratio, 1)}×</strong><p>SAMU ${fmt(pair.samu.median_ms, 4)} ms · RG-LRU ${fmt(pair.rglru.median_ms, 2)} ms<br><span class="mono">${esc(pair.samu.backend)}</span></p></div>`;
  }).join("");
}

function decoratePairSeries(sourceRows, xKey, samuLabel = "SAMU · auto") {
  return sourceRows.map(row => cloneSeries(row, row.model === "samu" ? samuLabel : "Official-source RG-LRU", xKey));
}

function viewData() {
  const experiment = $("#filter-experiment").value;
  if (experiment === "length") {
    const values = [128, 512, 2048, 8192, 32768, 65536];
    const data = [];
    values.forEach(length => {
      const pair = pairAt(row => row.workload === "forward" && row.batch === 1 && row.length === length && isBase(row));
      data.push(cloneSeries(pair.samu, "SAMU · fastest measured", "length"), cloneSeries(pair.rglru, "Official-source RG-LRU", "length"));
    });
    return { data: data.filter(Boolean), xLabel: "sequence length L", title: "Prefill · latency vs sequence length" };
  }
  if (experiment === "batch") {
    const source = measured(row => row.workload === "forward" && row.length === 512 && /triton_auto|official_pytorch_source/.test(row.backend) && isBase(row));
    return { data: decoratePairSeries(source, "batch"), xLabel: "batch size B", title: "Prefill · batch scaling at L=512" };
  }
  if (experiment === "state") {
    const source = measured(row => row.workload === "forward" && row.batch === 1 && row.length === 512 && /triton_auto|official_pytorch_source/.test(row.backend));
    return { data: decoratePairSeries(source, "modes"), xLabel: "SAMU complex modes M · RG width = 2M", title: "Prefill · equal-state scaling at L=512" };
  }
  if (experiment === "decode") {
    const source = measured(row => row.workload === "decode" && isBase(row));
    return { data: decoratePairSeries(source, "batch"), xLabel: "decode batch B", title: "One-token update · batch scaling" };
  }
  const source = measured(row => row.model === "samu" && row.workload === "forward" && row.batch === 1 && isBase(row) && (row.backend === "triton_serial" || row.backend.startsWith("triton_chunk_c")));
  const labels = { triton_serial: "SAMU · serial", triton_chunk_c8: "SAMU · C8", triton_chunk_c16: "SAMU · C16", triton_chunk_c32: "SAMU · C32" };
  return { data: source.map(row => cloneSeries(row, labels[row.backend], "length")), xLabel: "sequence length L", title: "SAMU · serial-to-chunk crossover" };
}

function speedupRows(data) {
  const grouped = new Map();
  data.forEach(row => {
    const group = grouped.get(row._x) || {};
    if (row.model === "rglru") group.rglru = row;
    if (row.model === "samu") group.samu = row;
    grouped.set(row._x, group);
  });
  return [...grouped.entries()].filter(([, group]) => group.samu && group.rglru).map(([x, group]) => ({
    id: `speedup-${x}`, status: "derived", model: "derived",
    backend: "RG-LRU median / SAMU median", _series: "Measured speedup", _x: Number(x),
    speedup: group.rglru.median_ms / group.samu.median_ms, samu: group.samu, rglru: group.rglru
  }));
}

const metricValue = (row, metric) => metric === "speedup" ? row.speedup : Number(row[metric]);
function formatMetric(value, metric) {
  if (metric === "tokens_per_second") return `${fmt(value / 1e6, 2)}M tok/s`;
  if (metric === "speedup") return `${fmt(value, 1)}×`;
  return `${fmt(value, value < 1 ? 4 : 2)} ms`;
}

function renderChart() {
  const svg = $("#benchmark-chart");
  const view = viewData();
  let metric = $("#filter-metric").value;
  let data = view.data;
  if (metric === "speedup") {
    data = speedupRows(data);
    if (!data.length) metric = "median_ms";
  }
  $("#chart-title").textContent = view.title;
  svg.replaceChildren();
  if (!data.length) {
    svg.append(svgEl("text", { x: 480, y: 250, "text-anchor": "middle", fill: "var(--muted)" }, "No measured points for this view"));
    return;
  }
  const W = 960, H = 520, margin = { l: 88, r: 28, t: 32, b: 78 };
  const xValues = [...new Set(data.map(row => Number(row._x)))].sort((a, b) => a - b);
  const values = data.map(row => metricValue(row, metric)).filter(value => Number.isFinite(value) && value > 0);
  const lo = Math.min(...values), hi = Math.max(...values);
  const logY = hi / Math.max(lo, 1e-12) > 20;
  const x = value => margin.l + (xValues.length === 1 ? .5 : xValues.indexOf(Number(value)) / (xValues.length - 1)) * (W - margin.l - margin.r);
  const normalized = value => logY ? (Math.log(value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo) || 1) : (value - lo) / (hi - lo || 1);
  const y = value => H - margin.b - normalized(value) * (H - margin.t - margin.b);
  for (let index = 0; index <= 5; index++) {
    const yy = margin.t + index * (H - margin.t - margin.b) / 5;
    const value = logY ? Math.exp(Math.log(hi) + (Math.log(lo) - Math.log(hi)) * index / 5) : hi + (lo - hi) * index / 5;
    svg.append(svgEl("line", { x1: margin.l, y1: yy, x2: W - margin.r, y2: yy, stroke: "var(--line)", "stroke-width": 1 }));
    svg.append(svgEl("text", { x: margin.l - 12, y: yy + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono" }, formatMetric(value, metric)));
  }
  xValues.forEach(value => svg.append(svgEl("text", { x: x(value), y: H - margin.b + 28, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono" }, fmt(value, 0))));
  const names = [...new Set(data.map(row => row._series))];
  names.forEach((name, index) => {
    const series = data.filter(row => row._series === name).sort((a, b) => a._x - b._x);
    const path = series.map((row, point) => `${point ? "L" : "M"}${x(row._x)},${y(metricValue(row, metric))}`).join(" ");
    svg.append(svgEl("path", { d: path, fill: "none", stroke: colors[name] || "var(--ink)", "stroke-width": 2.7 }));
    series.forEach(row => {
      const circle = svgEl("circle", { cx: x(row._x), cy: y(metricValue(row, metric)), r: 6, fill: colors[name] || "var(--ink)", stroke: "var(--paper)", "stroke-width": 2, tabindex: 0 });
      circle.addEventListener("click", () => renderDetail(row, metric));
      circle.addEventListener("focus", () => renderDetail(row, metric));
      svg.append(circle);
    });
    const legendX = margin.l + index * Math.min(230, (W - margin.l - margin.r) / Math.max(1, names.length));
    svg.append(svgEl("line", { x1: legendX, y1: H - 18, x2: legendX + 22, y2: H - 18, stroke: colors[name] || "var(--ink)", "stroke-width": 3 }));
    svg.append(svgEl("text", { x: legendX + 28, y: H - 14, fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono" }, name));
  });
  svg.append(svgEl("text", { x: (margin.l + W - margin.r) / 2, y: H - 40, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono" }, view.xLabel));
  $("#chart-caption").textContent = `${data.length} 个实测/实测比值点 · ${metric} · ${logY ? "log" : "linear"} y-axis。点选圆点查看配置与 raw samples。`;
  renderDetail(data[0], metric);
}

function renderDetail(row, metric) {
  if (row.status === "derived") {
    $("#benchmark-detail").innerHTML = `<p class="aside-title">Measured ratio</p><span class="evidence derived">Derived</span><strong class="big">${formatMetric(row.speedup, "speedup")}</strong><p>同一个实测 shape 的 RG-LRU median / SAMU median。</p><dl><dt>SAMU</dt><dd>${fmt(row.samu.median_ms, 4)} ms</dd><dt>RG-LRU</dt><dd>${fmt(row.rglru.median_ms, 3)} ms</dd><dt>SAMU backend</dt><dd>${esc(row.samu.backend)}</dd></dl><p class="fairness-inline">这是实现速度比，不是对未知 optimized RG-LRU kernel 的外推。</p>`;
    return;
  }
  const samples = row.raw_samples_ms || [];
  $("#benchmark-detail").innerHTML = `<p class="aside-title">Selected point</p><span class="evidence measured">Measured</span><strong class="big">${formatMetric(metricValue(row, metric), metric)}</strong><p>${esc(row._series)}</p><dl><dt>shape</dt><dd>B${row.batch} · L${row.length} · d${row.d_model} · M${row.modes}</dd><dt>backend</dt><dd>${esc(row.backend)}</dd><dt>state / batch</dt><dd>${fmt(row.state_bytes_per_batch, 0)} B</dd><dt>median</dt><dd>${fmt(row.median_ms, 4)} ms</dd><dt>P10 / P95</dt><dd>${fmt(row.p10_ms, 4)} / ${fmt(row.p95_ms, 4)}</dd><dt>setup</dt><dd>${fmt(row.compile_setup_seconds, 2)} s excluded</dd><dt>samples × inner</dt><dd>${row.samples} × ${row.inner_iterations}</dd></dl><p class="code-label">raw CUDA-event samples (ms)</p><p class="mono raw-samples">${samples.map(value => fmt(value, 4)).join(" · ")}</p><a href="${DATA_ROOT}/raw/${encodeURIComponent(row.id)}.json">Open raw JSON →</a>`;
}

function renderLab() {
  const panel = $("#lab-panel");
  if (tab === "protocol") panel.innerHTML = `<div class="coverage-grid"><div><span class="context-label">Shape match</span><strong>d = 2M</strong><p>主对照 d=128、SAMU M=64 complex modes；RG-LRU width=128。</p></div><div><span class="context-label">State match</span><strong>512 B</strong><p>两边 recurrent state 都是 128 个 FP32 scalars / batch。</p></div><div><span class="context-label">Implementation scope</span><strong>source vs kernel</strong><p>RG-LRU 是未修改官方 PyTorch source；SAMU 是新写的 inference Triton kernel。</p></div></div><p class="fairness-inline">这组实验回答“我们的结构能否形成更快的当前执行路径”。它不能单独回答“同等成熟的两个 custom kernels 谁的数学架构更优”。</p>`;
  if (tab === "correctness") panel.innerHTML = `<div class="coverage-grid"><div><span class="context-label">Serial prefill</span><strong>${fmt(correctness.serial_max_abs, 7)}</strong><p>max abs error；mean ${fmt(correctness.serial_mean_abs, 9)}</p></div><div><span class="context-label">C16 chunk prefill</span><strong>${fmt(correctness.chunk16_max_abs, 7)}</strong><p>max abs error；mean ${fmt(correctness.chunk16_mean_abs, 9)}</p></div><div><span class="context-label">Fused decode</span><strong>${fmt(correctness.decode_max_abs, 7)}</strong><p>max abs error；mean ${fmt(correctness.decode_mean_abs, 8)}</p></div></div><p class="fairness-inline">Reference policy: ${esc(correctness.reference_policy)}。BF16 输出的最大误差落在其量化粒度内。</p>`;
  if (tab === "environment") panel.innerHTML = `<dl class="environment-grid">${Object.entries({ GPU: environment.gpu, "Board / driver": environment.driver_and_board, "Compute capability": environment.compute_capability, "SM count": environment.sm_count, "CUDA runtime": environment.torch_cuda, PyTorch: environment.torch, Triton: environment.triton, Python: environment.python, Timestamp: environment.timestamp_utc }).map(([key, value]) => `<div><dt>${key}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl>`;
  if (tab === "raw") panel.innerHTML = `<p><b>Source of truth:</b> ${rows.length} rows，${rows.filter(row => row.status === "measured").length} measured，${rows.filter(row => row.status !== "measured").length} failed。聚合文件不补数、不平滑、不外推。</p><p><a class="button" href="${DATA_ROOT}/summary.json">Open summary.json</a> <a class="button" href="${DATA_ROOT}/summary.csv">Download summary.csv</a> <a class="button" href="${DATA_ROOT}/run_manifest.json">Run manifest</a></p>`;
}

export async function initBenchmarkLab() {
  try {
    const summary = await fetch(`${DATA_ROOT}/summary.json`).then(response => response.json());
    rows = summary.rows || [];
    environment = summary.environment || {};
    correctness = summary.correctness || {};
    renderHeadline();
    renderChart();
    renderLab();
    $("#benchmark-controls").addEventListener("change", renderChart);
    document.querySelector(".lab-tabs").addEventListener("click", event => {
      const button = event.target.closest("[data-lab-tab]");
      if (!button) return;
      tab = button.dataset.labTab;
      document.querySelectorAll("[data-lab-tab]").forEach(item => item.classList.toggle("active", item === button));
      renderLab();
    });
  } catch (error) {
    $("#headline-results").innerHTML = `<div class="headline-result"><strong>Benchmark data unavailable</strong><p>${esc(error.message)}。请通过 HTTP server 打开本页。</p></div>`;
    $("#chart-caption").textContent = `Load error: ${error.message}`;
  }
}
