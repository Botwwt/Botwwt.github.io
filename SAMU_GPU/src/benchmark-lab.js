const $ = selector => document.querySelector(selector);
const DATA_ROOT = "benchmark_results_equal_kernel";
const ns = "http://www.w3.org/2000/svg";

const colors = {
  "SAMU Triton": "var(--state)",
  "SAMU dense": "var(--state)",
  "RG-LRU Triton": "var(--rglru)",
  "RG‑LRU‑2": "var(--rglru)",
  "SAMU / RG ratio": "var(--state)",
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
  if (!samu || !other) return `<div class="headline-result"><h3>${esc(label)}</h3><p>缺少匹配的实测点。</p></div>`;
  const ratio = other.median_ms / samu.median_ms;
  const tied = Math.abs(other.median_ms - samu.median_ms) < 1e-12;
  const samuWins = ratio >= 1;
  return `<div class="headline-result ${tied ? "" : samuWins ? "winner-samu" : "winner-other"}">
    <h3>${esc(label)}</h3>
    <p><b>${tied ? "在计时分辨率内持平" : samuWins ? `SAMU 快 ${fmt(ratio, 2)} 倍` : `${esc(otherLabel)} 快 ${fmt(1 / ratio, 2)} 倍`}</b>。SAMU 为 ${fmt(samu.median_ms, 4)} 毫秒，${esc(otherLabel)} 为 ${fmt(other.median_ms, 4)} 毫秒。${caveat ? `<br><small>${esc(caveat)}</small>` : ""}</p>
  </div>`;
}

function renderHeadline() {
  $("#headline-env").textContent = `${environment.gpu || "GPU unknown"} · CUDA ${environment.torch_cuda || "?"} · Triton ${environment.triton || "?"}`;
  const long = pairedPoint("prefill", 65536);
  const decode = pairedPoint("decode", 1);
  $("#headline-results").innerHTML = [
    ratioCard("长序列预填充（B=1，L=65,536）", long.samu, long.rglru, "RG‑LRU"),
    ratioCard("单 token 解码（B=1）", decode.samu, decode.rglru, "RG‑LRU", "这个结果只代表 d=128 的一启动微内核；它明确提醒我们，SAMU 还没有在所有形状上占优。")
  ].join("");
}

function currentView() {
  const experiment = $("#filter-experiment").value;
  const common = row => row.batch === 1 && row.d_model === 128 && row.modes === 64;

  if (experiment === "equal-length") {
    const data = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.backend === "triton_auto" && common(row))
      .map(row => clonePoint(row, row.model === "samu" ? "SAMU dense" : "RG‑LRU‑2", row.length));
    return { data, title: "人工参数匹配轨道：预填充延迟随长度变化", xLabel: "序列长度 L", protocol: "d=128；RG 使用 2 个分块，不是论文默认 16 分块", ratioModel: "rglru" };
  }
  if (experiment === "batch") {
    const data = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.length === 512 && row.backend === "triton_auto" && row.d_model === 128 && row.modes === 64)
      .map(row => clonePoint(row, row.model === "samu" ? "SAMU dense" : "RG‑LRU‑2", row.batch));
    return { data, title: "L=512 时，预填充延迟随批量大小的变化", xLabel: "批量大小 B", protocol: "同类内核；相同递推状态字节数", ratioModel: "rglru" };
  }
  if (experiment === "decode") {
    const data = measured(row => row.track === "A_equal_triton" && row.workload === "decode" && row.backend === "triton_fused_decode" && row.d_model === 128 && row.modes === 64)
      .map(row => clonePoint(row, row.model === "samu" ? "SAMU dense" : "RG‑LRU‑2", row.batch));
    return { data, title: "融合单步解码随批量大小的变化", xLabel: "解码批量大小 B", protocol: "双方每一步都只启动一次 CUDA 内核", ratioModel: "rglru" };
  }
  const data = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.batch === 1 && row.length === 65536 && row.backend.startsWith("triton_chunk_c"))
    .map(row => clonePoint(row, row.model === "samu" ? "SAMU dense" : "RG‑LRU‑2", row.chunk_size));
  return { data, title: "B=1、L=65,536 时的分块长度对比", xLabel: "分块长度 C", protocol: "分别运行 C=8、16、32 的实际内核", ratioModel: "rglru" };
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
    _series: "SAMU / RG ratio",
    _x: Number(x),
    speedup: group.other.median_ms / group.samu.median_ms,
    samu: group.samu,
    other: group.other,
    otherLabel: "RG‑LRU‑2"
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
    svg.append(svgEl("text", { x: 480, y: 250, "text-anchor": "middle", fill: "var(--muted)" }, "没有匹配的实测点"));
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
  const metricNames = { median_ms: "延迟中位数", tokens_per_second: "吞吐", speedup: "速度比" };
  $("#chart-caption").textContent = `${data.length} 个实测点；纵轴为${logY ? "对数" : "线性"}刻度，指标是${metricNames[metric]}。速度比定义为人工参数匹配 RG‑LRU‑2 延迟中位数除以 SAMU 延迟中位数；大于 1 表示 SAMU 更快。`;
  renderDetail(data[0], metric);
}

function renderDetail(row, metric) {
  if (row.status === "derived") {
    const tied = Math.abs(row.speedup - 1) < 1e-12;
    const winner = row.speedup >= 1 ? "SAMU" : row.otherLabel;
    $("#benchmark-detail").innerHTML = `<p class="aside-title">两者速度比</p><span class="evidence derived">由延迟计算</span><strong class="big">${metricLabel(row.speedup, "speedup")}</strong><p>${tied ? "在事件分辨率内持平" : `${esc(winner)} 更快`}；计算方法是 RG‑LRU 延迟中位数除以 SAMU 延迟中位数。</p><dl><dt>SAMU</dt><dd>${fmt(row.samu.median_ms, 4)} ms</dd><dt>${esc(row.otherLabel)}</dt><dd>${fmt(row.other.median_ms, 4)} ms</dd></dl>`;
    return;
  }
  const sfu = row.sfu_static_counts || {};
  const raw = row.raw_samples_ms || [];
  $("#benchmark-detail").innerHTML = `<p class="aside-title">选中数据点</p><span class="evidence measured">实测</span><strong class="big">${metricLabel(metricValue(row, metric), metric)}</strong><p>${esc(row._series)}</p><dl>
    <dt>实验轨道</dt><dd>公平 Triton 对照</dd><dt>张量形状</dt><dd>B${row.batch} · L${row.length} · d${row.d_model}</dd>
    <dt>实现路径</dt><dd>${esc(row.backend)}</dd><dt>参数量</dt><dd>${fmt(row.parameter_count, 0)}</dd><dt>每个样本的 FP32 状态</dt><dd>${fmt(row.state_bytes_per_batch, 0)} B</dd>
    <dt>延迟中位数</dt><dd>${fmt(row.median_ms, 4)} ms</dd><dt>P10 / P95</dt><dd>${fmt(row.p10_ms, 4)} / ${fmt(row.p95_ms, 4)}</dd>
    <dt>逻辑流量下界</dt><dd>${fmt(row.logical_bytes_lower_bound / 1048576, 2)} MiB</dd><dt>逻辑有效速率</dt><dd>${fmt(row.logical_effective_gbps, 2)} GB/s*</dd>
    <dt>每 token 的 exp/sigmoid</dt><dd>${fmt(sfu.exp_or_sigmoid_per_token, 0)}</dd><dt>每 token 的平方根</dt><dd>${fmt(sfu.sqrt_per_token, 0)}</dd><dt>预计内核启动数</dt><dd>${fmt(row.expected_cuda_launches, 0)}</dd>
  </dl><p class="code-label">原始 CUDA event 样本（毫秒）</p><p class="mono raw-samples">${raw.map(value => fmt(value, 4)).join(" · ")}</p><a href="${DATA_ROOT}/raw/${encodeURIComponent(row.id)}.json">打开原始 JSON →</a><p class="fairness-inline">* 逻辑字节数除以延迟；它不是硬件 DRAM 计数器测得的带宽。</p>`;
}

function compiledKernelSummary(profile) {
  return (profile.triton_compiled_kernels || []).map(kernel => `${kernel.name.replace(/^_/, "")}: ${kernel.registers_per_thread} 个寄存器/线程 · ${fmt(kernel.derived_occupancy * 100, 1)}% 推导占用率 · ${kernel.spills_per_thread} 个溢出/线程`).join("<br>") || "未测";
}

function renderProfile() {
  const profiles = (profileData.profiles || []).filter(profile => /^(samu|rglru)_/.test(profile.name));
  const equalRows = measured(row => row.track === "A_equal_triton" && row.workload === "prefill" && row.batch === 1 && row.length === 2048 && row.backend === "triton_auto");
  const samu = equalRows.find(row => row.model === "samu");
  const rg = equalRows.find(row => row.model === "rglru");
  return `<div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>操作</th><th>实际 CUDA 启动数</th><th>内核时间合计</th><th>寄存器、占用率与溢出</th></tr></thead><tbody>${profiles.map(profile => `<tr><td>${esc(profile.name)}</td><td>${profile.cuda_kernel_launch_count}</td><td>${fmt(profile.self_cuda_total_us, 0)} μs</td><td class="mono">${compiledKernelSummary(profile)}</td></tr>`).join("")}</tbody></table></div>
    <div class="coverage-grid metric-audit"><div><span class="context-label">特殊函数静态计数（C=32，L=2048）</span><strong>${fmt(samu?.sfu_static_counts?.exp_or_sigmoid_per_token, 0)} 对 ${fmt(rg?.sfu_static_counts?.exp_or_sigmoid_per_token, 0)}</strong><p>左边是 SAMU，右边是 RG‑LRU 的每 token exp/sigmoid 次数；RG‑LRU 另有 ${fmt(rg?.sfu_static_counts?.sqrt_per_token, 0)} 次平方根。它来自源码静态计数，不是特殊函数单元的硬件利用率。</p></div><div><span class="context-label">逻辑有效速率</span><strong>${fmt(samu?.logical_effective_gbps, 2)} 对 ${fmt(rg?.logical_effective_gbps, 2)} GB/s</strong><p>逻辑流量下界除以延迟，用于检查数量级；不等于实测 HBM/DRAM 带宽。</p></div><div><span class="context-label">硬件计数器</span><strong>权限阻止</strong><p>H800 节点已安装 Nsight Compute，但宿主机只允许管理员读取性能计数器；DRAM 字节数、实际带宽和特殊函数单元利用率保持空值。</p></div></div>
    <p class="fairness-inline">“占用率”表示一个 SM 上可同时驻留的活跃 warp 比例。这里根据编译器元数据和 H800 查询到的寄存器、共享内存与线程上限推导，尚未计入分配粒度、调度器和 barrier 限制，因此只用于解释资源上界，不当作硬件实测。</p>`;
}

function renderLab() {
  const panel = $("#lab-panel");
  if (activeTab === "protocol") {
    panel.innerHTML = `<div class="coverage-grid"><div><span class="context-label">RG‑LRU 官方方程</span><strong>Griffin 公式 1–4</strong><p>r=σ(Wₐx+bₐ)，i=σ(Wₓx+bₓ)，a=a_base^(8r)，h=a·h_prev+√(1−a²)·(i·x)。序列重置位置强制 a=0，并把写入归一化系数设为 1。</p></div><div><span class="context-label">人工参数匹配</span><strong>16,772 对 16,768 个参数</strong><p>d=128；RG 使用 2 个门分块来匹配参数量，不是论文默认的 16 分块。递推状态均为每样本 512 字节。</p></div><div><span class="context-label">计时方法</span><strong>反向顺序复测</strong><p>先用 BF16 GEMM 稳定 GPU 时钟；每个点再预热并保留全部 CUDA event 样本。完整套件按 SAMU→RG‑LRU 和 RG‑LRU→SAMU 各运行一次，合并原始样本后重算统计量。</p></div></div><p class="fairness-inline">本轨道只隔离方程与实现成本；主对手的结构结论必须使用页面上方的官方 RG‑LRU‑16 结果。</p>`;
  } else if (activeTab === "profile") {
    panel.innerHTML = renderProfile();
  } else if (activeTab === "correctness") {
    panel.innerHTML = `<div class="coverage-grid"><div><span class="context-label">RG-LRU serial</span><strong>${fmt(audit.rglru_serial_max_abs, 8)}</strong><p>BF16 output max abs；FP32 cache max abs ${fmt(audit.rglru_serial_cache_max_abs, 9)}。</p></div><div><span class="context-label">RG-LRU C16</span><strong>${fmt(audit.rglru_chunk16_max_abs, 7)}</strong><p>与固定 commit 官方实现逐值对照；cache max abs ${fmt(audit.rglru_chunk16_cache_max_abs, 9)}。</p></div><div><span class="context-label">RG-LRU fused decode</span><strong>${fmt(audit.rglru_adversarial_decode_max_abs, 8)}</strong><p>对抗尺度 sweep 的 output max abs；cache max abs ${fmt(audit.rglru_adversarial_decode_cache_max_abs, 9)}。</p></div></div><p class="fairness-inline">SAMU auto-short output max abs ${fmt(audit.samu_auto_short_max_abs, 8)}，cache ${fmt(audit.samu_auto_short_cache_max_abs, 9)}；|d|≤${fmt(audit.samu_phase_delta_bound_rad, 8)} rad 内 sin/cos polynomial max abs 为 ${fmt(audit.samu_sin_polynomial_max_abs, 10)} / ${fmt(audit.samu_cos_polynomial_max_abs, 10)}。</p>`;
  } else if (activeTab === "environment") {
    const fields = { GPU: environment.gpu, "显卡与驱动": environment.driver_and_board, "计算能力": environment.compute_capability, "SM 数量": environment.sm_count, "CUDA 运行时": environment.torch_cuda, PyTorch: environment.torch, Triton: environment.triton, Python: environment.python, "RecurrentGemma 固定版本": environment.source_commits?.recurrentgemma, "测试时间": environment.timestamp_utc };
    panel.innerHTML = `<dl class="environment-grid">${Object.entries(fields).map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl>`;
  } else {
    const measuredCount = rows.filter(row => row.status === "measured").length;
    panel.innerHTML = `<p><b>页面使用的原始依据：</b>${rows.length} 行公平对照，其中 ${measuredCount} 行实测。聚合文件不补数、不平滑、不外推。</p><p><a class="button" href="${DATA_ROOT}/summary.json">打开汇总 JSON</a> <a class="button" href="${DATA_ROOT}/summary.csv">下载汇总 CSV</a> <a class="button" href="${DATA_ROOT}/profiles.json">打开资源数据</a> <a class="button" href="${DATA_ROOT}/run_manifest.json">查看运行清单</a></p>`;
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
    rows = (summary.rows || []).filter(row => row.track === "A_equal_triton" && (row.model === "samu" || row.model === "rglru"));
    environment = summary.environment || {};
    audit = summary.equation_audit || {};
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
    $("#headline-results").innerHTML = `<div class="headline-result"><strong>无法读取实验数据</strong><p>${esc(error.message)}。请通过 HTTP 服务器打开页面。</p></div>`;
    $("#chart-caption").textContent = `读取失败：${error.message}`;
  }
}
