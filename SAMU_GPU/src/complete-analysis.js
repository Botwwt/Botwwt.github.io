const NS = "http://www.w3.org/2000/svg";
const RESULT = "benchmark_results_griffin_complete/canonical_griffin_axes.json";

const fmt = (value, digits = 2) => Number(value).toLocaleString("zh-CN", {maximumFractionDigits: digits});
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

function node(tag, attributes = {}, text = "") {
  const element = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  element.textContent = text;
  return element;
}

function lineChart(svg, series, {xLabel, yLabel, yFormat}) {
  const width = Number(svg.viewBox.baseVal.width) || 760;
  const height = Number(svg.viewBox.baseVal.height) || 430;
  const margin = {left: 82, right: 24, top: 30, bottom: 86};
  const points = series.flatMap(item => item.points);
  svg.replaceChildren();
  if (!points.length) {
    svg.append(node("text", {x: width / 2, y: height / 2, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12}, "结果尚未写入"));
    return;
  }
  const xs = [...new Set(points.map(point => point.x))].sort((a, b) => a - b);
  const values = points.map(point => point.y);
  const minimum = Math.min(...values) * .92;
  const maximum = Math.max(...values) * 1.08;
  const x = value => margin.left + xs.indexOf(value) * (width - margin.left - margin.right) / Math.max(1, xs.length - 1);
  const y = value => height - margin.bottom - (value - minimum) / Math.max(1e-12, maximum - minimum) * (height - margin.top - margin.bottom);
  for (let index = 0; index <= 5; index += 1) {
    const yy = margin.top + index * (height - margin.top - margin.bottom) / 5;
    const value = maximum + (minimum - maximum) * index / 5;
    svg.append(node("line", {x1: margin.left, y1: yy, x2: width - margin.right, y2: yy, stroke: "var(--line)", "stroke-width": 1}));
    svg.append(node("text", {x: margin.left - 10, y: yy + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono"}, yFormat(value)));
  }
  xs.forEach(value => svg.append(node("text", {x: x(value), y: height - margin.bottom + 25, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono"}, fmt(value, 0))));
  series.forEach((item, index) => {
    const ordered = [...item.points].sort((a, b) => a.x - b.x);
    const path = ordered.map((point, pointIndex) => `${pointIndex ? "L" : "M"}${x(point.x)},${y(point.y)}`).join(" ");
    svg.append(node("path", {d: path, fill: "none", stroke: item.color, "stroke-width": item.dashed ? 1.9 : 2.8, "stroke-dasharray": item.dashed ? "6 5" : "none"}));
    ordered.forEach(point => svg.append(node("circle", {cx: x(point.x), cy: y(point.y), r: 4.5, fill: item.color, stroke: "var(--paper)", "stroke-width": 1.5})));
    const legendX = margin.left + (index % 2) * Math.min(310, (width - margin.left - margin.right) / 2);
    const legendY = height - 33 + Math.floor(index / 2) * 17;
    svg.append(node("line", {x1: legendX, y1: legendY - 4, x2: legendX + 20, y2: legendY - 4, stroke: item.color, "stroke-width": 2.5, "stroke-dasharray": item.dashed ? "5 4" : "none"}));
    svg.append(node("text", {x: legendX + 27, y: legendY, fill: "var(--muted)", "font-size": 9, "font-family": "IBM Plex Mono"}, item.label));
  });
  svg.append(node("text", {x: (margin.left + width - margin.right) / 2, y: height - margin.bottom + 50, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10}, xLabel));
  svg.append(node("text", {x: 16, y: (margin.top + height - margin.bottom) / 2, transform: `rotate(-90 16 ${(margin.top + height - margin.bottom) / 2})`, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10}, yLabel));
}

function summaryPoints(summary, method) {
  return summary.filter(row => row.method === method).map(row => ({x: row.length, y: row.order_balanced_median_ms}));
}

function renderTraining(data) {
  const summary = data.summary || [];
  const find = (method, length) => summary.find(row => row.method === method && row.length === length)?.order_balanced_median_ms;
  const lengths = [2048, 4096, 8192, 16384];
  const chunkRatios = lengths.map(length => find("rglru_chunk32_official16", length) / find("samu_chunk32_exact", length));
  const serialRatios = lengths.map(length => find("rglru_serial_official16", length) / find("samu_serial_exact", length));
  const range = values => ({min: Math.min(...values), max: Math.max(...values)});
  const chunk = range(chunkRatios), serial = range(serialRatios);
  document.querySelector("#training-result").innerHTML = `<div class="summary-grid">
    <div><h3>分块前向递推</h3><p><strong>${fmt(chunk.min, 2)}–${fmt(chunk.max, 2)} 倍</strong><br>${chunk.min >= 1 ? "标准 SAMU 更快" : chunk.max < 1 ? "官方 RG‑LRU 更快" : "不同长度的胜负不同"}。这张图不包含投影。</p></div>
    <div><h3>顺序前向递推</h3><p><strong>${fmt(serial.min, 2)}–${fmt(serial.max, 2)} 倍</strong><br>用于判断收益来自方程本身，还是来自分块后的并行调度。</p></div>
    <div><h3>不能由此推出</h3><p>没有反向传播、优化器和多卡通信，因此不能把这组结果改写成“完整训练快若干倍”。</p></div>
  </div>`;
  lineChart(document.querySelector("#training-chart"), [
    {label: "SAMU 分块", color: css("--state"), points: summaryPoints(summary, "samu_chunk32_exact")},
    {label: "RG‑LRU‑16 分块", color: css("--rglru"), points: summaryPoints(summary, "rglru_chunk32_official16")},
    {label: "SAMU 顺序", color: css("--write"), dashed: true, points: summaryPoints(summary, "samu_serial_exact")},
    {label: "RG‑LRU‑16 顺序", color: css("--motion"), dashed: true, points: summaryPoints(summary, "rglru_serial_official16")},
  ], {xLabel: "序列长度 L", yLabel: "延迟（毫秒）", yFormat: value => `${fmt(value, value < 1 ? 2 : 1)}`});
  return {chunk, serial};
}

function groupedRows(rows, predicate) {
  const groups = new Map();
  rows.filter(predicate).forEach(row => {
    const key = [row.model, row.workload, row.batch, row.prompt_length, row.decode_length].join("|");
    const values = groups.get(key) || [];
    values.push(row);
    groups.set(key, values);
  });
  return groups;
}

function renderInference(data) {
  const rows = data.rows || [];
  const groups = groupedRows(rows, () => true);
  const metric = (model, workload, batch, prompt, length, field = "median_ms") => {
    const values = groups.get([model, workload, batch, prompt, length].join("|")) || [];
    return values.length ? mean(values.map(row => row[field])) : undefined;
  };
  const latencyPoints = (model, prompt) => [128, 256, 512, 1024, 2048, 4096].map(length => ({x: length, y: metric(model, "continuous_decode_latency", 16, prompt, length)})).filter(point => Number.isFinite(point.y));
  const bestThroughput = (model, length) => {
    const candidates = [...groups.entries()].filter(([key]) => {
      const [entryModel, workload, , prompt, decoded] = key.split("|");
      return entryModel === model && workload === "maximum_throughput_candidate" && Number(prompt) === 0 && Number(decoded) === length;
    }).map(([key, values]) => ({batch: Number(key.split("|")[2]), value: mean(values.map(row => row.tokens_per_second))}));
    return candidates.sort((a, b) => b.value - a.value)[0];
  };
  const lengths = [512, 1024, 2048, 4096];
  const samuBest = lengths.map(length => ({x: length, ...bestThroughput("samu_canonical", length)})).filter(point => point.value);
  const rgBest = lengths.map(length => ({x: length, ...bestThroughput("rglru_official16", length)})).filter(point => point.value);
  const latencyRatios = [0, 4096].flatMap(prompt => [128, 256, 512, 1024, 2048, 4096].map(length => metric("rglru_official16", "continuous_decode_latency", 16, prompt, length) / metric("samu_canonical", "continuous_decode_latency", 16, prompt, length))).filter(Number.isFinite);
  const throughputRatios = lengths.map((length, index) => samuBest[index]?.value / rgBest[index]?.value).filter(Number.isFinite);
  const range = values => ({min: Math.min(...values), max: Math.max(...values)});
  const latency = range(latencyRatios), throughput = range(throughputRatios);

  const latencyWinner = latency.min >= 1 ? `SAMU 快 ${fmt(latency.min, 3)}–${fmt(latency.max, 3)} 倍` : latency.max < 1 ? `RG‑LRU 快 ${fmt(1 / latency.max, 3)}–${fmt(1 / latency.min, 3)} 倍` : "胜负随轨迹变化";
  const throughputWinner = throughput.min >= 1 ? `SAMU 高 ${fmt((throughput.min - 1) * 100, 1)}%–${fmt((throughput.max - 1) * 100, 1)}%` : throughput.max < 1 ? `RG‑LRU 高 ${fmt((1 / throughput.max - 1) * 100, 1)}%–${fmt((1 / throughput.min - 1) * 100, 1)}%` : "胜负随生成长度变化";
  document.querySelector("#inference-result").innerHTML = `<div class="system-facts"><div><h3>连续生成延迟</h3><p><b>${latencyWinner}</b>。包含 canonical SAMU 的稠密写入投影和官方 RG‑LRU 的两组块对角门投影。</p></div><div><h3>最佳完整轨迹吞吐</h3><p><b>${throughputWinner}</b>。每个点都完成整条轨迹，不从短探针直接外推。</p></div></div>`;
  lineChart(document.querySelector("#inference-latency-chart"), [
    {label: "SAMU · 空提示", color: css("--state"), points: latencyPoints("samu_canonical", 0)},
    {label: "RG‑LRU · 空提示", color: css("--rglru"), points: latencyPoints("rglru_official16", 0)},
    {label: "SAMU · 4K 提示", color: css("--write"), dashed: true, points: latencyPoints("samu_canonical", 4096)},
    {label: "RG‑LRU · 4K 提示", color: css("--motion"), dashed: true, points: latencyPoints("rglru_official16", 4096)},
  ], {xLabel: "连续执行步数", yLabel: "延迟（毫秒）", yFormat: value => fmt(value, value < 10 ? 2 : 0)});
  lineChart(document.querySelector("#inference-throughput-chart"), [
    {label: "SAMU", color: css("--state"), points: samuBest.map(point => ({x: point.x, y: point.value}))},
    {label: "RG‑LRU‑16", color: css("--rglru"), points: rgBest.map(point => ({x: point.x, y: point.value}))},
  ], {xLabel: "完整轨迹步数", yLabel: "每秒处理词元数", yFormat: value => fmt(value, 0)});

  const resources = data.resource_accounting;
  const rowsHTML = [
    ["标准 SAMU", resources.samu_canonical, "2"],
    ["官方 RG‑LRU‑16", resources.rglru_official16, "2（本轮所有批量均选择矩阵乘路径）"],
  ].map(([name, item, launches]) => `<tr><td>${name}</td><td>${fmt(item.learned_temporal_parameters, 0)}</td><td>${fmt(item.bf16_parameter_bytes / 1024 / 1024, 2)} MiB</td><td>${fmt(item.fp32_state_bytes_per_sequence / 1024, 1)} KiB</td><td>${launches}</td><td>未测</td></tr>`).join("");
  document.querySelector("#resource-table").innerHTML = `<h3>单层资源对比</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>方法</th><th>递推参数</th><th>BF16 参数字节</th><th>每序列 FP32 状态</th><th>每步 CUDA 启动</th><th>DRAM/L2/实际占用率</th></tr></thead><tbody>${rowsHTML}</tbody></table></div><p class="fairness-inline">参数和状态来自张量形状；启动数来自实际调用路径。硬件性能计数器被宿主机权限阻止，因此保持“未测”。</p>`;
  return {latency, throughput, latencyWinner, throughputWinner};
}

function renderHeadline(data, training, inference) {
  const environment = data.environment || {};
  document.querySelector("#headline-env").textContent = `${environment.gpu || "NVIDIA H800 PCIe"} · SM${String(environment.compute_capability || "9.0").replace(".", "")} · Triton ${environment.triton || ""}`;
  const correctness = data.correctness || {};
  document.querySelector("#headline-results").innerHTML = `
    <div class="headline-result winner-samu"><h3>投影后长序列递推</h3><p><b>分块路径速度比 ${fmt(training.chunk.min, 2)}–${fmt(training.chunk.max, 2)} 倍</b>。只说明前向递推扫描，不包含双方投影和反向传播。</p></div>
    <div class="headline-result ${inference.latency.min >= 1 ? "winner-samu" : "winner-other"}"><h3>D_RNN=2560 单层解码</h3><p><b>${inference.latencyWinner}</b>。这一轨道完整计入双方方程需要的投影，是真正检查稠密写入瓶颈的主结果。</p></div>
    <div class="headline-result"><h3>正确性与缺口</h3><p><b>${correctness.passed ? "两条方程对照通过" : "正确性未通过"}</b>。完整训练、训练质量和完整 1.3B 模型推理仍未测，不能由微基准替代。</p></div>`;
}

export async function initCompleteAnalysis() {
  const response = await fetch(RESULT);
  if (!response.ok) throw new Error("canonical result missing");
  const data = await response.json();
  if (!data.training_scan || !data.inference_recurrent_layer) throw new Error("canonical result incomplete");
  const training = renderTraining(data.training_scan);
  const inference = renderInference(data.inference_recurrent_layer);
  renderHeadline(data, training, inference);
}
