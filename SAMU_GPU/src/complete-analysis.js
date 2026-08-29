const NS = "http://www.w3.org/2000/svg";
const RESULT_ROOT = "benchmark_results_small_model";
const SCAN_RESULT = "benchmark_results_griffin_complete/canonical_griffin_axes.json";

const fmt = (value, digits = 2) => Number(value).toLocaleString("zh-CN", {maximumFractionDigits: digits});
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function svgNode(tag, attributes = {}, text = "") {
  const element = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  element.textContent = text;
  return element;
}

async function readJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`无法读取 ${path}`);
  return response.json();
}

function lineChart(svg, series, {xLabel, yLabel, yFormat = value => fmt(value)}) {
  if (!svg) return;
  const width = Number(svg.viewBox.baseVal.width) || 760;
  const height = Number(svg.viewBox.baseVal.height) || 430;
  const margin = {left: 82, right: 24, top: 32, bottom: 90};
  const points = series.flatMap(item => item.points)
    .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
  svg.replaceChildren();
  if (!points.length) {
    svg.append(svgNode("text", {x: width / 2, y: height / 2, "text-anchor": "middle", fill: "var(--muted)", "font-size": 13}, "结果正在写入"));
    return;
  }

  const xs = [...new Set(points.map(point => point.x))].sort((a, b) => a - b);
  const ys = points.map(point => point.y);
  let minimum = Math.min(...ys), maximum = Math.max(...ys);
  const pad = Math.max((maximum - minimum) * .12, Math.abs(maximum) * .025, 1e-9);
  minimum -= pad;
  maximum += pad;
  const x = value => margin.left + xs.indexOf(value) * (width - margin.left - margin.right) / Math.max(1, xs.length - 1);
  const y = value => height - margin.bottom - (value - minimum) / (maximum - minimum) * (height - margin.top - margin.bottom);

  for (let index = 0; index <= 5; index += 1) {
    const yy = margin.top + index * (height - margin.top - margin.bottom) / 5;
    const value = maximum + (minimum - maximum) * index / 5;
    svg.append(svgNode("line", {x1: margin.left, y1: yy, x2: width - margin.right, y2: yy, stroke: "var(--line)", "stroke-width": 1}));
    svg.append(svgNode("text", {x: margin.left - 10, y: yy + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono"}, yFormat(value)));
  }
  xs.forEach(value => svg.append(svgNode("text", {x: x(value), y: height - margin.bottom + 24, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono"}, fmt(value, 0))));

  series.forEach((item, index) => {
    const ordered = [...item.points].filter(point => Number.isFinite(point.y)).sort((a, b) => a.x - b.x);
    if (!ordered.length) return;
    const path = ordered.map((point, pointIndex) => `${pointIndex ? "L" : "M"}${x(point.x)},${y(point.y)}`).join(" ");
    svg.append(svgNode("path", {d: path, fill: "none", stroke: item.color, "stroke-width": item.dashed ? 2 : 2.8, "stroke-dasharray": item.dashed ? "6 5" : "none"}));
    ordered.forEach(point => svg.append(svgNode("circle", {cx: x(point.x), cy: y(point.y), r: 4.4, fill: item.color, stroke: "var(--paper)", "stroke-width": 1.5})));
    const legendX = margin.left + (index % 2) * Math.min(320, (width - margin.left - margin.right) / 2);
    const legendY = height - 34 + Math.floor(index / 2) * 18;
    svg.append(svgNode("line", {x1: legendX, y1: legendY - 4, x2: legendX + 21, y2: legendY - 4, stroke: item.color, "stroke-width": 2.5, "stroke-dasharray": item.dashed ? "5 4" : "none"}));
    svg.append(svgNode("text", {x: legendX + 28, y: legendY, fill: "var(--muted)", "font-size": 9, "font-family": "IBM Plex Mono"}, item.label));
  });
  svg.append(svgNode("text", {x: (margin.left + width - margin.right) / 2, y: height - margin.bottom + 50, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10}, xLabel));
  svg.append(svgNode("text", {x: 17, y: (margin.top + height - margin.bottom) / 2, transform: `rotate(-90 17 ${(margin.top + height - margin.bottom) / 2})`, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10}, yLabel));
}

function renderScan(data) {
  const summary = data.training_scan?.summary || [];
  const points = method => summary.filter(row => row.method === method).map(row => ({x: row.length, y: row.order_balanced_median_ms}));
  const value = (method, length) => summary.find(row => row.method === method && row.length === length)?.order_balanced_median_ms;
  const lengths = [2048, 4096, 8192, 16384].filter(length => Number.isFinite(value("samu_chunk32_exact", length)));
  const ratios = lengths.map(length => value("rglru_chunk32_official16", length) / value("samu_chunk32_exact", length));
  const range = ratios.length ? `${fmt(Math.min(...ratios), 2)}–${fmt(Math.max(...ratios), 2)} 倍` : "等待结果";
  document.querySelector("#scan-result").innerHTML = `<p><b>只计递推前向时，SAMU 分块扫描相对 RG-LRU 为 ${range}</b>。这项诊断排除了输入、输出投影和优化器，只用于定位扫描内核；完整训练结果见上图与下表。</p>`;
  lineChart(document.querySelector("#training-chart"), [
    {label: "SAMU 分块扫描", color: css("--state"), points: points("samu_chunk32_exact")},
    {label: "RG-LRU-16 分块扫描", color: css("--rglru"), points: points("rglru_chunk32_official16")},
    {label: "SAMU 顺序扫描", color: css("--write"), dashed: true, points: points("samu_serial_exact")},
    {label: "RG-LRU-16 顺序扫描", color: css("--motion"), dashed: true, points: points("rglru_serial_official16")},
  ], {xLabel: "序列长度 L", yLabel: "前向延迟（毫秒）", yFormat: value => fmt(value, value < 1 ? 2 : 1)});
}

function renderDeviceScan(data) {
  const summary = data.summary || [];
  const lengths = [2048, 4096, 8192, 16384];
  const value = (method, length) => summary.find(row => row.method === method && row.length === length)?.order_balanced_median_ms;
  const methods = [
    ["SAMU", "框架逐步循环", "samu_framework_linear_reference"],
    ["SAMU", "自定义顺序扫描", "samu_linear"],
    ["SAMU", "精确 32 步分块", "samu_chunk32"],
    ["SAMU", "BF16 结合扫描", "samu_associative_bf16_reference"],
    ["SAMU", "FP32 结合扫描", "samu_associative_fp32_reference"],
    ["RG-LRU", "框架逐步循环", "rglru_framework_linear_reference"],
    ["RG-LRU", "自定义顺序扫描", "rglru_linear"],
    ["RG-LRU", "精确 32 步分块", "rglru_chunk32"],
    ["RG-LRU", "BF16 结合扫描", "rglru_associative_bf16_reference"],
    ["RG-LRU", "FP32 结合扫描", "rglru_associative_fp32_reference"],
  ];
  const rows = methods.map(([architecture, label, method]) => `<tr><td>${architecture}</td><td>${label}</td>${lengths.map(length => {
    const measured = value(method, length);
    return `<td>${Number.isFinite(measured) ? `${fmt(measured, measured < 1 ? 3 : 2)} ms` : "—"}</td>`;
  }).join("")}</tr>`).join("");
  const ratios = lengths.map(length => value("rglru_chunk32", length) / value("samu_chunk32", length)).filter(Number.isFinite);
  const range = ratios.length ? `${fmt(Math.min(...ratios), 2)}–${fmt(Math.max(...ratios), 2)} 倍` : "等待结果";
  document.querySelector("#device-scan-table").innerHTML = `<h3>固定 B=8、1024 个实状态量的前向扫描</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>递推单元</th><th>扫描方法</th><th>L=2K</th><th>L=4K</th><th>L=8K</th><th>L=16K</th></tr></thead><tbody>${rows}</tbody></table></div><p class="fairness-inline">两种递推都获得同一种精确分块算法后，SAMU 的扫描本体仍快 ${range}。原因是每个词元只生成两个共享控制量；RG-LRU 必须为每个状态通道生成并读取输入门与衰减门。这个比例不等于完整模型加速比，因为它不含矩阵乘法、卷积、前馈网络和优化器。</p>`;
}

function renderScanBackends(data) {
  const labels = {small: "536 万参数", medium: "1543 万参数", large: "4265 万参数"};
  const backendLabels = {
    triton: "自定义片上顺序扫描",
    framework_eager: "框架逐步循环",
    associative_bf16: "BF16 结合扫描",
    associative_fp32: "FP32 结合扫描",
  };
  const rows = ["small", "medium", "large"].flatMap(scale => ["samu", "rglru"].map(architecture => {
    const cells = Object.keys(backendLabels).map(backend => {
      const row = (data.rows || []).find(item => item.scale === scale && item.architecture === architecture && item.backend === backend && item.status === "measured");
      return `<td>${row ? `${fmt(row.median_step_ms, 2)} ms` : "—"}</td>`;
    }).join("");
    return `<tr><td>${labels[scale]}</td><td>${architecture === "samu" ? "SAMU" : "RG-LRU"}</td>${cells}</tr>`;
  })).join("");
  document.querySelector("#scan-backend-table").innerHTML = `<h3>完整训练步只更换扫描后端（B=4，L=2K）</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>完整模型规模</th><th>递推单元</th>${Object.values(backendLabels).map(label => `<th>${label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div><p class="fairness-inline">这些是完整训练步的绝对耗时。自定义扫描把时间循环放在一个或少数 CUDA 程序中，并让状态尽量停留在寄存器或片上存储；框架逐步循环会为每个词元发起多次运算；结合扫描虽然增加时间并行度，却要物化更大的中间张量。这里固定使用“片上顺序扫描”是为了复现 Griffin 的后端消融，不是选择双方各自最快路径；最终最快路径见上方完整训练矩阵。</p>`;
}

const rgRows = (matrix, scale) => matrix.records.find(row => row.scale === scale && row.architecture === "rglru")?.rows || [];
const samuRows = (matrix, scale) => matrix.records.find(row => row.scale === scale && row.architecture === "samu")?.rows || [];

function renderTraining(training, matrix) {
  const {samu, rglru} = training.aggregate;
  const bpcDelta = samu.mean_test_bpc - rglru.mean_test_bpc;
  const measuredShortSAMU = matrix.trained_L256_point?.find(row => row.architecture === "samu");
  const smallSAMU = [
    ...(measuredShortSAMU ? [measuredShortSAMU] : []),
    ...samuRows(matrix, "small"),
  ];
  const smallRG = rgRows(matrix, "small");
  const rgAt = length => smallRG.find(row => row.sequence_length === length)
    || matrix.trained_L256_point?.find(row => row.architecture === "rglru" && row.sequence_length === length);
  const samuAt = length => smallSAMU.find(row => row.sequence_length === length);
  const lengths = smallSAMU.map(row => row.sequence_length).filter(length => rgAt(length));
  const speedups = new Map(lengths.map(length => [length, rgAt(length).median_step_ms / samuAt(length).median_step_ms]));
  const shortSpeed = speedups.get(256), longSpeed = speedups.get(8192);
  const largeSAMU = samuRows(matrix, "large");
  const largeRG = rgRows(matrix, "large");
  const largeSpeedups = largeSAMU.map(row => {
    const comparison = largeRG.find(item => item.sequence_length === row.sequence_length);
    return comparison?.median_step_ms / row.median_step_ms;
  }).filter(Number.isFinite);
  const largeSpeedRange = largeSpeedups.length
    ? [Math.min(...largeSpeedups), Math.max(...largeSpeedups)] : [NaN, NaN];
  const describeSpeed = ratio => {
    if (!Number.isFinite(ratio)) return "等待结果";
    if (Math.abs(ratio - 1) < .02) return "双方差距小于 2%，基本持平";
    return ratio >= 1 ? `SAMU 快 ${fmt(ratio, 2)} 倍` : `RG-LRU 快 ${fmt(1 / ratio, 2)} 倍`;
  };
  const qualityText = `${bpcDelta < 0 ? "SAMU" : "RG-LRU"} 低 ${fmt(Math.abs(bpcDelta), 4)} BPC`;
  document.querySelector("#training-result").innerHTML = `<div class="summary-grid">
    <div><h3>训练质量</h3><p><strong>SAMU ${fmt(samu.mean_test_bpc, 4)} ± ${fmt(samu.std_test_bpc, 4)} BPC</strong><br>RG-LRU ${fmt(rglru.mean_test_bpc, 4)} ± ${fmt(rglru.std_test_bpc, 4)} BPC；${qualityText}。</p></div>
    <div><h3>短序列完整训练步</h3><p><strong>${describeSpeed(shortSpeed)}</strong><br>L=256、B=32，每步总计 8192 个词元。</p></div>
    <div><h3>宽状态完整训练步</h3><p><strong>${Number.isFinite(largeSpeedRange[0]) ? `SAMU 快 ${fmt(largeSpeedRange[0], 2)}–${fmt(largeSpeedRange[1], 2)} 倍` : "等待结果"}</strong><br>4265 万参数、递推宽度 768，覆盖 L=2K、4K、8K。</p></div>
  </div>`;

  const runPoints = architecture => training.runs.filter(run => run.architecture === architecture).map((run, index) => ({
    label: `${architecture === "samu" ? "SAMU" : "RG-LRU"} · 种子 ${run.seed}`,
    color: architecture === "samu" ? css("--state") : css("--rglru"), dashed: index > 0,
    points: run.log.filter(row => row.validation).map(row => ({x: row.step, y: row.validation.bits_per_character})),
  }));
  lineChart(document.querySelector("#training-quality-chart"), [...runPoints("samu"), ...runPoints("rglru")], {
    xLabel: "优化器更新次数", yLabel: "验证集每字符比特数（越低越好）", yFormat: value => fmt(value, 3),
  });
  lineChart(document.querySelector("#training-step-chart"), [
    {label: "SAMU 精确分块训练", color: css("--state"), points: smallSAMU.map(row => ({x: row.sequence_length, y: row.tokens_per_second}))},
    {label: "RG-LRU-16", color: css("--rglru"), points: [...smallRG, ...(matrix.trained_L256_point || []).filter(row => row.architecture === "rglru")].map(row => ({x: row.sequence_length, y: row.tokens_per_second}))},
  ], {xLabel: "序列长度 L（每步总词元固定为 8192）", yLabel: "完整训练吞吐（词元/秒）", yFormat: value => fmt(value, 0)});

  const labels = {small: "536 万参数", medium: "1543 万参数", large: "4265 万参数"};
  const rows = ["small", "medium", "large"].flatMap(scale => {
    const measuredSAMU = samuRows(matrix, scale);
    const comparison = rgRows(matrix, scale);
    return measuredSAMU.map(row => {
      const rg = comparison.find(item => item.sequence_length === row.sequence_length);
      const ratio = rg?.median_step_ms / row.median_step_ms;
      const winner = ratio >= 1
        ? `SAMU 快 ${fmt(ratio, 2)} 倍`
        : `RG-LRU 快 ${fmt(1 / ratio, 2)} 倍`;
      return rg ? `<tr><td>${labels[scale]}</td><td>${fmt(row.sequence_length / 1024, 0)}K</td><td>${fmt(row.median_step_ms, 2)} ms</td><td>${fmt(rg.median_step_ms, 2)} ms</td><td><strong>${winner}</strong></td><td>${fmt(row.peak_allocated_bytes / 2 ** 30, 2)} / ${fmt(rg.peak_allocated_bytes / 2 ** 30, 2)} GiB</td></tr>` : "";
    });
  }).join("");
  const table = document.querySelector("#training-matrix-table");
  if (table) table.innerHTML = `<div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>完整模型规模</th><th>序列长度</th><th>SAMU 每步耗时</th><th>RG-LRU 每步耗时</th><th>更快的一方</th><th>峰值显存：SAMU / RG-LRU</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  return {samu, rglru, bpcDelta, shortSpeed, longSpeed, largeSpeedRange};
}

function renderInference(data) {
  const latencyPoints = (architecture, prompt) => (data[architecture]?.decode?.latency || []).filter(row => row.prompt_tokens === prompt).map(row => ({x: row.generated_tokens, y: row.median_trajectory_ms}));
  const bestThroughput = architecture => {
    const groups = new Map();
    (data[architecture]?.decode?.throughput || []).forEach(row => {
      const previous = groups.get(row.generated_tokens);
      if (!previous || row.median_tokens_per_second > previous.y) groups.set(row.generated_tokens, {x: row.generated_tokens, y: row.median_tokens_per_second, batch: row.batch_size});
    });
    return [...groups.values()].sort((a, b) => a.x - b.x);
  };
  const samuBest = bestThroughput("samu"), rgBest = bestThroughput("rglru");
  const map = rows => new Map(rows.map(row => [row.x, row.y]));
  const rgMap = map(rgBest);
  const throughputRatios = samuBest.filter(row => rgMap.has(row.x)).map(row => row.y / rgMap.get(row.x));
  const samuLatency = latencyPoints("samu", 0), rgLatencyMap = map(latencyPoints("rglru", 0));
  const latencyRatios = samuLatency.filter(row => rgLatencyMap.has(row.x)).map(row => rgLatencyMap.get(row.x) / row.y);
  const range = values => values.length ? [Math.min(...values), Math.max(...values)] : [NaN, NaN];
  const throughputRange = range(throughputRatios), latencyRange = range(latencyRatios);
  const describe = ([low, high], noun) => {
    if (!Number.isFinite(low)) return "等待结果";
    if (low >= 1) return `SAMU ${noun}高 ${fmt(low, 2)}–${fmt(high, 2)} 倍`;
    if (high < 1) return `RG-LRU ${noun}高 ${fmt(1 / high, 2)}–${fmt(1 / low, 2)} 倍`;
    return "双方随生成长度互有胜负";
  };
  document.querySelector("#inference-result").innerHTML = `<div class="system-facts">
    <div><h3>固定批量连续生成</h3><p><b>${describe(latencyRange, "速度")}</b>。B=16，测量整条生成轨迹；4K 提示只改变起始缓存，不计入生成延迟。</p></div>
    <div><h3>候选批量中的最高吞吐</h3><p><b>${describe(throughputRange, "吞吐")}</b>。每个长度都在双方相同的 B=1–256 候选集合内选取最高值。</p></div>
  </div>`;
  lineChart(document.querySelector("#inference-latency-chart"), [
    {label: "SAMU · 空提示", color: css("--state"), points: latencyPoints("samu", 0)},
    {label: "RG-LRU · 空提示", color: css("--rglru"), points: latencyPoints("rglru", 0)},
    {label: "SAMU · 4K 提示", color: css("--write"), dashed: true, points: latencyPoints("samu", 4096)},
    {label: "RG-LRU · 4K 提示", color: css("--motion"), dashed: true, points: latencyPoints("rglru", 4096)},
  ], {xLabel: "连续生成词元数", yLabel: "完整轨迹延迟（毫秒）", yFormat: value => fmt(value, value < 10 ? 2 : 0)});
  lineChart(document.querySelector("#inference-throughput-chart"), [
    {label: "SAMU", color: css("--state"), points: samuBest},
    {label: "RG-LRU", color: css("--rglru"), points: rgBest},
  ], {xLabel: "完整生成词元数", yLabel: "最高吞吐（词元/秒）", yFormat: value => fmt(value, 0)});
  return {latencyRange, throughputRange};
}

function renderResources(training) {
  const rows = [["SAMU", training.aggregate.samu], ["RG-LRU-16", training.aggregate.rglru]].map(([name, row]) => `<tr><td>${name}</td><td>${fmt(row.parameters, 0)}</td><td>${fmt(row.recurrent_mixer_parameters, 0)}</td><td>${fmt(row.parameters * 2 / 2 ** 20, 2)} MiB</td><td>22.5 KiB</td></tr>`).join("");
  document.querySelector("#resource-table").innerHTML = `<h3>完整六层小模型的资源规模</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>方法</th><th>完整模型参数</th><th>六个递推混合器参数</th><th>BF16 权重字节</th><th>每序列 FP32 状态与卷积缓存</th></tr></thead><tbody>${rows}</tbody></table></div><p class="fairness-inline">词嵌入、归一化、输入与输出线性层、卷积和前馈网络保持相同。SAMU 复用递推块已有的输入分支作为复数写入，所以完整模型参数为 536 万，而不是旧微基准中错误边界下的 656 万。</p>`;
}

function renderHeadline(environment, training, inference) {
  document.querySelector("#headline-env").textContent = `${environment.gpu || "NVIDIA H800 PCIe"} · 完整六层字符模型 · 3 个随机种子`;
  const quality = `${training.bpcDelta < 0 ? "SAMU" : "RG-LRU"} 测试集 BPC 低 ${fmt(Math.abs(training.bpcDelta), 4)}`;
  const trainText = Number.isFinite(training.largeSpeedRange[0])
    ? `宽状态模型中 SAMU 快 ${fmt(training.largeSpeedRange[0], 2)}–${fmt(training.largeSpeedRange[1], 2)} 倍`
    : "等待训练结果";
  const inferenceText = Number.isFinite(inference.throughputRange[0]) ? (inference.throughputRange[0] >= 1 ? "SAMU 在所有生成长度取得更高峰值吞吐" : inference.throughputRange[1] < 1 ? "RG-LRU 在所有生成长度取得更高峰值吞吐" : "双方随生成长度互有胜负") : "等待 H800 结果";
  document.querySelector("#headline-results").innerHTML = `
    <div class="headline-result"><h3>训练质量</h3><p><b>${quality}</b>。报告三个随机种子的均值和标准差。</p></div>
    <div class="headline-result winner-samu"><h3>完整训练步</h3><p><b>${trainText}</b>。计入前向、反向、梯度裁剪和 AdamW 更新。</p></div>
    <div class="headline-result"><h3>完整模型生成</h3><p><b>${inferenceText}</b>。每一步包含六个递推块和词表投影。</p></div>`;
}

export async function initCompleteAnalysis() {
  const [scan, training, model, environment, matrix, deviceScan, scanBackends] = await Promise.all([
    readJSON(SCAN_RESULT), readJSON(`${RESULT_ROOT}/training_summary.json`),
    readJSON(`${RESULT_ROOT}/full_model_benchmark.json`), readJSON(`${RESULT_ROOT}/environment.json`),
    readJSON(`${RESULT_ROOT}/griffin_training_matrix_final.json`),
    readJSON(`${RESULT_ROOT}/training_on_device_complete.json`),
    readJSON(`${RESULT_ROOT}/griffin_scan_backend_matrix.json`),
  ]);
  renderScan(scan);
  renderDeviceScan(deviceScan);
  renderScanBackends(scanBackends);
  const trainingSummary = renderTraining(training, matrix);
  const inferenceSummary = renderInference(model);
  renderResources(training);
  renderHeadline(environment, trainingSummary, inferenceSummary);
}
