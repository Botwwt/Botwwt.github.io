import {renderAdvantageFigures} from "./advantage-figures.js?v=20260830-4";

const NS = "http://www.w3.org/2000/svg";
const RESULT_ROOT = "benchmark_results_small_model";
const SCAN_RESULT = "benchmark_results_griffin_complete/canonical_griffin_axes.json";

const fmt = (value, digits = 2) => Number(value).toLocaleString("zh-CN", {maximumFractionDigits: digits});
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const chartObservers = new WeakMap();

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

function drawLineChart(svg, series, {xLabel, yLabel, yFormat = value => fmt(value)}) {
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
  svg.append(svgNode("rect", {x: margin.left, y: margin.top, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom, fill: "none", stroke: "var(--line)", "stroke-width": 1, "data-chart-frame": ""}));

  for (let index = 0; index <= 5; index += 1) {
    const yy = margin.top + index * (height - margin.top - margin.bottom) / 5;
    const value = maximum + (minimum - maximum) * index / 5;
    svg.append(svgNode("line", {x1: margin.left, y1: yy, x2: width - margin.right, y2: yy, stroke: "var(--line)", "stroke-width": 1}));
    svg.append(svgNode("text", {x: margin.left - 10, y: yy + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono"}, yFormat(value)));
  }
  xs.forEach(value => svg.append(svgNode("text", {x: x(value), y: height - margin.bottom + 24, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono"}, fmt(value, 0))));

  series.forEach((item, index) => {
    const ordered = [...item.points].filter(point => Number.isFinite(point.y)).sort((a, b) => a.x - b.x);
    if (!ordered.length) return;
    const path = ordered.map((point, pointIndex) => `${pointIndex ? "L" : "M"}${x(point.x)},${y(point.y)}`).join(" ");
    svg.append(svgNode("path", {d: path, fill: "none", stroke: item.color, "stroke-width": item.dashed ? 2 : 2.8, "stroke-dasharray": item.dashed ? "6 5" : "none"}));
    ordered.forEach(point => svg.append(svgNode("circle", {cx: x(point.x), cy: y(point.y), r: 4.4, fill: item.color, stroke: "var(--paper)", "stroke-width": 1.5})));
    const legendX = margin.left + (index % 2) * Math.min(320, (width - margin.left - margin.right) / 2);
    const legendY = height - 34 + Math.floor(index / 2) * 18;
    svg.append(svgNode("line", {x1: legendX, y1: legendY - 4, x2: legendX + 21, y2: legendY - 4, stroke: item.color, "stroke-width": 2.5, "stroke-dasharray": item.dashed ? "5 4" : "none"}));
    svg.append(svgNode("text", {x: legendX + 28, y: legendY, fill: "var(--muted)", "font-size": 11, "font-family": "IBM Plex Mono"}, item.label));
  });
  svg.append(svgNode("text", {x: (margin.left + width - margin.right) / 2, y: height - margin.bottom + 50, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12, class: "axis-title", "data-axis": "x"}, xLabel));
  svg.append(svgNode("text", {x: 17, y: (margin.top + height - margin.bottom) / 2, transform: `rotate(-90 17 ${(margin.top + height - margin.bottom) / 2})`, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12, class: "axis-title", "data-axis": "y"}, yLabel));
}

function lineChart(svg, series, options) {
  if (!svg) return;
  const originalHeight = Number(svg.viewBox.baseVal.height) || 430;
  const paint = () => {
    const width = Math.max(300, Math.floor(svg.getBoundingClientRect().width || 760));
    svg.setAttribute("viewBox", `0 0 ${width} ${originalHeight}`);
    drawLineChart(svg, series, options);
  };
  paint();
  if (typeof ResizeObserver !== "undefined" && !chartObservers.has(svg)) {
    const observer = new ResizeObserver(() => requestAnimationFrame(paint));
    observer.observe(svg);
    chartObservers.set(svg, observer);
  }
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
    ["SAMU", "PyTorch 逐步参考", "samu_framework_linear_reference"],
    ["SAMU", "自定义顺序扫描", "samu_linear"],
    ["SAMU", "精确 32 步分块", "samu_chunk32"],
    ["SAMU", "BF16 结合扫描", "samu_associative_bf16_reference"],
    ["SAMU", "FP32 结合扫描", "samu_associative_fp32_reference"],
    ["RG-LRU", "PyTorch 逐步参考", "rglru_framework_linear_reference"],
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
  document.querySelector("#device-scan-table").innerHTML = `<h3>固定 B=8、1024 个实状态量的前向扫描</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>递推单元</th><th>扫描方法</th><th>L=2K</th><th>L=4K</th><th>L=8K</th><th>L=16K</th></tr></thead><tbody>${rows}</tbody></table></div><p class="fairness-inline">这一分解实验定位了递推阶段的差异：采用同一种精确分块算法时，SAMU 的扫描本体快 ${range}，因为每个词元只生成两个共享控制量；RG-LRU 则为每个状态通道生成并读取输入门和衰减门。把递推重新装入模型后，矩阵乘法、卷积、前馈网络和 AdamW 会成为共同成本，因此完整训练表同时报告绝对耗时和最终差距。</p>`;
}

function renderScanBackends(data) {
  const labels = {"400m": "400M 配置", "1.3b": "1.3B 配置"};
  const backendLabels = {
    triton: "自定义片上顺序扫描",
    framework_eager: "PyTorch 逐步参考",
    associative_bf16: "BF16 结合扫描",
    associative_fp32: "FP32 结合扫描",
    chunk16: "精确 16 步分块",
    chunk32: "精确 32 步分块",
  };
  const measured = (scale, architecture, backend) => (data.rows || []).find(item => item.scale === scale && item.architecture === architecture && item.backend === backend && item.status === "measured");
  const optimizedBackends = ["triton", "chunk16", "chunk32"];
  const engineeringRows = ["400m", "1.3b"].flatMap(scale => ["samu", "rglru"].map(architecture => {
    const reference = measured(scale, architecture, "framework_eager");
    const candidates = optimizedBackends.map(backend => measured(scale, architecture, backend)).filter(Boolean);
    const fastest = candidates.sort((a, b) => a.median_step_ms - b.median_step_ms)[0];
    const speedup = reference && fastest ? reference.median_step_ms / fastest.median_step_ms : NaN;
    return `<tr><td>${labels[scale]}</td><td>${architecture === "samu" ? "SAMU" : "RG-LRU"}</td><td>${reference ? `${fmt(reference.median_step_ms, 2)} ms` : "—"}</td><td>${fastest ? `${backendLabels[fastest.backend]} · ${fmt(fastest.median_step_ms, 2)} ms` : "—"}</td><td>${Number.isFinite(speedup) ? `${fmt(speedup, 2)} 倍` : "—"}</td></tr>`;
  })).join("");
  const rows = ["400m", "1.3b"].flatMap(scale => ["samu", "rglru"].map(architecture => {
    const cells = Object.keys(backendLabels).map(backend => {
      const row = measured(scale, architecture, backend);
      if (row) return `<td>${fmt(row.median_step_ms, 2)} ms</td>`;
      return `<td>—</td>`;
    }).join("");
    return `<tr><td>${labels[scale]}</td><td>${architecture === "samu" ? "SAMU" : "RG-LRU"}</td>${cells}</tr>`;
  })).join("");
  document.querySelector("#scan-backend-table").innerHTML = `<h3>从 PyTorch 参考到优化 GPU 实现</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>完整模型配置</th><th>递推单元</th><th>PyTorch 逐步参考</th><th>最快精确 GPU 候选</th><th>实现加速</th></tr></thead><tbody>${engineeringRows}</tbody></table></div><p class="fairness-inline">这里的“实现加速”只说明同一递推方程经过 GPU 内核适配后快了多少，不是 SAMU 相对 RG-LRU 的架构加速。</p><h3>完整训练步只更换扫描后端（B=4，L=2K）</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>完整模型配置</th><th>递推单元</th>${Object.values(backendLabels).map(label => `<th>${label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div><p class="fairness-inline">每个单元格都包含 32K 词表、完整 Hawk 层、交叉熵、反向传播、梯度裁剪和 AdamW 更新，唯一变化是递推扫描后端。PyTorch 逐步参考为每个时间步建立运算，其中 RG-LRU 路径保持官方 RecurrentGemma 的扫描语义；它主要用于正确性参考，不代表 H800 性能上限。片上顺序扫描在一个 CUDA 程序内沿时间更新；结合扫描先物化各时间步的仿射转移，再以树形前缀合并；16/32 步分块先并行计算块摘要，再传播块边界并回放块内状态。最终架构对比会用独立调优数据为双方选择最快的精确后端，再以新样本计时。</p>`;
}

function renderTraining(paperScale) {
  const record = (scale, architecture) => paperScale.records?.find(row => row.scale === scale && row.architecture === architecture);
  const measuredRows = (scale, architecture) => (record(scale, architecture)?.rows || []).filter(row => row.status === "measured");
  const speedRange = scale => {
    const samu = measuredRows(scale, "samu");
    const rglru = measuredRows(scale, "rglru");
    const ratios = samu.map(row => {
      const comparison = rglru.find(item => item.sequence_length === row.sequence_length);
      return comparison?.order_balanced_median_step_ms / row.order_balanced_median_step_ms;
    }).filter(Number.isFinite);
    return ratios.length ? [Math.min(...ratios), Math.max(...ratios)] : [NaN, NaN];
  };
  const memoryRange = (scale, architecture) => {
    const values = measuredRows(scale, architecture).map(row => row.peak_allocated_bytes / 2 ** 30).filter(Number.isFinite);
    return values.length ? [Math.min(...values), Math.max(...values)] : [NaN, NaN];
  };
  const speed400 = speedRange("400m"), speed13 = speedRange("1.3b");
  const memory400SAMU = memoryRange("400m", "samu"), memory400RG = memoryRange("400m", "rglru");
  const memory13SAMU = memoryRange("1.3b", "samu"), memory13RG = memoryRange("1.3b", "rglru");
  const describeRange = range => Number.isFinite(range[0])
    ? `SAMU 快 ${fmt(range[0], 2)}–${fmt(range[1], 2)} 倍`
    : "等待结果";
  document.querySelector("#training-result").innerHTML = `<div class="summary-grid two">
    <div><h3>400M 完整优化器步骤</h3><p><strong>${describeRange(speed400)}</strong><br>峰值已分配显存：SAMU ${fmt(memory400SAMU[1], 1)} GiB，RG-LRU ${fmt(memory400RG[0], 1)}–${fmt(memory400RG[1], 1)} GiB。</p></div>
    <div><h3>1.3B 完整优化器步骤</h3><p><strong>${describeRange(speed13)}</strong><br>峰值已分配显存：SAMU ${fmt(memory13SAMU[1], 1)} GiB，RG-LRU ${fmt(memory13RG[0], 1)}–${fmt(memory13RG[1], 1)} GiB。</p></div>
  </div>`;

  const points = (scale, architecture) => measuredRows(scale, architecture).map(row => ({
    x: row.sequence_length,
    y: row.order_balanced_median_step_ms,
  }));
  lineChart(document.querySelector("#training-step-chart"), [
    {label: "400M · SAMU", color: css("--state"), points: points("400m", "samu")},
    {label: "400M · RG-LRU", color: css("--rglru"), points: points("400m", "rglru")},
    {label: "1.3B · SAMU", color: css("--write"), dashed: true, points: points("1.3b", "samu")},
    {label: "1.3B · RG-LRU", color: css("--motion"), dashed: true, points: points("1.3b", "rglru")},
  ], {xLabel: "序列长度 L（每步总词元固定为 8192）", yLabel: "完整优化器步骤耗时（毫秒）", yFormat: value => fmt(value, 0)});
  return {speed400, speed13};
}

function renderInference(data) {
  const rows = data.rows || [];
  const latencyPoints = (architecture, prompt) => rows.filter(row => row.architecture === architecture && row.workload === "continuous_decode_latency" && row.prompt_length === prompt).map(row => ({x: row.decode_length, y: row.median_ms / row.decode_length}));
  const bestThroughput = architecture => rows.filter(row => row.architecture === architecture && row.workload === "maximum_throughput").map(row => ({x: row.decode_length, y: row.tokens_per_second, batch: row.batch_size})).sort((a, b) => a.x - b.x);
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
    if (low >= 1 && high < 1.02) return `SAMU ${noun}高 ${fmt((low - 1) * 100, 1)}%–${fmt((high - 1) * 100, 1)}%，差距小于 2%`;
    if (high < 1 && low > 1 / 1.02) return `RG-LRU ${noun}高 ${fmt((1 / high - 1) * 100, 1)}%–${fmt((1 / low - 1) * 100, 1)}%，差距小于 2%`;
    if (low >= 1) return `SAMU ${noun}高 ${fmt(low, 2)}–${fmt(high, 2)} 倍`;
    if (high < 1) return `RG-LRU ${noun}高 ${fmt(1 / high, 2)}–${fmt(1 / low, 2)} 倍`;
    return "双方随生成长度互有胜负";
  };
  document.querySelector("#inference-result").innerHTML = `<div class="system-facts">
    <div><h3>固定批量连续生成</h3><p><b>${describe(latencyRange, "速度")}</b>。B=16，测量整条生成轨迹。4K 提示只改变固定大小递推状态与卷积缓存中的数值，不改变后续每一步的张量形状，因此空提示与 4K 提示曲线接近重合是预期现象。</p></div>
    <div><h3>候选批量中的最高吞吐</h3><p><b>${describe(throughputRange, "吞吐")}</b>。每个长度都在双方相同的 B=1–512 声明集合内筛选，并以完成整条轨迹的实测值决定。</p></div>
  </div>`;
  lineChart(document.querySelector("#inference-latency-chart"), [
    {label: "SAMU · 空提示", color: css("--state"), points: latencyPoints("samu", 0)},
    {label: "RG-LRU · 空提示", color: css("--rglru"), points: latencyPoints("rglru", 0)},
    {label: "SAMU · 4K 提示", color: css("--state"), dashed: true, points: latencyPoints("samu", 4096)},
    {label: "RG-LRU · 4K 提示", color: css("--rglru"), dashed: true, points: latencyPoints("rglru", 4096)},
  ], {xLabel: "连续生成词元数", yLabel: "平均单步延迟（毫秒/词元）", yFormat: value => fmt(value, 3)});
  lineChart(document.querySelector("#inference-throughput-chart"), [
    {label: "SAMU", color: css("--state"), points: samuBest},
    {label: "RG-LRU", color: css("--rglru"), points: rgBest},
  ], {xLabel: "完整生成词元数", yLabel: "最高吞吐（词元/秒）", yFormat: value => fmt(value, 0)});
  return {latencyRange, throughputRange};
}

function renderResources(inference) {
  const metadata = architecture => inference.measurement_passes?.find(item => item.architecture === architecture);
  const rows = [["SAMU", metadata("samu")], ["RG-LRU-16", metadata("rglru")]].map(([name, row]) => `<tr><td>${name}</td><td>${fmt(row?.parameters, 0)}</td><td>${fmt(row?.bf16_parameter_bytes / 2 ** 30, 2)} GiB</td><td>${fmt(row?.fp32_state_bytes_per_sequence / 1024, 1)} KiB</td><td>${fmt(row?.bf16_convolution_cache_bytes_per_sequence / 1024, 1)} KiB</td><td>${fmt((row?.fp32_state_bytes_per_sequence + row?.bf16_convolution_cache_bytes_per_sequence) / 1024, 1)} KiB</td></tr>`).join("");
  document.querySelector("#resource-table").innerHTML = `<h3>1.3B 配置的权重与每序列缓存</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>方法</th><th>完整模型参数</th><th>BF16 权重</th><th>FP32 递推状态</th><th>BF16 卷积缓存</th><th>缓存合计</th></tr></thead><tbody>${rows}</tbody></table></div><p class="fairness-inline">双方都使用模型宽度 2048、递推宽度 2560、24 层和 32K 词表，并且递推缓存字节完全相同。RG-LRU 保留官方 16 组块对角门，因此其门参数会随通道数展开；SAMU 只增加两个共享控制方向和每模态静态谱参数。速度实验使用随机权重，因为权重数值不改变算子形状；本报告只回答系统性能，不从这些随机权重推导准确率或损失。</p>`;
}

function renderPaperScale(data) {
  const names = {"400m": "400M 尺度", "1.3b": "1.3B 尺度"};
  const records = data.records || [];
  const record = (scale, architecture) => records.find(item => item.scale === scale && item.architecture === architecture);
  const measured = (item, length) => item?.rows?.find(row => row.sequence_length === length && row.status === "measured");
  const rows = [];
  for (const scale of ["400m", "1.3b"]) {
    const samu = record(scale, "samu"), rglru = record(scale, "rglru");
    for (const length of [2048, 4096, 8192]) {
      const s = measured(samu, length), r = measured(rglru, length);
      if (s && r) {
        const ratio = r.order_balanced_median_step_ms / s.order_balanced_median_step_ms;
        const winner = ratio >= 1 ? `SAMU 快 ${fmt(ratio, 2)} 倍` : `RG-LRU 快 ${fmt(1 / ratio, 2)} 倍`;
        rows.push(`<tr><td>${names[scale]}</td><td>${fmt(samu.parameters / 1e6, 1)}M / ${fmt(rglru.parameters / 1e6, 1)}M</td><td>${fmt(length / 1024, 0)}K</td><td>${fmt(s.order_balanced_median_step_ms, 1)} ms</td><td>${fmt(r.order_balanced_median_step_ms, 1)} ms</td><td><strong>${winner}</strong></td><td>${fmt(s.peak_allocated_bytes / 2 ** 30, 1)} / ${fmt(r.peak_allocated_bytes / 2 ** 30, 1)} GiB</td></tr>`);
      } else {
        rows.push(`<tr><td>${names[scale]}</td><td>${fmt(samu?.parameters / 1e6, 1)}M / ${fmt(rglru?.parameters / 1e6, 1)}M</td><td>${fmt(length / 1024, 0)}K</td><td colspan="4">该形状在本次单卡运行中未得到有效计时</td></tr>`);
      }
    }
  }
  document.querySelector("#paper-scale-table").innerHTML = `<h3>按 Griffin 表 2 宽度与层数扩展的完整优化器步骤</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>配置</th><th>SAMU / RG-LRU 参数</th><th>序列长度</th><th>SAMU</th><th>RG-LRU</th><th>对比</th><th>峰值显存：SAMU / RG-LRU</th></tr></thead><tbody>${rows.join("")}</tbody></table></div><p class="fairness-inline">网页显示 400M 与 1.3B 配置的精确参数量。每个点固定 8192 个词元，计入前向、交叉熵、反向、梯度裁剪和 AdamW，并用正序与逆序两轮计时抵消运行顺序。</p>`;
}

function renderRoofline(data) {
  const copy = data.measured_copy_bandwidth;
  const gemm = data.measured_bf16_gemm;
  const roof = data.measured_roofline;
  const recurrence = data.paper_recurrence_arithmetic_intensity;
  const accounting = data.paper_scale_decode_accounting || data.decode_accounting;
  const effective = architecture => (accounting?.[architecture]?.fixed_batch_latency || []).map(row => row.bf16_byte_model_effective_gb_per_second).filter(Number.isFinite);
  const range = values => values.length ? `${fmt(Math.min(...values), 0)}–${fmt(Math.max(...values), 0)} GB/s` : "—";
  const samu = effective("samu"), rglru = effective("rglru");
  document.querySelector("#roofline-table").innerHTML = `<h3>H800 的计算上限、复制带宽与完整解码解释</h3><div class="profile-table-wrap"><table class="profile-table"><thead><tr><th>量</th><th>实测或推导值</th><th>如何得到</th><th>用于回答什么</th></tr></thead><tbody>
    <tr><td>大矩阵 BF16 吞吐</td><td>${fmt(gemm.median_tflops_per_second, 1)} TFLOP/s</td><td>${gemm.m}×${gemm.n}×${gemm.k} 矩阵乘法，中位数</td><td>H800 的计算侧参照</td></tr>
    <tr><td>设备内复制带宽</td><td>${fmt(copy.median_gb_per_second, 0)} GB/s</td><td>读取并写入 ${fmt(copy.tensor_bytes / 2 ** 20, 0)} MiB BF16 张量</td><td>显存流量侧参照</td></tr>
    <tr><td>屋顶线转折点</td><td>${fmt(roof.ridge_flops_per_byte, 1)} FLOP/字节</td><td>计算吞吐除以复制带宽</td><td>低于该值的运算更可能受数据搬运限制</td></tr>
    <tr><td>RG-LRU 单状态更新</td><td>${fmt(recurrence.rglru_flops_per_byte, 2)} FLOP/字节</td><td>Griffin 第 4.2 节：6 次浮点运算、8 字节读写</td><td>远低于转折点，说明递推扫描属于带宽受限阶段</td></tr>
    <tr><td>完整 B=16 解码的字节模型有效带宽</td><td>SAMU ${range(samu)}<br>RG-LRU ${range(rglru)}</td><td>BF16 计算权重与批量相关缓存字节除以每步实测时间</td><td>检查完整模型是否接近权重读取上限</td></tr>
  </tbody></table></div><p class="fairness-inline">复制带宽和矩阵乘法吞吐来自 H800 实测；屋顶线和“权重加缓存”的有效带宽是由这些实测时间与明确的字节模型推导，并非性能计数器读数。租用主机禁止读取 DRAM、L2、特殊函数与占用率计数器，因此这里不以理论数字冒充硬件计数器结果。</p>`;
}

function renderHeadline(environment, training, inference) {
  document.querySelector("#headline-env").textContent = `${environment.gpu || "NVIDIA H800 PCIe"} · Griffin 第 4、5 节实验轴 · 400M/1.3B 系统实测`;
  const describeTraining = (scale, range) => Number.isFinite(range[0])
    ? `${scale} 配置中 SAMU 快 ${fmt(range[0], 2)}–${fmt(range[1], 2)} 倍`
    : `等待 ${scale} 训练步结果`;
  const inferenceText = Number.isFinite(inference.throughputRange[0]) ? (
    inference.throughputRange[0] >= 1 && inference.throughputRange[1] < 1.02
      ? `SAMU 峰值吞吐高 ${fmt((inference.throughputRange[0] - 1) * 100, 1)}%–${fmt((inference.throughputRange[1] - 1) * 100, 1)}%，两者接近持平`
      : inference.throughputRange[0] >= 1
        ? "SAMU 在所有生成长度取得更高峰值吞吐"
        : inference.throughputRange[1] < 1
          ? "RG-LRU 在所有生成长度取得更高峰值吞吐"
          : "双方随生成长度互有胜负"
  ) : "等待 H800 结果";
  document.querySelector("#headline-results").innerHTML = `
    <div class="headline-result winner-samu"><h3>400M 完整训练步</h3><p><b>${describeTraining("400M", training.speed400)}</b>。计入全部 12 层、交叉熵、反向、梯度裁剪和 AdamW 更新。</p></div>
    <div class="headline-result winner-samu"><h3>1.3B 完整训练步</h3><p><b>${describeTraining("1.3B", training.speed13)}</b>。计入全部 24 层、交叉熵、反向、梯度裁剪和 AdamW 更新。</p></div>
    <div class="headline-result"><h3>完整模型生成</h3><p><b>${inferenceText}</b>。每一步包含 24 个完整 Hawk 递推块和 32K 词表投影。</p></div>`;
}

export async function initCompleteAnalysis() {
  const [scan, environment, deviceScan, scanBackends, paperScale, roofline, paperInference] = await Promise.all([
    readJSON(SCAN_RESULT), readJSON(`${RESULT_ROOT}/environment.json`),
    readJSON(`${RESULT_ROOT}/training_on_device_complete.json`),
    readJSON(`${RESULT_ROOT}/paper_scale_backend_ablation.json`),
    readJSON(`${RESULT_ROOT}/paper_scale_h800.json`),
    readJSON(`${RESULT_ROOT}/h800_roofline_decode.json`),
    readJSON(`${RESULT_ROOT}/paper_scale_inference_h800.json`),
  ]);
  renderScan(scan);
  renderDeviceScan(deviceScan);
  renderScanBackends(scanBackends);
  const trainingSummary = renderTraining(paperScale);
  const inferenceSummary = renderInference(paperInference);
  renderResources(paperInference);
  renderPaperScale(paperScale);
  renderRoofline(roofline);
  renderAdvantageFigures({scan, paperScale, inference: paperInference});
  renderHeadline(environment, trainingSummary, inferenceSummary);
}
