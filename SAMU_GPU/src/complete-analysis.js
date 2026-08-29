const NS = "http://www.w3.org/2000/svg";
const ROOT = "benchmark_results_griffin_complete";

const fmt = (value, digits = 2) => Number(value).toLocaleString(undefined, {
  minimumFractionDigits: 0,
  maximumFractionDigits: digits,
});

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function svgNode(tag, attributes = {}, value = "") {
  const node = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([key, item]) => node.setAttribute(key, item));
  node.textContent = value;
  return node;
}

function lineChart(svg, series, {xLabel, yLabel, yFormat}) {
  const width = Number(svg.viewBox.baseVal.width) || 760;
  const height = Number(svg.viewBox.baseVal.height) || 430;
  const margin = {left: 82, right: 24, top: 28, bottom: 82};
  const points = series.flatMap(item => item.points);
  svg.replaceChildren();
  if (!points.length) {
    svg.append(svgNode("text", {
      x: width / 2, y: height / 2, "text-anchor": "middle",
      fill: "var(--muted)", "font-size": 12,
    }, "结果文件尚未写入"));
    return;
  }
  const xs = [...new Set(points.map(point => point.x))].sort((a, b) => a - b);
  const values = points.map(point => point.y);
  const min = Math.min(...values) * .92;
  const max = Math.max(...values) * 1.06;
  const x = value => margin.left + xs.indexOf(value) *
    (width - margin.left - margin.right) / Math.max(1, xs.length - 1);
  const y = value => height - margin.bottom - (value - min) / Math.max(1e-12, max - min) *
    (height - margin.top - margin.bottom);

  for (let index = 0; index <= 5; index += 1) {
    const yy = margin.top + index * (height - margin.top - margin.bottom) / 5;
    const value = max + (min - max) * index / 5;
    svg.append(svgNode("line", {
      x1: margin.left, y1: yy, x2: width - margin.right, y2: yy,
      stroke: "var(--line)", "stroke-width": 1,
    }));
    svg.append(svgNode("text", {
      x: margin.left - 10, y: yy + 4, "text-anchor": "end",
      fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono",
    }, yFormat(value)));
  }
  xs.forEach(value => svg.append(svgNode("text", {
    x: x(value), y: height - margin.bottom + 25, "text-anchor": "middle",
    fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono",
  }, fmt(value, 0))));

  series.forEach((item, index) => {
    const ordered = [...item.points].sort((a, b) => a.x - b.x);
    const path = ordered.map((point, pointIndex) =>
      `${pointIndex ? "L" : "M"}${x(point.x)},${y(point.y)}`).join(" ");
    svg.append(svgNode("path", {
      d: path, fill: "none", stroke: item.color, "stroke-width": item.dashed ? 1.8 : 2.8,
      "stroke-dasharray": item.dashed ? "6 5" : "none",
    }));
    ordered.forEach(point => svg.append(svgNode("circle", {
      cx: x(point.x), cy: y(point.y), r: 4.5,
      fill: item.color, stroke: "var(--paper)", "stroke-width": 1.5,
    })));
    const legendX = margin.left + (index % 2) * Math.min(300, (width - margin.left - margin.right) / 2);
    const legendY = height - 31 + Math.floor(index / 2) * 17;
    svg.append(svgNode("line", {
      x1: legendX, y1: legendY - 4, x2: legendX + 20, y2: legendY - 4,
      stroke: item.color, "stroke-width": 2.5,
      "stroke-dasharray": item.dashed ? "5 4" : "none",
    }));
    svg.append(svgNode("text", {
      x: legendX + 27, y: legendY, fill: "var(--muted)",
      "font-size": 9, "font-family": "IBM Plex Mono",
    }, item.label));
  });
  svg.append(svgNode("text", {
    x: (margin.left + width - margin.right) / 2, y: height - margin.bottom + 49,
    "text-anchor": "middle", fill: "var(--muted)", "font-size": 10,
    "font-family": "IBM Plex Mono",
  }, xLabel));
  svg.append(svgNode("text", {
    x: 14, y: (margin.top + height - margin.bottom) / 2,
    transform: `rotate(-90 14 ${(margin.top + height - margin.bottom) / 2})`,
    "text-anchor": "middle", fill: "var(--muted)", "font-size": 10,
    "font-family": "IBM Plex Mono",
  }, yLabel));
}

function byMethod(summary, method) {
  return summary.filter(row => row.method === method)
    .map(row => ({x: row.length, y: row.order_balanced_median_ms}));
}

function range(values) {
  return {min: Math.min(...values), max: Math.max(...values)};
}

function renderTraining(data) {
  const summary = data.summary || [];
  const lengths = [2048, 4096, 8192, 16384];
  const get = (method, length) => summary.find(row => row.method === method && row.length === length)
    ?.order_balanced_median_ms;
  const chunkRatios = lengths.map(length => get("rglru_chunk32", length) / get("samu_chunk32", length));
  const linearRatios = lengths.map(length => get("rglru_linear", length) / get("samu_linear", length));
  const chunk = range(chunkRatios);
  const linear = range(linearRatios);
  document.querySelector("#training-result").innerHTML = `<div class="summary-grid">
    <div><h3>chunk‑32 的实测差异</h3><p><strong>${fmt(chunk.min, 2)}–${fmt(chunk.max, 2)} 倍</strong><br>SAMU 更快。收益随长度基本保持，说明不是一次启动延迟造成的偶然差异。</p></div>
    <div><h3>串行路径用于拆分原因</h3><p><strong>${fmt(linear.min, 2)}–${fmt(linear.max, 2)} 倍</strong><br>SAMU 更快。串行只有约 8% 差异，五倍结果来自共享控制与 chunk 并行实现共同作用。</p></div>
    <div><h3>没有形成收益的原型</h3><p>materialized associative reference 明显更慢；compressed transition 也未超过直接 chunk‑32，因此两者都不进入默认分派。</p></div>
  </div>`;
  lineChart(document.querySelector("#training-chart"), [
    {label: "SAMU chunk‑32", color: css("--state"), points: byMethod(summary, "samu_chunk32")},
    {label: "RG‑LRU‑16 chunk‑32", color: css("--rglru"), points: byMethod(summary, "rglru_chunk32")},
    {label: "SAMU serial", color: css("--write"), dashed: true, points: byMethod(summary, "samu_linear")},
    {label: "RG‑LRU‑16 serial", color: css("--motion"), dashed: true, points: byMethod(summary, "rglru_linear")},
  ], {xLabel: "序列长度 L", yLabel: "延迟（ms）", yFormat: value => `${fmt(value, value < 1 ? 2 : 1)} ms`});
  return {chunk, linear};
}

function renderFullForward(data) {
  const rows = data.rows || [];
  const grouped = new Map();
  rows.forEach(row => {
    const key = [row.model, row.sequence_length].join("|");
    const values = grouped.get(key) || [];
    values.push(row.median_ms);
    grouped.set(key, values);
  });
  const points = model => [...grouped.entries()].filter(([key]) => key.startsWith(`${model}|`))
    .map(([key, values]) => ({
      x: Number(key.split("|")[1]),
      y: values.reduce((sum, value) => sum + value, 0) / values.length,
    }));
  const samu = points("samu_grouped16_direct");
  const rg = points("rglru_official16");
  const ratios = samu.map(row => {
    const other = rg.find(candidate => candidate.x === row.x);
    return other.y / row.y;
  });
  const forwardRange = range(ratios);
  lineChart(document.querySelector("#full-forward-chart"), [
    {label: "SAMU‑16 direct · 需重训", color: css("--state"), points: samu},
    {label: "官方 RG‑LRU‑16", color: css("--rglru"), points: rg},
  ], {xLabel: "序列长度 L", yLabel: "完整前向延迟（ms）", yFormat: value => `${fmt(value, value < 100 ? 1 : 0)} ms`});
  document.querySelector("#forward-result").innerHTML = `<div class="summary-grid">
    <div><h3>完整前向仍有收益</h3><p><strong>${fmt(forwardRange.min, 3)}–${fmt(forwardRange.max, 3)} 倍</strong><br>SAMU 更快；B=8、L=2K–16K 的每个已测点方向一致。</p></div>
    <div><h3>五倍 scan 没有变成五倍模型</h3><p>共享 embedding、MLP、归一化和残差加入后，收益缩小到约 3%–5%。这说明下一步应优化共同外壳的融合与参数读取，而不是继续只压低 recurrence 时间。</p></div>
    <div><h3>仍不是训练 step</h3><p>每个 AB/BA 顺序各保留 5 次完整前向样本；没有 backward、优化器、all‑reduce 或 ZeRO，因此不能用这组数字声称训练吞吐领先。</p></div>
  </div>`;
  return forwardRange;
}

function comparison(data, model, batch) {
  return data.comparisons.find(row => row.model === model && row.batch === batch);
}

function modelSeries(data, model) {
  return data.comparisons.filter(row => row.model === model)
    .map(row => ({x: row.batch, y: row.samu_order_balanced_median_ms * 1000}));
}

function rgSeries(data) {
  return data.comparisons.filter(row => row.model === "grouped16_direct")
    .map(row => ({x: row.batch, y: row.rglru_order_balanced_median_ms * 1000}));
}

function speedCells(data, model) {
  return [1, 16, 128].map(batch => `${fmt(comparison(data, model, batch).rglru_over_samu, 3)}×`);
}

function renderAdaptation(data) {
  const directRows = data.comparisons.filter(row => row.model === "grouped16_direct");
  const directRange = range(directRows.map(row => row.rglru_over_samu));
  const counts = data.counts;
  const rg = counts.rglru_official16;
  const direct = counts.grouped16_direct;
  const dense = counts.grouped16_dense;
  document.querySelector("#adaptation-result").innerHTML = `<div class="summary-grid">
    <div><h3>已测批量全部领先</h3><p><strong>${fmt(directRange.min, 3)}–${fmt(directRange.max, 3)} 倍</strong><br>B=1、4、16、64、128 上，SAMU‑16 direct 的 RMSNorm+递推延迟均低于官方 RG‑LRU‑16。</p></div>
    <div><h3>少掉的是投影，不是状态</h3><p><strong>${fmt(direct.projection_macs_per_token / rg.projection_macs_per_token, 2)}× MAC</strong><br>状态仍同为 ${fmt(direct.state_bytes_per_sequence / 1024, 0)} KiB/序列；递推参数 ${fmt(direct.learned_temporal_parameters, 0)} 对 ${fmt(rg.learned_temporal_parameters, 0)}。</p></div>
    <div><h3>资源与计数器</h3><p>SAMU 已测编译变体均无 spill；RG 选中的 2‑warp 路径也无 spill。NCU 计数权限被宿主机阻止，所以 DRAM、L2、SFU 与 achieved occupancy 不填估算值。</p></div>
  </div>`;
  lineChart(document.querySelector("#adaptation-chart"), [
    {label: "SAMU‑16 direct · 需重训", color: css("--state"), points: modelSeries(data, "grouped16_direct")},
    {label: "SAMU‑16 dense · 需重训", color: css("--write"), dashed: true, points: modelSeries(data, "grouped16_dense")},
    {label: "官方 RG‑LRU‑16", color: css("--rglru"), points: rgSeries(data)},
  ], {xLabel: "批量 B", yLabel: "单步延迟（µs）", yFormat: value => `${fmt(value, 1)} µs`});

  const rows = [
    ["grouped‑8 dense", counts.grouped8_dense.learned_temporal_parameters, ...speedCells(data, "grouped8_dense"), "人工参数匹配"],
    ["grouped‑16 dense", dense.learned_temporal_parameters, ...speedCells(data, "grouped16_dense"), "自然 16 组分片，需重训"],
    ["grouped‑16 direct", direct.learned_temporal_parameters, ...speedCells(data, "grouped16_direct"), "GPU 候选，需重训"],
  ];
  document.querySelector("#candidate-table").innerHTML = `<table class="candidate-table"><thead><tr><th>轨道</th><th>递推参数</th><th>B1</th><th>B16</th><th>B128</th><th>用途</th></tr></thead><tbody>${rows.map((row, index) => `<tr><td>${row[0]}</td><td>${fmt(row[1], 0)}</td><td class="${Number.parseFloat(row[2]) >= 1 ? "wins" : ""}">${row[2]}</td><td class="${Number.parseFloat(row[3]) >= 1 ? "wins" : ""}">${row[3]}</td><td class="${Number.parseFloat(row[4]) >= 1 ? "wins" : ""}">${row[4]}</td><td class="${index ? "needs-training" : ""}">${row[5]}</td></tr>`).join("")}</tbody></table>`;
  return directRange;
}

function systemHeadline(data) {
  const grouped = new Map();
  (data.rows || []).forEach(row => {
    const key = [row.model, row.workload, row.batch, row.prompt_length, row.decode_length].join("|");
    const values = grouped.get(key) || [];
    values.push(row.median_ms);
    grouped.set(key, values);
  });
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  const latencyRatios = [];
  for (const prompt of [0, 4096]) for (const length of [128, 256, 512, 1024, 2048, 4096]) {
    const samu = grouped.get(["samu_grouped16_direct", "continuous_decode_latency", 16, prompt, length].join("|"));
    const rg = grouped.get(["rglru_official16", "continuous_decode_latency", 16, prompt, length].join("|"));
    if (samu && rg) latencyRatios.push(mean(rg) / mean(samu));
  }
  const throughputRatios = [];
  for (const length of [512, 1024, 2048, 4096]) {
    const best = model => [...grouped.entries()].filter(([key]) => {
      const [entryModel, workload, , prompt, decoded] = key.split("|");
      return entryModel === model && workload === "bounded_maximum_throughput_candidate" && Number(prompt) === 0 && Number(decoded) === length;
    }).map(([key, values]) => {
      const batch = Number(key.split("|")[2]);
      return batch * length * 1000 / mean(values);
    }).sort((a, b) => b - a)[0];
    const samu = best("samu_grouped16_direct"), rg = best("rglru_official16");
    if (samu && rg) throughputRatios.push(samu / rg);
  }
  return {latency: range(latencyRatios), throughput: range(throughputRatios)};
}

function renderHeadline(training, adaptation, trainingRange, forwardRange, adaptationRange, systemRange) {
  const environment = adaptation.environment || {};
  document.querySelector("#headline-env").textContent = `${environment.gpu || "NVIDIA H800 PCIe"} · SM${String(environment.compute_capability || "90").replace(".", "")} · Triton ${environment.triton || "3.5.0"}`;
  document.querySelector("#headline-results").innerHTML = `
    <div class="headline-result winner-samu"><h3>长序列前向路径</h3><p><b>scan 快 ${fmt(trainingRange.min, 2)}–${fmt(trainingRange.max, 2)} 倍，完整前向快 ${fmt(forwardRange.min, 3)}–${fmt(forwardRange.max, 3)} 倍</b>。两者均为 B=8、L=2K–16K；完整前向仍不含 backward。</p></div>
    <div class="headline-result winner-samu"><h3>D=2048 单步解码候选</h3><p><b>SAMU‑16 direct 快 ${fmt(adaptationRange.min, 3)}–${fmt(adaptationRange.max, 3)} 倍</b>。在 B=1/4/16/64/128 全部已测点领先官方 RG‑LRU‑16。</p></div>
    <div class="headline-result winner-other"><h3>约 1B 参数完整代理</h3><p><b>RG‑LRU‑16 仍快 ${fmt(1 / systemRange.latency.max, 3)}–${fmt(1 / systemRange.latency.min, 3)} 倍</b>；完整轨迹吞吐高 ${fmt((1 / systemRange.throughput.max - 1) * 100, 2)}%–${fmt((1 / systemRange.throughput.min - 1) * 100, 2)}%。微内核收益尚未转化成整模型领先。</p></div>`;
}

export async function initCompleteAnalysis() {
  const [response, systemResponse, forwardResponse] = await Promise.all([
    fetch(`${ROOT}/public_summary.json`),
    fetch(`${ROOT}/griffin_candidate_inference.json`),
    fetch(`${ROOT}/griffin_full_forward_proxy.json`),
  ]);
  if (!response.ok || !systemResponse.ok || !forwardResponse.ok) throw new Error("complete analysis data missing");
  const [{training, adaptation}, system, forward] = await Promise.all([response.json(), systemResponse.json(), forwardResponse.json()]);
  const trainingResult = renderTraining(training);
  const forwardResult = renderFullForward(forward);
  const adaptationResult = renderAdaptation(adaptation);
  renderHeadline(training, adaptation, trainingResult.chunk, forwardResult, adaptationResult, systemHeadline(system));
}
