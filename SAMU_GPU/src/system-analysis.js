const NS = "http://www.w3.org/2000/svg";
const RESULT = "benchmark_results_griffin_complete/griffin_candidate_inference.json";

const fmt = (value, digits = 2) => Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const svgNode = (tag, attrs = {}, text = "") => {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  return node;
};

function aggregate(rows) {
  const groups = new Map();
  rows.forEach(row => {
    const key = [row.model, row.workload, row.batch, row.prompt_length, row.decode_length].join("|");
    const group = groups.get(key) || {...row, samples: []};
    group.samples.push(row.median_ms);
    groups.set(key, group);
  });
  return [...groups.values()].map(row => {
    const medianMs = row.samples.reduce((sum, value) => sum + value, 0) / row.samples.length;
    return {
      ...row,
      median_ms: medianMs,
      tokens_per_second: row.batch * row.decode_length * 1000 / medianMs,
      order_samples: row.samples.length,
    };
  });
}

function draw(svg, series, {yFormat, xLabel}) {
  const W = 760, H = 430, margin = {l: 82, r: 24, t: 25, b: 78};
  const all = series.flatMap(item => item.points);
  svg.replaceChildren();
  if (!all.length) {
    svg.append(svgNode("text", {x: W / 2, y: H / 2, "text-anchor": "middle", fill: "var(--muted)"}, "完整轨迹结果仍在运行"));
    return;
  }
  const xs = [...new Set(all.map(point => point.x))].sort((a, b) => a - b);
  const values = all.map(point => point.y);
  const min = Math.min(...values) * .94, max = Math.max(...values) * 1.06;
  const x = value => margin.l + xs.indexOf(value) * (W - margin.l - margin.r) / Math.max(1, xs.length - 1);
  const y = value => H - margin.b - (value - min) / Math.max(1e-12, max - min) * (H - margin.t - margin.b);
  for (let tick = 0; tick <= 4; tick += 1) {
    const yy = margin.t + tick * (H - margin.t - margin.b) / 4;
    const value = max + (min - max) * tick / 4;
    svg.append(svgNode("line", {x1: margin.l, y1: yy, x2: W - margin.r, y2: yy, stroke: "var(--line)"}));
    svg.append(svgNode("text", {x: margin.l - 10, y: yy + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono"}, yFormat(value)));
  }
  xs.forEach(value => svg.append(svgNode("text", {x: x(value), y: H - margin.b + 25, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono"}, fmt(value, 0))));
  series.forEach((item, index) => {
    const points = [...item.points].sort((a, b) => a.x - b.x);
    svg.append(svgNode("path", {
      d: points.map((point, i) => `${i ? "L" : "M"}${x(point.x)},${y(point.y)}`).join(" "),
      fill: "none", stroke: item.color, "stroke-width": item.dashed ? 1.8 : 2.8,
      "stroke-dasharray": item.dashed ? "6 5" : "none",
    }));
    points.forEach(point => {
      svg.append(svgNode("circle", {cx: x(point.x), cy: y(point.y), r: 4.5, fill: item.color, stroke: "var(--paper)", "stroke-width": 1.5}));
      if (point.label) svg.append(svgNode("text", {
        x: x(point.x) + 6, y: y(point.y) - 7, fill: item.color,
        "font-size": 8, "font-family": "IBM Plex Mono",
      }, point.label));
    });
    const legendX = margin.l + (index % 2) * 245;
    const legendY = H - 29 + Math.floor(index / 2) * 17;
    svg.append(svgNode("line", {x1: legendX, y1: legendY - 4, x2: legendX + 20, y2: legendY - 4, stroke: item.color, "stroke-width": 2.5, "stroke-dasharray": item.dashed ? "5 4" : "none"}));
    svg.append(svgNode("text", {x: legendX + 27, y: legendY, fill: "var(--muted)", "font-size": 9, "font-family": "IBM Plex Mono"}, item.label));
  });
  svg.append(svgNode("text", {x: (margin.l + W - margin.r) / 2, y: H - margin.b + 48, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono"}, xLabel));
}

function bestThroughput(rows, model) {
  const candidates = rows.filter(row => row.workload === "bounded_maximum_throughput_candidate" && row.model === model);
  const lengths = [...new Set(candidates.map(row => row.decode_length))];
  return lengths.map(length => candidates.filter(row => row.decode_length === length)
    .sort((a, b) => b.tokens_per_second - a.tokens_per_second)[0]);
}

function ratioRange(left, right, field, higherIsBetter = false) {
  const ratios = left.map(row => {
    const other = right.find(candidate => candidate.decode_length === row.decode_length && candidate.prompt_length === row.prompt_length);
    return other ? (higherIsBetter ? row[field] / other[field] : other[field] / row[field]) : null;
  }).filter(Number.isFinite);
  return ratios.length ? {min: Math.min(...ratios), max: Math.max(...ratios)} : null;
}

function comparisonText(value, subject) {
  if (!value) return "结果不足。";
  if (value.min >= 1) return `SAMU ${subject}为 RG‑LRU 的 ${fmt(value.min, 3)}–${fmt(value.max, 3)} 倍。`;
  if (value.max <= 1) return `RG‑LRU ${subject}领先 ${fmt(1 / value.max, 3)}–${fmt(1 / value.min, 3)} 倍。`;
  return `${subject}速度比跨过 1，范围为 ${fmt(value.min, 3)}–${fmt(value.max, 3)} 倍。`;
}

function render(data) {
  const rows = aggregate(data.rows || []);
  const state = css("--state"), rgColor = css("--rglru");
  const latency = rows.filter(row => row.workload === "continuous_decode_latency");
  const series = [];
  [["samu_grouped16_direct", 0, "SAMU‑16 direct · 空前缀", state, false], ["rglru_official16", 0, "RG‑LRU‑16 · 空前缀", rgColor, false], ["samu_grouped16_direct", 4096, "SAMU‑16 direct · 4K 前缀", state, true], ["rglru_official16", 4096, "RG‑LRU‑16 · 4K 前缀", rgColor, true]].forEach(([model, prompt, label, color, dashed]) => {
    series.push({label, color, dashed, points: latency.filter(row => row.model === model && row.prompt_length === prompt).map(row => ({x: row.decode_length, y: row.median_ms / 1000}))});
  });
  draw(document.querySelector("#section5-latency-chart"), series, {yFormat: value => `${fmt(value, value < 10 ? 2 : 1)} s`, xLabel: "连续生成 token 数"});

  const bestSamu = bestThroughput(rows, "samu_grouped16_direct");
  const bestRg = bestThroughput(rows, "rglru_official16");
  draw(document.querySelector("#section5-throughput-chart"), [
    {label: "SAMU‑16 direct · 需重训", color: state, points: bestSamu.map(row => ({x: row.decode_length, y: row.tokens_per_second, label: `B${row.batch}`}))},
    {label: "官方 RG‑LRU‑16", color: rgColor, points: bestRg.map(row => ({x: row.decode_length, y: row.tokens_per_second, label: `B${row.batch}`}))},
  ], {yFormat: value => `${fmt(value, 0)} tok/s`, xLabel: "每个序列生成 token 数"});

  const allSamu = latency.filter(row => row.model === "samu_grouped16_direct");
  const allRg = latency.filter(row => row.model === "rglru_official16");
  const latencyRatio = ratioRange(allSamu, allRg, "median_ms");
  const throughputRatio = ratioRange(bestSamu, bestRg, "tokens_per_second", true);
  const at4096 = model => latency.find(row => row.model === model && row.prompt_length === 0 && row.decode_length === 4096);
  const at4096Prompt = model => latency.find(row => row.model === model && row.prompt_length === 4096 && row.decode_length === 4096);
  const samuEnd = at4096("samu_grouped16_direct"), rgEnd = at4096("rglru_official16");
  const samuPrompt = at4096Prompt("samu_grouped16_direct"), rgPrompt = at4096Prompt("rglru_official16");
  const modelSamu = data.models?.samu_grouped16_direct || {};
  const modelRg = data.models?.rglru_official16 || {};
  const audit = data.correctness || {};
  const latencyRepeats = data.paper_protocol?.latency_repeats_per_order || 1;
  const throughputRepeats = data.paper_protocol?.throughput_repeats_per_order || 1;
  const topSamu = [...bestSamu].sort((a, b) => b.tokens_per_second - a.tokens_per_second)[0];
  const topRg = [...bestRg].sort((a, b) => b.tokens_per_second - a.tokens_per_second)[0];
  document.querySelector("#system-result").innerHTML = `<div class="system-facts">
    <div><h3>官方对手与正确性</h3><p>RG‑LRU 来自固定 RecurrentGemma commit ${String(data.environment?.rglru_commit || "").slice(0, 12)}，使用 16 个门分块。官方输出误差 ${fmt(audit.rglru_fused_vs_official_output_max_abs || 0, 8)}，cache 最大误差 ${fmt(audit.rglru_fused_vs_official_cache_max_abs || 0, 8)}；审计 ${audit.passed ? "通过" : "未通过"}。</p></div>
    <div><h3>同一外壳，只换递推层</h3><p>两边直接引用同一批 embedding、24 层 MLP 和归一化张量，状态也均为 ${fmt((modelSamu.state_bytes_per_sequence || 0) / 1024, 0)} KiB/序列。SAMU 代理 ${fmt(modelSamu.parameter_count || 0, 0)} 参数，RG‑LRU 代理 ${fmt(modelRg.parameter_count || 0, 0)}；参数差异来自候选递推层，不伪装成参数匹配。</p></div>
    <div><h3>B=16 连续解码</h3><p>${comparisonText(latencyRatio, "生成速度")}${samuEnd && samuPrompt ? `4K 前缀相对空前缀的 4096-step 延迟比为 SAMU ${fmt(samuPrompt.median_ms / samuEnd.median_ms, 3)}、RG‑LRU ${fmt(rgPrompt.median_ms / rgEnd.median_ms, 3)}。` : ""}</p></div>
    <div><h3>完整轨迹吞吐</h3><p>${comparisonText(throughputRatio, "完整轨迹吞吐")}${topSamu && topRg ? `已测最高点：SAMU ${fmt(topSamu.tokens_per_second, 0)} tok/s（B=${topSamu.batch}），RG‑LRU ${fmt(topRg.tokens_per_second, 0)} tok/s（B=${topRg.batch}）。` : ""}</p></div>
  </div><p class="fairness-inline">结论很直接：SAMU 专属解码微内核在测试形状上更快，但共同 MLP、词表投影和 24 层完整调度加入后，当前 1B 代理仍由 RG‑LRU 领先。延迟轨迹在 AB 与 BA 每个顺序下各重复 ${latencyRepeats} 次，吞吐轨迹各重复 ${throughputRepeats} 次；每个顺序内先取中位数，再对两个运行顺序等权合并。完整原始样本、32-step 批量探针和模型元数据见 <a href="${RESULT}">原始 JSON</a>。这仍是随机权重系统代理；SAMU direct 必须重训后才能报告模型质量。</p>`;
}

function pending() {
  document.querySelector("#system-result").innerHTML = `<p><b>1B 代理完整轨迹正在 H800 后台运行。</b> 文件写入后，本节会显示 AB/BA 双顺序合并的连续解码与完整轨迹吞吐；训练扫描和单步解码微内核结果不受此任务影响。</p>`;
  draw(document.querySelector("#section5-latency-chart"), [], {yFormat: String, xLabel: ""});
  draw(document.querySelector("#section5-throughput-chart"), [], {yFormat: String, xLabel: ""});
}

export async function initSystemAnalysis() {
  try {
    const response = await fetch(RESULT);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    pending();
  }
}
