const ns = "http://www.w3.org/2000/svg";
const resultRoot = "benchmark_results_griffin_section5";

const svgNode = (tag, attrs = {}, text = "") => {
  const node = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  return node;
};
const fmt = (value, digits = 2) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const color = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function paired(rows, prompt, length) {
  const find = model => rows.find(row => row.workload === "continuous_decode_latency"
    && row.model === model && row.prompt_length === prompt && row.decode_length === length);
  return { samu: find("samu"), rglru: find("rglru") };
}

function ratioRange(rows, workload, field, lengths, prompt = null, higherIsBetter = false) {
  const ratios = lengths.map(length => {
    const candidates = rows.filter(row => row.workload === workload
      && row.decode_length === length && (prompt === null || row.prompt_length === prompt));
    const samu = candidates.find(row => row.model === "samu");
    const rglru = candidates.find(row => row.model === "rglru");
    return samu && rglru
      ? (higherIsBetter ? samu[field] / rglru[field] : rglru[field] / samu[field])
      : null;
  }).filter(value => Number.isFinite(value));
  return ratios.length ? { min: Math.min(...ratios), max: Math.max(...ratios) } : null;
}

function drawLines(svg, series, { yFormat, xLabel }) {
  svg.replaceChildren();
  const W = 760, H = 430, margin = { l: 82, r: 24, t: 24, b: 78 };
  const all = series.flatMap(item => item.points);
  if (!all.length) {
    svg.append(svgNode("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "var(--muted)" }, "完整实验仍在运行"));
    return;
  }
  const xs = [...new Set(all.map(point => point.x))].sort((a, b) => a - b);
  const values = all.map(point => point.y);
  const min = Math.min(...values), max = Math.max(...values);
  const logY = max / Math.max(min, 1e-12) > 8;
  const x = value => margin.l + xs.indexOf(value) * (W - margin.l - margin.r) / Math.max(1, xs.length - 1);
  const scaled = value => logY
    ? (Math.log(value) - Math.log(min)) / (Math.log(max) - Math.log(min) || 1)
    : (value - min) / (max - min || 1);
  const y = value => H - margin.b - scaled(value) * (H - margin.t - margin.b);
  for (let tick = 0; tick <= 4; tick++) {
    const yy = margin.t + tick * (H - margin.t - margin.b) / 4;
    const value = logY
      ? Math.exp(Math.log(max) + (Math.log(min) - Math.log(max)) * tick / 4)
      : max + (min - max) * tick / 4;
    svg.append(svgNode("line", { x1: margin.l, y1: yy, x2: W - margin.r, y2: yy, stroke: "var(--line)" }));
    svg.append(svgNode("text", { x: margin.l - 10, y: yy + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono" }, yFormat(value)));
  }
  xs.forEach(value => svg.append(svgNode("text", { x: x(value), y: H - margin.b + 25, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono" }, fmt(value, 0))));
  series.forEach((item, index) => {
    const points = item.points.sort((a, b) => a.x - b.x);
    const path = points.map((point, i) => `${i ? "L" : "M"}${x(point.x)},${y(point.y)}`).join(" ");
    svg.append(svgNode("path", { d: path, fill: "none", stroke: item.color, "stroke-width": item.dashed ? 1.8 : 2.8, "stroke-dasharray": item.dashed ? "6 5" : "none" }));
    points.forEach(point => svg.append(svgNode("circle", { cx: x(point.x), cy: y(point.y), r: 4.5, fill: item.color, stroke: "var(--paper)", "stroke-width": 1.5 })));
    const legendX = margin.l + (index % 2) * 245;
    const legendY = H - 28 + Math.floor(index / 2) * 17;
    svg.append(svgNode("line", { x1: legendX, y1: legendY - 4, x2: legendX + 20, y2: legendY - 4, stroke: item.color, "stroke-width": 2.5, "stroke-dasharray": item.dashed ? "5 4" : "none" }));
    svg.append(svgNode("text", { x: legendX + 27, y: legendY, fill: "var(--muted)", "font-size": 9, "font-family": "IBM Plex Mono" }, item.label));
  });
  svg.append(svgNode("text", { x: (margin.l + W - margin.r) / 2, y: H - margin.b + 48, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10, "font-family": "IBM Plex Mono" }, xLabel));
}

function render(data) {
  const rows = data.rows || [];
  const state = color("--state"), rg = color("--rglru");
  const latencyRows = rows.filter(row => row.workload === "continuous_decode_latency");
  const latencySeries = [];
  [["samu", 0, "SAMU · 空前缀", state, false], ["rglru", 0, "RG‑LRU · 空前缀", rg, false], ["samu", 4096, "SAMU · 4K 前缀", state, true], ["rglru", 4096, "RG‑LRU · 4K 前缀", rg, true]].forEach(([model, prompt, label, seriesColor, dashed]) => {
    latencySeries.push({ label, color: seriesColor, dashed, points: latencyRows.filter(row => row.model === model && row.prompt_length === prompt).map(row => ({ x: row.decode_length, y: row.median_ms / 1000 })) });
  });
  drawLines(document.querySelector("#section5-latency-chart"), latencySeries, { yFormat: value => `${fmt(value, value < 10 ? 2 : 1)} s`, xLabel: "连续生成 token 数" });

  const throughput = rows.filter(row => row.workload === "maximum_throughput");
  drawLines(document.querySelector("#section5-throughput-chart"), [
    { label: "SAMU", color: state, points: throughput.filter(row => row.model === "samu").map(row => ({ x: row.decode_length, y: row.tokens_per_second })) },
    { label: "RG‑LRU", color: rg, points: throughput.filter(row => row.model === "rglru").map(row => ({ x: row.decode_length, y: row.tokens_per_second })) },
  ], { yFormat: value => `${fmt(value, 0)} tok/s`, xLabel: "每个序列生成 token 数" });

  const modelSamu = data.models?.samu, modelRg = data.models?.rglru;
  const end = paired(rows, 0, 4096);
  const end4k = paired(rows, 4096, 4096);
  const speed = end.samu && end.rglru ? end.rglru.median_ms / end.samu.median_ms : null;
  const latencyRange = ratioRange(rows, "continuous_decode_latency", "median_ms", [128, 256, 512, 1024, 2048, 4096], 0);
  const throughputRange = ratioRange(rows, "maximum_throughput", "tokens_per_second", [512, 1024, 2048, 4096], null, true);
  const promptEffectSamu = end.samu && end4k.samu ? end4k.samu.median_ms / end.samu.median_ms : null;
  const promptEffectRg = end.rglru && end4k.rglru ? end4k.rglru.median_ms / end.rglru.median_ms : null;
  const maxThroughput = model => throughput.filter(row => row.model === model).sort((a, b) => b.tokens_per_second - a.tokens_per_second)[0];
  const ts = maxThroughput("samu"), tr = maxThroughput("rglru");
  const stress = data.throughput_saturation_stress || {};
  const stressPeak = model => (stress[model] || [])
    .filter(row => row.status === "measured")
    .sort((a, b) => b.tokens_per_second - a.tokens_per_second)[0];
  const ss = stressPeak("samu"), sr = stressPeak("rglru");
  const stressBlock = ss && sr ? `<div class="stress-note"><h3>大批量压力测试（短轨迹，单独报告）</h3><p>8-step 压力测试扩展到 B&gt;128，用来观察吞吐饱和点：SAMU 在已测范围内的最高点为 B=${ss.batch}、${fmt(ss.tokens_per_second, 1)} token/s；RG‑LRU 为 B=${sr.batch}、${fmt(sr.tokens_per_second, 1)} token/s。它不能替代 512–4096 token 的完整轨迹，因此不会参与上方速度比。</p></div>` : "";
  document.querySelector("#system-result").innerHTML = `<div class="system-facts">
    <div><h3>实验环境与顺序控制</h3><p>${data.environment?.gpu || "H800"}，${fmt((data.environment?.gpu_total_memory_bytes || 0) / 2 ** 30, 1)} GiB；两次完整运行采用相反的模型顺序和前缀顺序，合并原始 CUDA 样本后重新计算中位数。</p></div>
    <div><h3>参数与缓存已经匹配</h3><p>SAMU ${fmt(modelSamu?.parameter_count || 0, 0)} 个参数，RG‑LRU ${fmt(modelRg?.parameter_count || 0, 0)} 个参数，只相差 ${fmt(Math.abs((modelSamu?.parameter_count || 0) - (modelRg?.parameter_count || 0)), 0)} 个。每个序列的 FP32 递推状态均为 ${fmt(modelSamu?.state_bytes_per_sequence || 0, 0)} 字节。</p></div>
    <div><h3>固定批量的连续生成延迟</h3><p>${speed ? `B=16、空前缀、连续生成 4096 个 token 时，${speed >= 1 ? `SAMU 用时少 ${fmt((1 - 1 / speed) * 100, 2)}%` : `RG‑LRU 用时少 ${fmt((1 - speed) * 100, 2)}%`}。` : "结果待写入。"}${latencyRange ? `六个生成长度上的 SAMU 加速范围为 ${fmt(latencyRange.min, 3)}–${fmt(latencyRange.max, 3)} 倍。` : ""}${promptEffectSamu ? `4K 前缀不改变缓存大小；实测延迟相对空前缀为 SAMU ${fmt(promptEffectSamu, 3)} 倍、RG‑LRU ${fmt(promptEffectRg, 3)} 倍。` : ""}</p></div>
    <div><h3>B≤128 的完整轨迹吞吐</h3><p>${ts && tr ? `在预先声明的 B=1–128 范围内，SAMU 最高为 ${fmt(ts.tokens_per_second, 1)} token/s（批量 ${ts.batch}），RG‑LRU 最高为 ${fmt(tr.tokens_per_second, 1)} token/s（批量 ${tr.batch}）。${throughputRange ? `四个生成长度上的 SAMU/RG‑LRU 吞吐比范围为 ${fmt(throughputRange.min, 3)}–${fmt(throughputRange.max, 3)} 倍。` : ""}` : "结果待写入。"}</p></div>
  </div>${stressBlock}<p class="fairness-inline">系统级时间路径中，两边每层都包含“BF16 投影 + Triton 递推”两个逻辑算子。实际 CUDA 启动数由矩阵乘实现和 shape 决定：H800 profiler 中 B=1/128 两边均为 2 次，B=16 时 SAMU 的 cuBLAS split-K 另有 1 次归约，共 3 次，RG‑LRU 为 2 次。这里的“最高”只指 B≤128 且完成 512–4096 token 整条轨迹的候选，不外推为无限制单卡最大吞吐。完整原始样本与独立的大批量短轨迹压力测试见 <a href="${resultRoot}/summary.json">系统实验 JSON</a>。</p>`;
}

export async function initSystemAnalysis() {
  try {
    const response = await fetch(`${resultRoot}/summary.json`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    document.querySelector("#system-result").innerHTML = `<p><b>H800 完整实验结果尚未写入。</b> 数据文件生成后，本节会自动显示连续解码与 B≤128 的完整轨迹吞吐曲线。</p>`;
    drawLines(document.querySelector("#section5-latency-chart"), [], { yFormat: String, xLabel: "" });
    drawLines(document.querySelector("#section5-throughput-chart"), [], { yFormat: String, xLabel: "" });
  }
}
