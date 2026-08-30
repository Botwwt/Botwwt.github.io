const VERSION = "20260831-2";
const root = "results/gpu_optimization";

const paths = {
  mixer: "selected_dispatch_grouped_k32_h800.json",
  counterexample: "selected_dispatch_l8192_d1024_serial_k32_h800.json",
  length: "selected_dispatch_length_scaling_grouped_k32_h800.json",
  veryLong: "selected_dispatch_very_long_hybrid_h800_v2.json",
  width: "selected_dispatch_width_scaling_extra_grouped_k32_h800.json",
  block: "block_dispatch_grouped_k32_h800.json",
  optimizer: "optimizer_step_grouped_k32_h800.json",
  fattori: "public_fattori_h800.json",
  scanOnly: "public_accelerated_scan_h800.json",
};

const fetchJSON = async name => {
  const response = await fetch(`${root}/${name}?v=${VERSION}`);
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response.json();
};

const fmt = (value, digits = 3) => Number(value).toFixed(digits);
const pct = (reference, candidate) => (100 * (reference - candidate) / reference);
const gib = bytes => Number(bytes) / (1024 ** 3);
const shapeKey = shape => `${shape.batch}/${shape.length}/${shape.width}`;
const shapeLabel = shape => `B${shape.batch} · L${shape.length >= 1024 ? `${shape.length / 1024}K` : shape.length} · D${shape.width}`;
const byArchitecture = rows => {
  const grouped = new Map();
  for (const row of rows) {
    const key = shapeKey(row.shape);
    if (!grouped.has(key)) grouped.set(key, {shape: row.shape});
    grouped.get(key)[row.architecture] = row;
  }
  return [...grouped.values()];
};

const statsText = stats => `${fmt(stats.median_ms)} [${fmt(stats.minimum_ms)}, ${fmt(stats.maximum_ms)}]`;
const table = (headers, rows) => `<table><thead><tr>${headers.map(value => `<th>${value}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;

function setTheme() {
  const button = document.querySelector("#theme-toggle");
  const saved = localStorage.getItem("samu-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  button.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("samu-theme", next);
  });
}

function svgText(x, y, text, className = "axis", anchor = "start") {
  return `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${text}</text>`;
}

function observeChart(svg, draw) {
  const render = () => draw(Math.max(320, Math.round(svg.getBoundingClientRect().width || 620)));
  render();
  const observer = new ResizeObserver(render);
  observer.observe(svg);
}

function drawMixerChart(svg, groups, width) {
  const height = width < 500 ? 430 : 410;
  const margin = {top: 55, right: width < 500 ? 30 : 58, bottom: 54, left: width < 500 ? 118 : 142};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const all = groups.flatMap(group => [group.rglru, group.samu]);
  const maximum = Math.max(...all.map(row => row.forward_backward.maximum_ms));
  const xMax = Math.ceil(maximum / 2) * 2;
  const x = value => margin.left + (value / xMax) * plotWidth;
  const band = plotHeight / groups.length;
  const ticks = width < 500 ? 4 : 5;
  let content = `<title>complete mixer F+B latency</title><desc>每个形状显示 SAMU 与 strengthened RG-LRU 的中位数和最小最大区间。</desc>`;
  content += `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" class="frame"/>`;
  for (let i = 0; i <= ticks; i++) {
    const value = xMax * i / ticks;
    const xx = x(value);
    content += `<line x1="${xx}" y1="${margin.top}" x2="${xx}" y2="${margin.top + plotHeight}" class="grid"/>`;
    content += svgText(xx, margin.top + plotHeight + 20, fmt(value, value < 10 ? 1 : 0), "axis", "middle");
  }
  groups.forEach((group, index) => {
    const center = margin.top + band * (index + .5);
    content += svgText(margin.left - 12, center + 4, shapeLabel(group.shape), "axis", "end");
    [[group.rglru, -10, "rglru", "RG-LRU"], [group.samu, 10, "samu", "SAMU"]].forEach(([row, offset, cls, label]) => {
      const stat = row.forward_backward;
      const y = center + offset;
      content += `<line x1="${x(stat.minimum_ms)}" y1="${y}" x2="${x(stat.maximum_ms)}" y2="${y}" class="whisker ${cls}"/>`;
      content += `<line x1="${x(stat.minimum_ms)}" y1="${y - 4}" x2="${x(stat.minimum_ms)}" y2="${y + 4}" class="whisker ${cls}"/><line x1="${x(stat.maximum_ms)}" y1="${y - 4}" x2="${x(stat.maximum_ms)}" y2="${y + 4}" class="whisker ${cls}"/>`;
      content += `<circle cx="${x(stat.median_ms)}" cy="${y}" r="4.2" class="${cls}"><title>${label}: ${statsText(stat)} ms</title></circle>`;
      content += svgText(Math.min(width - 5, x(stat.maximum_ms) + 42), y + 4, fmt(stat.median_ms), "value-label", "end");
    });
  });
  content += `<circle cx="${margin.left}" cy="20" r="4" class="samu"/>${svgText(margin.left + 10, 24, "SAMU", "direct-label")}`;
  content += `<circle cx="${margin.left + 82}" cy="20" r="4" class="rglru"/>${svgText(margin.left + 92, 24, "strengthened RG-LRU", "direct-label")}`;
  content += svgText(margin.left + plotWidth / 2, height - 12, "F+B latency (ms; lower is better)", "axis-title", "middle");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = content;
}

function drawLengthChart(svg, points, width) {
  const height = width < 500 ? 430 : 410;
  const margin = {top: 50, right: width < 500 ? 38 : 76, bottom: 58, left: 66};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maximum = Math.max(...points.flatMap(point => [point.rglru, point.samu]).map(row => row.forward_backward.maximum_ms));
  const yMax = Math.ceil(maximum / 5) * 5;
  const x = index => margin.left + index * plotWidth / (points.length - 1);
  const y = value => margin.top + plotHeight - (value / yMax) * plotHeight;
  let content = `<title>D=1024 length crossover</title><desc>8K 时 SAMU 较慢，32K 后 SAMU 较快；65K 与 131K 使用 hybrid dispatch。</desc>`;
  content += `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" class="frame"/>`;
  for (let i = 0; i <= 4; i++) {
    const value = yMax * i / 4;
    const yy = y(value);
    content += `<line x1="${margin.left}" y1="${yy}" x2="${margin.left + plotWidth}" y2="${yy}" class="grid"/>${svgText(margin.left - 10, yy + 4, fmt(value, 0), "axis", "end")}`;
  }
  points.forEach((point, index) => { content += svgText(x(index), margin.top + plotHeight + 22, `${point.length / 1024}K`, "axis", "middle"); });
  for (const [architecture, cls, label] of [["rglru", "rglru", "strengthened RG-LRU"], ["samu", "samu", "SAMU"]]) {
    const d = points.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point[architecture].forward_backward.median_ms)}`).join(" ");
    content += `<path d="${d}" class="series-line ${cls}"/>`;
    points.forEach((point, index) => {
      const stat = point[architecture].forward_backward;
      content += `<line x1="${x(index)}" y1="${y(stat.minimum_ms)}" x2="${x(index)}" y2="${y(stat.maximum_ms)}" class="whisker ${cls}"/>`;
      content += `<circle cx="${x(index)}" cy="${y(stat.median_ms)}" r="4.3" class="${cls}"><title>${label}, L=${point.length}: ${statsText(stat)} ms</title></circle>`;
    });
    const last = points.at(-1)[architecture].forward_backward.median_ms;
    content += svgText(x(points.length - 1) - 8, y(last) + (architecture === "samu" ? 14 : -8), label, "direct-label", "end");
  }
  const first = points[0];
  const loss = -pct(first.rglru.forward_backward.median_ms, first.samu.forward_backward.median_ms);
  content += svgText(x(0) + 8, y(first.samu.forward_backward.median_ms) - 10, `${fmt(loss, 1)}% slower`, "value-label");
  content += svgText(margin.left + plotWidth / 2, height - 13, "sequence length L", "axis-title", "middle");
  content += `<text x="16" y="${margin.top + plotHeight / 2}" class="axis-title" text-anchor="middle" transform="rotate(-90 16 ${margin.top + plotHeight / 2})">F+B latency (ms)</text>`;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = content;
}

function drawWidthChart(svg, points, width) {
  const height = width < 500 ? 410 : 390;
  const margin = {top: 42, right: 36, bottom: 58, left: 66};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const yMin = 34, yMax = 43;
  const x = index => margin.left + index * plotWidth / (points.length - 1);
  const y = value => margin.top + plotHeight - (value - yMin) / (yMax - yMin) * plotHeight;
  let content = `<title>L=32768 width scaling</title><desc>SAMU F+B latency advantage across D=1024,1536,2048,2560。</desc>`;
  content += `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" class="frame"/>`;
  [35, 37, 39, 41, 43].forEach(value => { const yy = y(value); content += `<line x1="${margin.left}" y1="${yy}" x2="${margin.left + plotWidth}" y2="${yy}" class="grid"/>${svgText(margin.left - 9, yy + 4, `${value}%`, "axis", "end")}`; });
  points.forEach((point, index) => { content += svgText(x(index), margin.top + plotHeight + 22, String(point.width), "axis", "middle"); });
  const d = points.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.advantage)}`).join(" ");
  content += `<path d="${d}" class="series-line samu"/>`;
  points.forEach((point, index) => {
    content += `<circle cx="${x(index)}" cy="${y(point.advantage)}" r="4.5" class="samu"><title>D=${point.width}: ${fmt(point.advantage, 1)}% faster</title></circle>`;
    content += svgText(x(index), y(point.advantage) - 10, `${fmt(point.advantage, 1)}%`, "value-label", "middle");
  });
  content += svgText(margin.left + plotWidth / 2, height - 13, "real recurrent width D", "axis-title", "middle");
  content += `<text x="16" y="${margin.top + plotHeight / 2}" class="axis-title" text-anchor="middle" transform="rotate(-90 16 ${margin.top + plotHeight / 2})">SAMU F+B advantage</text>`;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = content;
}

function renderMixerTable(groups) {
  const rows = groups.map(group => {
    const rg = group.rglru, samu = group.samu;
    const advantage = pct(rg.forward_backward.median_ms, samu.forward_backward.median_ms);
    const memory = pct(rg.peak_allocated_delta_bytes, samu.peak_allocated_delta_bytes);
    return `<tr><td>${shapeLabel(group.shape)}</td><td>${fmt(rg.forward.median_ms)}</td><td>${fmt(samu.forward.median_ms)}</td><td>${fmt(rg.backward.median_ms)}</td><td>${fmt(samu.backward.median_ms)}</td><td>${statsText(rg.forward_backward)}</td><td class="best">${statsText(samu.forward_backward)}</td><td class="best">${fmt(advantage, 1)}%</td><td class="best">${fmt(memory, 1)}%</td></tr>`;
  });
  document.querySelector("#mixer-table").innerHTML = table(["shape (B,L,D)", "RG F", "SAMU F", "RG B", "SAMU B", "RG F+B [min,max]", "SAMU F+B [min,max]", "latency ↓", "allocated ↓"], rows);
}

function renderLengthTable(points) {
  const rows = points.map(point => {
    const rg = point.rglru.forward_backward, samu = point.samu.forward_backward;
    const advantage = pct(rg.median_ms, samu.median_ms);
    return `<tr><td>${point.length.toLocaleString()}</td><td>${statsText(rg)}</td><td>${statsText(samu)}</td><td class="${advantage >= 0 ? "best" : "loss"}">${advantage >= 0 ? `${fmt(advantage, 1)}% faster` : `${fmt(-advantage, 1)}% slower`}</td><td>${point.path}</td></tr>`;
  });
  document.querySelector("#length-table").innerHTML = table(["L", "RG F+B", "SAMU F+B", "SAMU", "SAMU path"], rows);
}

function renderWidthTable(points) {
  const rows = points.map(point => `<tr><td>${point.width}</td><td>${fmt(point.rglru)}</td><td>${fmt(point.samu)}</td><td class="best">${fmt(point.advantage, 1)}%</td></tr>`);
  document.querySelector("#width-table").innerHTML = table(["D", "RG F+B", "SAMU F+B", "SAMU faster"], rows);
}

const caseNames = {
  state2048_short: "short · B4 L2K · Dmodel/RNN 2048/2048",
  state2560_medium: "medium · B1 L8K · Dmodel/RNN 2560/2560",
  state1024_long: "long · B1 L32K · Dmodel/RNN 1024/1024",
  "400m_block_long": "400M-width · B1 L32K · Dmodel/RNN 1536/2048（单 block）",
};

function renderBlockTable(data) {
  const rows = data.cases.map(item => {
    const rg = item.rows.find(row => row.architecture === "rglru");
    const samu = item.rows.find(row => row.architecture === "samu");
    return `<tr><td>${caseNames[item.case]}</td><td>${statsText(rg.forward_backward)}</td><td class="best">${statsText(samu.forward_backward)}</td><td class="best">${fmt(pct(rg.forward_backward.median_ms, samu.forward_backward.median_ms), 1)}%</td><td>${fmt(gib(rg.peak_allocated_delta_bytes), 3)}</td><td class="best">${fmt(gib(samu.peak_allocated_delta_bytes), 3)}</td><td>${rg.parameters.toLocaleString()}</td><td>${samu.parameters.toLocaleString()}</td></tr>`;
  });
  document.querySelector("#block-table").innerHTML = table(["one-block case", "RG F+B", "SAMU F+B", "SAMU faster", "RG alloc GiB", "SAMU alloc GiB", "RG params", "SAMU params"], rows);
}

function renderOptimizerTable(data) {
  const rows = data.cases.map(item => {
    const rg = item.rows.find(row => row.architecture === "rglru");
    const samu = item.rows.find(row => row.architecture === "samu");
    return `<tr><td>${caseNames[item.case]}</td><td>${statsText(rg.optimizer_step)}</td><td class="best">${statsText(samu.optimizer_step)}</td><td class="best">${fmt(pct(rg.optimizer_step.median_ms, samu.optimizer_step.median_ms), 1)}%</td><td>${fmt(gib(rg.peak_allocated_delta_bytes), 3)} / ${fmt(gib(samu.peak_allocated_delta_bytes), 3)}</td><td>${fmt(gib(rg.peak_reserved_delta_bytes), 3)} / ${fmt(gib(samu.peak_reserved_delta_bytes), 3)}</td></tr>`;
  });
  document.querySelector("#optimizer-table").innerHTML = table(["one-block case", "RG step", "SAMU step", "SAMU faster", "allocated RG/SAMU GiB", "reserved RG/SAMU GiB"], rows);
}

function renderPublicTable(fattori, scanOnly) {
  const byName = (data, name) => data.rows.find(row => row.implementation === name);
  const fattoriRow = byName(fattori, "fattori_original");
  const ours = byName(fattori, "ours_restricted");
  const hippogriff = byName(scanOnly, "hippogriff_accelerated_scan");
  const lingua = byName(scanOnly, "lingua_original_wrapper");
  const chunk32 = byName(scanOnly, "ours_materialized_chunk32");
  document.querySelector("#public-table").innerHTML = `
    <h4>complete restricted no-reset · B4 L2048 D2048 · PyTorch 2.4.1/CUDA 12.4</h4>
    ${table(["implementation", "F", "F+B", "contract"], [
      `<tr><td>ours restricted</td><td>${fmt(ours.forward.median_ms)}</td><td class="best">${fmt(ours.forward_backward.median_ms)}</td><td>zero h₀, no reset secondary comparison</td></tr>`,
      `<tr><td>Fattori original hawk-pytorch</td><td>${fmt(fattoriRow.forward.median_ms)}</td><td>${fmt(fattoriRow.forward_backward.median_ms)}</td><td>original source; same restricted contract</td></tr>`,
    ])}
    <p class="table-note">2.260 ms vs 4.535 ms：ours latency 低 50.2%，或等价为约 2.01× throughput ratio。该表不进入 canonical reset 主排名。</p>
    <h4>pure scan-only · coefficients pre-materialized · PyTorch 2.4.1/CUDA 12.4</h4>
    ${table(["implementation", "F", "F+B", "结论"], [
      `<tr><td>accelerated-scan / Hippogriff</td><td>${fmt(hippogriff.forward.median_ms)}</td><td class="best">${fmt(hippogriff.forward_backward.median_ms)}</td><td>本表最快</td></tr>`,
      `<tr><td>Lingua original wrapper</td><td>${fmt(lingua.forward.median_ms)}</td><td>${fmt(lingua.forward_backward.median_ms)}</td><td>调用同一 accelerated-scan kernel</td></tr>`,
      `<tr><td>ours generic materialized chunk32</td><td>${fmt(chunk32.forward.median_ms)}</td><td>${fmt(chunk32.forward_backward.median_ms)}</td><td>不是最快 pure scan</td></tr>`,
    ])}`;
}

async function main() {
  setTheme();
  try {
    const data = {};
    await Promise.all(Object.entries(paths).map(async ([key, path]) => { data[key] = await fetchJSON(path); }));
    const mixerGroups = byArchitecture(data.mixer.rows);
    const counter = byArchitecture(data.counterexample.rows)[0];
    const length32 = byArchitecture(data.length.rows).find(group => group.shape.length === 32768);
    const veryLong = byArchitecture(data.veryLong.rows);
    const lengthPoints = [
      {length: 8192, rglru: counter.rglru, samu: counter.samu, path: "serial K32"},
      {length: 32768, rglru: length32.rglru, samu: length32.samu, path: "grouped64 K32"},
      ...veryLong.map(group => ({length: group.shape.length, rglru: group.rglru, samu: group.samu, path: "serial F + grouped reverse"})),
    ];
    const primaryWidth = mixerGroups.filter(group => group.shape.length === 32768 && [1024, 2048].includes(group.shape.width));
    const extraWidth = byArchitecture(data.width.rows);
    const widthPoints = [...primaryWidth, ...extraWidth].map(group => ({
      width: group.shape.width,
      rglru: group.rglru.forward_backward.median_ms,
      samu: group.samu.forward_backward.median_ms,
      advantage: pct(group.rglru.forward_backward.median_ms, group.samu.forward_backward.median_ms),
    })).sort((a, b) => a.width - b.width);

    renderMixerTable(mixerGroups);
    renderLengthTable(lengthPoints);
    renderWidthTable(widthPoints);
    renderBlockTable(data.block);
    renderOptimizerTable(data.optimizer);
    renderPublicTable(data.fattori, data.scanOnly);
    observeChart(document.querySelector("#mixer-chart"), width => drawMixerChart(document.querySelector("#mixer-chart"), mixerGroups, width));
    observeChart(document.querySelector("#length-chart"), width => drawLengthChart(document.querySelector("#length-chart"), lengthPoints, width));
    observeChart(document.querySelector("#width-chart"), width => drawWidthChart(document.querySelector("#width-chart"), widthPoints, width));
    document.querySelector("#environment-line").textContent = `主结果环境：${data.mixer.environment.gpu} · PyTorch ${data.mixer.environment.torch} · CUDA ${data.mixer.environment.cuda} · 编译排除 · AB/BA · 10 timed samples/row。公开库复现环境：${data.fattori.environment.gpu} · PyTorch ${data.fattori.environment.torch} · CUDA ${data.fattori.environment.cuda}。`;
  } catch (error) {
    console.error(error);
    document.querySelectorAll(".loading").forEach(node => { node.textContent = `结果加载失败：${error.message}`; node.setAttribute("role", "alert"); });
    document.querySelector("#environment-line").textContent = `环境记录加载失败：${error.message}`;
  }
}

main();
