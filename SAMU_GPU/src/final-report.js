const VERSION = "20260831-4";
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
  if (!response.ok) throw new Error(`${name}：HTTP ${response.status}`);
  return response.json();
};

const fmt = (value, digits = 3) => Number(value).toFixed(digits);
const pct = (reference, candidate) => 100 * (reference - candidate) / reference;
const gib = bytes => Number(bytes) / (1024 ** 3);
const shapeKey = shape => `${shape.batch}/${shape.length}/${shape.width}`;
const shapeLabel = shape => `B=${shape.batch} · L=${shape.length >= 1024 ? `${shape.length / 1024}K` : shape.length} · D=${shape.width}`;
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

function svgText(x, y, value, className = "axis", anchor = "start") {
  return `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${value}</text>`;
}

function observeChart(svg, draw) {
  const render = () => draw(Math.max(320, Math.round(svg.getBoundingClientRect().width || 620)));
  render();
  const observer = new ResizeObserver(render);
  observer.observe(svg);
}

function drawMixerChart(svg, groups, width) {
  const height = width < 500 ? 430 : 410;
  const margin = {top: 55, right: width < 500 ? 30 : 58, bottom: 58, left: width < 500 ? 124 : 148};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maximum = Math.max(...groups.flatMap(group => [group.rglru, group.samu]).map(row => row.forward_backward.maximum_ms));
  const xMax = Math.ceil(maximum / 2) * 2;
  const x = value => margin.left + value / xMax * plotWidth;
  const band = plotHeight / groups.length;
  const ticks = width < 500 ? 4 : 5;
  let content = `<title>完整递归混合器前向加反向延迟</title><desc>每个形状显示 SAMU 与经过充分优化的 RG-LRU 的中位数和最小值至最大值区间。</desc>`;
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
    [[group.rglru, -10, "rglru", "优化后的 RG-LRU"], [group.samu, 10, "samu", "SAMU"]].forEach(([row, offset, cls, label]) => {
      const stat = row.forward_backward;
      const y = center + offset;
      content += `<line x1="${x(stat.minimum_ms)}" y1="${y}" x2="${x(stat.maximum_ms)}" y2="${y}" class="whisker ${cls}"/>`;
      content += `<line x1="${x(stat.minimum_ms)}" y1="${y - 4}" x2="${x(stat.minimum_ms)}" y2="${y + 4}" class="whisker ${cls}"/><line x1="${x(stat.maximum_ms)}" y1="${y - 4}" x2="${x(stat.maximum_ms)}" y2="${y + 4}" class="whisker ${cls}"/>`;
      content += `<circle cx="${x(stat.median_ms)}" cy="${y}" r="4.2" class="${cls}"><title>${label}：${statsText(stat)} 毫秒</title></circle>`;
      content += svgText(Math.min(width - 5, x(stat.maximum_ms) + 42), y + 4, fmt(stat.median_ms), "value-label", "end");
    });
  });
  content += `<circle cx="${margin.left}" cy="20" r="4" class="samu"/>${svgText(margin.left + 10, 24, "SAMU", "direct-label")}`;
  content += `<circle cx="${margin.left + 82}" cy="20" r="4" class="rglru"/>${svgText(margin.left + 92, 24, "优化后的 RG-LRU", "direct-label")}`;
  content += svgText(margin.left + plotWidth / 2, height - 12, "前向+反向延迟（毫秒，越低越好）", "axis-title", "middle");
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
  const y = value => margin.top + plotHeight - value / yMax * plotHeight;
  let content = `<title>D=1024 时的序列长度转折</title><desc>L=8192 时 SAMU 较慢，L=32768 后 SAMU 较快，L=65536 与 L=131072 使用混合调度。</desc>`;
  content += `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" class="frame"/>`;
  for (let i = 0; i <= 4; i++) {
    const value = yMax * i / 4;
    const yy = y(value);
    content += `<line x1="${margin.left}" y1="${yy}" x2="${margin.left + plotWidth}" y2="${yy}" class="grid"/>${svgText(margin.left - 10, yy + 4, fmt(value, 0), "axis", "end")}`;
  }
  points.forEach((point, index) => { content += svgText(x(index), margin.top + plotHeight + 22, `${point.length / 1024}K`, "axis", "middle"); });
  for (const [architecture, cls, label] of [["rglru", "rglru", "优化后的 RG-LRU"], ["samu", "samu", "SAMU"]]) {
    const d = points.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point[architecture].forward_backward.median_ms)}`).join(" ");
    content += `<path d="${d}" class="series-line ${cls}"/>`;
    points.forEach((point, index) => {
      const stat = point[architecture].forward_backward;
      content += `<line x1="${x(index)}" y1="${y(stat.minimum_ms)}" x2="${x(index)}" y2="${y(stat.maximum_ms)}" class="whisker ${cls}"/>`;
      content += `<circle cx="${x(index)}" cy="${y(stat.median_ms)}" r="4.3" class="${cls}"><title>${label}，L=${point.length}：${statsText(stat)} 毫秒</title></circle>`;
    });
    const last = points.at(-1)[architecture].forward_backward.median_ms;
    content += svgText(x(points.length - 1) - 8, y(last) + (architecture === "samu" ? 14 : -8), label, "direct-label", "end");
  }
  const first = points[0];
  const loss = -pct(first.rglru.forward_backward.median_ms, first.samu.forward_backward.median_ms);
  content += svgText(x(0) + 8, y(first.samu.forward_backward.median_ms) - 10, `慢 ${fmt(loss, 1)}%`, "value-label");
  content += svgText(margin.left + plotWidth / 2, height - 13, "序列长度 L", "axis-title", "middle");
  content += `<text x="16" y="${margin.top + plotHeight / 2}" class="axis-title" text-anchor="middle" transform="rotate(-90 16 ${margin.top + plotHeight / 2})">前向+反向延迟（毫秒）</text>`;
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
  let content = `<title>L=32768 时的状态宽度扩展</title><desc>SAMU 在 D=1024、1536、2048、2560 时的前向加反向延迟优势。</desc>`;
  content += `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" class="frame"/>`;
  [35, 37, 39, 41, 43].forEach(value => {
    const yy = y(value);
    content += `<line x1="${margin.left}" y1="${yy}" x2="${margin.left + plotWidth}" y2="${yy}" class="grid"/>${svgText(margin.left - 9, yy + 4, `${value}%`, "axis", "end")}`;
  });
  points.forEach((point, index) => { content += svgText(x(index), margin.top + plotHeight + 22, String(point.width), "axis", "middle"); });
  const d = points.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.advantage)}`).join(" ");
  content += `<path d="${d}" class="series-line samu"/>`;
  points.forEach((point, index) => {
    content += `<circle cx="${x(index)}" cy="${y(point.advantage)}" r="4.5" class="samu"><title>D=${point.width}：快 ${fmt(point.advantage, 1)}%</title></circle>`;
    content += svgText(x(index), y(point.advantage) - 10, `${fmt(point.advantage, 1)}%`, "value-label", "middle");
  });
  content += svgText(margin.left + plotWidth / 2, height - 13, "实数状态宽度 D", "axis-title", "middle");
  content += `<text x="16" y="${margin.top + plotHeight / 2}" class="axis-title" text-anchor="middle" transform="rotate(-90 16 ${margin.top + plotHeight / 2})">SAMU 前向+反向优势</text>`;
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
  document.querySelector("#mixer-table").innerHTML = table(["形状（B，L，D）", "RG-LRU 前向", "SAMU 前向", "RG-LRU 反向", "SAMU 反向", "RG-LRU 前向+反向 [最小，最大]", "SAMU 前向+反向 [最小，最大]", "延迟降低", "实际分配显存降低"], rows);
}

function renderLengthTable(points) {
  const rows = points.map(point => {
    const rg = point.rglru.forward_backward, samu = point.samu.forward_backward;
    const advantage = pct(rg.median_ms, samu.median_ms);
    return `<tr><td>${point.length.toLocaleString()}</td><td>${statsText(rg)}</td><td>${statsText(samu)}</td><td class="${advantage >= 0 ? "best" : "loss"}">${advantage >= 0 ? `快 ${fmt(advantage, 1)}%` : `慢 ${fmt(-advantage, 1)}%`}</td><td>${point.path}</td></tr>`;
  });
  document.querySelector("#length-table").innerHTML = table(["序列长度 L", "RG-LRU 前向+反向", "SAMU 前向+反向", "SAMU 相对结果", "SAMU 路径"], rows);
}

function renderWidthTable(points) {
  const rows = points.map(point => `<tr><td>${point.width}</td><td>${fmt(point.rglru)}</td><td>${fmt(point.samu)}</td><td class="best">${fmt(point.advantage, 1)}%</td></tr>`);
  document.querySelector("#width-table").innerHTML = table(["状态宽度 D", "RG-LRU 前向+反向", "SAMU 前向+反向", "SAMU 快"], rows);
}

const caseNames = {
  state2048_short: "短形状 · B=4，L=2K · 模型宽度/实状态宽度=2048/2048",
  state2560_medium: "中形状 · B=1，L=8K · 模型宽度/实状态宽度=2560/2560",
  state1024_long: "长形状 · B=1，L=32K · 模型宽度/实状态宽度=1024/1024",
  "400m_block_long": "400M 宽度配置 · B=1，L=32K · 模型宽度/实状态宽度=1536/2048（单循环块）",
};

function renderBlockTable(data) {
  const rows = data.cases.map(item => {
    const rg = item.rows.find(row => row.architecture === "rglru");
    const samu = item.rows.find(row => row.architecture === "samu");
    return `<tr><td>${caseNames[item.case]}</td><td>${statsText(rg.forward_backward)}</td><td class="best">${statsText(samu.forward_backward)}</td><td class="best">${fmt(pct(rg.forward_backward.median_ms, samu.forward_backward.median_ms), 1)}%</td><td>${fmt(gib(rg.peak_allocated_delta_bytes), 3)}</td><td class="best">${fmt(gib(samu.peak_allocated_delta_bytes), 3)}</td><td>${rg.parameters.toLocaleString()}</td><td>${samu.parameters.toLocaleString()}</td></tr>`;
  });
  document.querySelector("#block-table").innerHTML = table(["单循环块形状", "RG-LRU 前向+反向", "SAMU 前向+反向", "SAMU 快", "RG-LRU 实际分配 GiB", "SAMU 实际分配 GiB", "RG-LRU 参数量", "SAMU 参数量"], rows);
}

function renderOptimizerTable(data) {
  const rows = data.cases.map(item => {
    const rg = item.rows.find(row => row.architecture === "rglru");
    const samu = item.rows.find(row => row.architecture === "samu");
    return `<tr><td>${caseNames[item.case]}</td><td>${statsText(rg.optimizer_step)}</td><td class="best">${statsText(samu.optimizer_step)}</td><td class="best">${fmt(pct(rg.optimizer_step.median_ms, samu.optimizer_step.median_ms), 1)}%</td><td>${fmt(gib(rg.peak_allocated_delta_bytes), 3)} / ${fmt(gib(samu.peak_allocated_delta_bytes), 3)}</td><td>${fmt(gib(rg.peak_reserved_delta_bytes), 3)} / ${fmt(gib(samu.peak_reserved_delta_bytes), 3)}</td></tr>`;
  });
  document.querySelector("#optimizer-table").innerHTML = table(["单循环块形状", "RG-LRU 一步", "SAMU 一步", "SAMU 快", "实际分配 RG-LRU/SAMU GiB", "保留显存 RG-LRU/SAMU GiB"], rows);
}

function renderPublicTable(fattori, scanOnly) {
  const byName = (data, name) => data.rows.find(row => row.implementation === name);
  const fattoriRow = byName(fattori, "fattori_original");
  const ours = byName(fattori, "ours_restricted");
  const hippogriff = byName(scanOnly, "hippogriff_accelerated_scan");
  const lingua = byName(scanOnly, "lingua_original_wrapper");
  const chunk32 = byName(scanOnly, "ours_materialized_chunk32");
  document.querySelector("#public-table").innerHTML = `
    <h4>受限完整无重置路径 · B=4，L=2048，D=2048 · PyTorch 2.4.1/CUDA 12.4</h4>
    ${table(["实现", "前向", "前向+反向", "测量约定"], [
      `<tr><td>我们的受限路径</td><td>${fmt(ours.forward.median_ms)}</td><td class="best">${fmt(ours.forward_backward.median_ms)}</td><td>零初始状态、无重置的次要对比</td></tr>`,
      `<tr><td>Fattori 原始 hawk-pytorch</td><td>${fmt(fattoriRow.forward.median_ms)}</td><td>${fmt(fattoriRow.forward_backward.median_ms)}</td><td>原仓库代码，相同受限约定</td></tr>`,
    ])}
    <p class="table-note">2.260 毫秒对 4.535 毫秒：我们的延迟低 50.2%，等价吞吐比约为 2.01 倍。该表不进入支持规范重置语义的主排名。</p>
    <h4>纯扫描 · 仿射系数预先生成 · PyTorch 2.4.1/CUDA 12.4</h4>
    ${table(["实现", "前向", "前向+反向", "结论"], [
      `<tr><td>accelerated-scan / Hippogriff</td><td>${fmt(hippogriff.forward.median_ms)}</td><td class="best">${fmt(hippogriff.forward_backward.median_ms)}</td><td>本表最快</td></tr>`,
      `<tr><td>Lingua 原始封装</td><td>${fmt(lingua.forward.median_ms)}</td><td>${fmt(lingua.forward_backward.median_ms)}</td><td>调用同一个 accelerated-scan 内核</td></tr>`,
      `<tr><td>我们的通用物化 K=32 分块扫描</td><td>${fmt(chunk32.forward.median_ms)}</td><td>${fmt(chunk32.forward_backward.median_ms)}</td><td>不是最快纯扫描</td></tr>`,
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
      {length: 8192, rglru: counter.rglru, samu: counter.samu, path: "每个程序处理 32 步；分块入口依次传播"},
      {length: 32768, rglru: length32.rglru, samu: length32.samu, path: "每个程序处理 32 步；每 64 个分块为一组求入口"},
      ...veryLong.map(group => ({length: group.shape.length, rglru: group.rglru, samu: group.samu, path: "前向入口依次传播；反向每 64 个分块为一组"})),
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
    document.querySelector("#environment-line").textContent = `主结果环境：${data.mixer.environment.gpu} · PyTorch ${data.mixer.environment.torch} · CUDA ${data.mixer.environment.cuda} · 首次编译排除 · 两轮交换先后顺序 · 每行 10 个计时样本。公开库复现环境：${data.fattori.environment.gpu} · PyTorch ${data.fattori.environment.torch} · CUDA ${data.fattori.environment.cuda}。`;
  } catch (error) {
    console.error(error);
    document.querySelectorAll(".loading").forEach(node => { node.textContent = `结果加载失败：${error.message}`; node.setAttribute("role", "alert"); });
    document.querySelector("#environment-line").textContent = `环境记录加载失败：${error.message}`;
  }
}

main();
