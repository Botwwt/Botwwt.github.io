const NS = "http://www.w3.org/2000/svg";
const SAMU = "var(--state)";
const RGLRU = "var(--rglru)";
const INK = "var(--ink)";
const MUTED = "var(--muted)";
const GRID = "var(--line)";
const PAPER = "var(--paper)";
const observers = new WeakMap();

function node(tag, attributes = {}, text = "") {
  const element = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  element.textContent = text;
  return element;
}

function label(svg, text, x, y, attributes = {}) {
  svg.append(node("text", {
    x, y, fill: INK, "font-size": 12,
    "font-family": "Inter, Noto Sans SC, sans-serif",
    ...attributes,
  }, text));
}

function clear(svg, title, description) {
  svg.replaceChildren(
    node("title", {}, title),
    node("desc", {}, description),
  );
}

function frame(svg, left, top, right, bottom) {
  svg.append(node("rect", {
    x: left, y: top, width: right - left, height: bottom - top,
    fill: "none", stroke: GRID, "stroke-width": 1,
    "data-chart-frame": "",
  }));
}

function horizontalGrid(svg, ticks, scale, left, right, format) {
  ticks.forEach(value => {
    const y = scale(value);
    svg.append(node("line", {x1: left, y1: y, x2: right, y2: y, stroke: GRID, "stroke-width": 1}));
    label(svg, format(value), left - 9, y + 4, {"text-anchor": "end", fill: MUTED, "font-size": 11});
  });
}

function axisTitle(svg, text, x, y, rotate = false) {
  label(svg, text, x, y, {
    "text-anchor": "middle", fill: MUTED, "font-size": 12,
    class: "axis-title", "data-axis": rotate ? "y" : "x",
    ...(rotate ? {transform: `rotate(-90 ${x} ${y})`} : {}),
  });
}

function marker(svg, shape, x, y, color, hollow = false, radius = 4.5) {
  const shared = {fill: hollow ? PAPER : color, stroke: color, "stroke-width": 1.8};
  if (shape === "square") {
    svg.append(node("rect", {x: x - radius, y: y - radius, width: radius * 2, height: radius * 2, ...shared}));
  } else {
    svg.append(node("circle", {cx: x, cy: y, r: radius, ...shared}));
  }
}

function legend(svg, items, left, top, availableWidth) {
  let x = left;
  let y = top;
  items.forEach(item => {
    const estimated = 34 + item.label.length * 12;
    if (x > left && x + estimated > left + availableWidth) {
      x = left;
      y += 22;
    }
    if (item.dashed) {
      svg.append(node("line", {x1: x, y1: y - 4, x2: x + 22, y2: y - 4, stroke: item.color, "stroke-width": 2, "stroke-dasharray": "5 4"}));
    } else {
      marker(svg, item.shape || "circle", x + 10, y - 4, item.color, item.hollow, 3.8);
    }
    label(svg, item.label, x + 29, y, {fill: MUTED, "font-size": 11});
    x += estimated;
  });
  return y;
}

function lineSeries(svg, points, x, y, {color, shape = "circle", dashed = false, hollow = false, error = false}) {
  const valid = points.filter(point => Number.isFinite(point.y));
  if (!valid.length) return;
  svg.append(node("path", {
    d: valid.map((point, index) => `${index ? "L" : "M"}${x(point)},${y(point.y)}`).join(" "),
    fill: "none", stroke: color, "stroke-width": 2.2,
    "stroke-dasharray": dashed ? "6 5" : "none",
  }));
  valid.forEach(point => {
    const px = x(point), py = y(point.y);
    if (error && Number.isFinite(point.low) && Number.isFinite(point.high)) {
      const low = y(point.low), high = y(point.high);
      svg.append(node("line", {x1: px, y1: low, x2: px, y2: high, stroke: color, "stroke-width": 1.2}));
      svg.append(node("line", {x1: px - 4, y1: low, x2: px + 4, y2: low, stroke: color, "stroke-width": 1.2}));
      svg.append(node("line", {x1: px - 4, y1: high, x2: px + 4, y2: high, stroke: color, "stroke-width": 1.2}));
    }
    marker(svg, shape, px, py, color, hollow);
  });
}

function responsive(svg, height, draw) {
  if (!svg) return;
  const paint = () => {
    const width = Math.max(300, Math.floor(svg.getBoundingClientRect().width || 640));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    draw(svg, width, height);
  };
  paint();
  if (typeof ResizeObserver !== "undefined" && !observers.has(svg)) {
    const observer = new ResizeObserver(() => requestAnimationFrame(paint));
    observer.observe(svg);
    observers.set(svg, observer);
  }
}

function quantile(values, probability) {
  const sorted = [...(values || [])].map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position), upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function compact(value) {
  if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e7 ? 0 : 1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(value >= 1e5 ? 0 : 1)}K`;
  return String(value);
}

function drawControlScaling(svg, width, height) {
  clear(svg, "每个词元的动态控制量", "RG-LRU 的动态门随递推宽度线性增加，SAMU 始终保持两个共享控制量。");
  const margin = {left: 68, right: 18, top: 55, bottom: 64};
  const left = margin.left, right = width - margin.right, top = margin.top, bottom = height - margin.bottom;
  const widths = [512, 1024, 2048, 2560, 4096, 8192];
  const x = point => left + widths.indexOf(point.width) * (right - left) / (widths.length - 1);
  const y = value => bottom - (Math.log2(value) - 1) / 13 * (bottom - top);
  frame(svg, left, top, right, bottom);
  horizontalGrid(svg, [2, 8, 32, 128, 512, 2048, 8192], y, left, right, compact);
  widths.forEach((value, index) => {
    if (width < 480 && ![0, 2, 4, 5].includes(index)) return;
    label(svg, value.toLocaleString(), x({width: value}), bottom + 23, {"text-anchor": "middle", fill: MUTED, "font-size": 11});
  });
  const rg = widths.map(value => ({width: value, y: 2 * value}));
  const samu = widths.map(value => ({width: value, y: 2}));
  lineSeries(svg, rg, x, y, {color: RGLRU, shape: "square"});
  lineSeries(svg, samu, x, y, {color: SAMU, shape: "circle"});
  legend(svg, [
    {label: "SAMU：2", color: SAMU, shape: "circle"},
    {label: "RG-LRU：2D", color: RGLRU, shape: "square"},
  ], left, 24, right - left);
  const focusX = x({width: 2560});
  label(svg, "D=2,560：少 2,560 倍", Math.min(right - 4, focusX + 10), y(5120) - 10, {"text-anchor": focusX > width * .6 ? "end" : "start", fill: INK, "font-size": 11});
  axisTitle(svg, "递推宽度 D（实状态标量）", (left + right) / 2, height - 15);
  axisTitle(svg, "动态标量/词元（对数刻度）", 17, (top + bottom) / 2, true);
}

function drawMixerParameters(svg, width, height, paperScale) {
  clear(svg, "每层递推混合器参数", "400M 与 1.3B 配置中，SAMU 共享控制混合器的参数远少于 RG-LRU 的 16 组块对角门。");
  const margin = {left: 68, right: 18, top: 55, bottom: 64};
  const left = margin.left, right = width - margin.right, top = margin.top, bottom = height - margin.bottom;
  const groups = [
    {scale: "400M", d: 2048},
    {scale: "1.3B", d: 2560},
  ].map(group => ({
    ...group,
    samu: 3 * group.d + 4,
    rglru: 2 * group.d * group.d / 16 + 3 * group.d,
  }));
  const y = value => bottom - (Math.log10(value) - 3) / 3 * (bottom - top);
  const centers = groups.map((_, index) => left + (index + .5) * (right - left) / groups.length);
  const barWidth = Math.min(46, (right - left) / 8);
  frame(svg, left, top, right, bottom);
  horizontalGrid(svg, [1e3, 1e4, 1e5, 1e6], y, left, right, compact);
  groups.forEach((group, index) => {
    const center = centers[index];
    const values = [
      {value: group.samu, color: SAMU, x: center - barWidth - 3},
      {value: group.rglru, color: RGLRU, x: center + 3},
    ];
    values.forEach(item => {
      svg.append(node("rect", {x: item.x, y: y(item.value), width: barWidth, height: bottom - y(item.value), fill: item.color, opacity: .82}));
      label(svg, compact(item.value), item.x + barWidth / 2, y(item.value) - 8, {"text-anchor": "middle", "font-size": 11});
    });
    label(svg, group.scale, center, bottom + 24, {"text-anchor": "middle", fill: MUTED, "font-size": 11});
    label(svg, `${(group.rglru / group.samu).toFixed(1)}×`, center, top + 17, {"text-anchor": "middle", fill: INK, "font-size": 12, "font-weight": 500});
  });
  legend(svg, [
    {label: "SAMU", color: SAMU, shape: "circle"},
    {label: "RG-LRU-16", color: RGLRU, shape: "square"},
  ], left, 24, right - left);
  const records = paperScale.records || [];
  const wholeModelReduction = ["400m", "1.3b"].map(scale => {
    const s = records.find(row => row.scale === scale && row.architecture === "samu")?.parameters;
    const r = records.find(row => row.scale === scale && row.architecture === "rglru")?.parameters;
    return Number.isFinite(s) && Number.isFinite(r) ? (1 - s / r) * 100 : NaN;
  });
  label(svg, `完整模型参数只减少 ${wholeModelReduction.map(value => value.toFixed(2)).join("% / ")}%`, right - 4, height - 15, {"text-anchor": "end", fill: MUTED, "font-size": 11});
  axisTitle(svg, "参数量（对数刻度）", 17, (top + bottom) / 2, true);
}

function trainingRows(paperScale) {
  const records = paperScale.records || [];
  const output = [];
  for (const scale of ["400m", "1.3b"]) {
    const samu = records.find(row => row.scale === scale && row.architecture === "samu");
    const rglru = records.find(row => row.scale === scale && row.architecture === "rglru");
    for (const length of [2048, 4096, 8192]) {
      const s = samu?.rows?.find(row => row.sequence_length === length && row.status === "measured");
      const r = rglru?.rows?.find(row => row.sequence_length === length && row.status === "measured");
      if (!s || !r) continue;
      output.push({
        scale: scale === "400m" ? "400M" : "1.3B",
        length,
        speedup: r.order_balanced_median_step_ms / s.order_balanced_median_step_ms,
        low: quantile(r.samples_ms, .1) / quantile(s.samples_ms, .9),
        high: quantile(r.samples_ms, .9) / quantile(s.samples_ms, .1),
        memorySaving: (1 - s.peak_allocated_bytes / r.peak_allocated_bytes) * 100,
      });
    }
  }
  return output;
}

function drawTrainingEvidence(svg, width, height, paperScale) {
  clear(svg, "完整训练加速与峰值显存", "六个完整训练形状上，SAMU 的优化器步骤更快，并使用更少峰值显存。");
  const points = trainingRows(paperScale);
  const left = 68, right = width - 18;
  const topA = 42, bottomA = 220, topB = 292, bottomB = height - 66;
  const x = point => {
    const index = points.findIndex(item => item.scale === point.scale && item.length === point.length);
    return left + (index + .5) * (right - left) / points.length;
  };
  const ySpeed = value => bottomA - (value - 1) / .4 * (bottomA - topA);
  const yMemory = value => bottomB - value / 30 * (bottomB - topB);
  frame(svg, left, topA, right, bottomA);
  horizontalGrid(svg, [1, 1.1, 1.2, 1.3, 1.4], ySpeed, left, right, value => `${value.toFixed(1)}×`);
  svg.append(node("line", {x1: left, y1: ySpeed(1), x2: right, y2: ySpeed(1), stroke: MUTED, "stroke-width": 1.4}));
  lineSeries(svg, points.map(point => ({...point, y: point.speedup})), x, ySpeed, {color: SAMU, shape: "circle", error: true});
  points.forEach(point => label(svg, `${point.speedup.toFixed(2)}×`, x(point), ySpeed(point.high) - 9, {"text-anchor": "middle", "font-size": 11}));
  label(svg, "完整优化器步骤加速", left, topA - 13, {"font-size": 12, "font-weight": 500});

  frame(svg, left, topB, right, bottomB);
  horizontalGrid(svg, [0, 10, 20, 30], yMemory, left, right, value => `${value}%`);
  const barWidth = Math.max(18, Math.min(42, (right - left) / points.length * .46));
  points.forEach(point => {
    const px = x(point);
    svg.append(node("rect", {x: px - barWidth / 2, y: yMemory(point.memorySaving), width: barWidth, height: bottomB - yMemory(point.memorySaving), fill: SAMU, opacity: .78}));
    label(svg, `${point.memorySaving.toFixed(1)}%`, px, yMemory(point.memorySaving) - 8, {"text-anchor": "middle", "font-size": 11});
    const text = node("text", {x: px, y: bottomB + 22, "text-anchor": "middle", fill: MUTED, "font-size": 11, "font-family": "Inter, Noto Sans SC, sans-serif"});
    text.append(node("tspan", {x: px, dy: 0}, point.scale));
    text.append(node("tspan", {x: px, dy: 15}, `L=${point.length / 1024}K`));
    svg.append(text);
  });
  const splitX = (x(points[2]) + x(points[3])) / 2;
  svg.append(node("line", {x1: splitX, y1: topA, x2: splitX, y2: bottomA, stroke: GRID, "stroke-width": 1, "stroke-dasharray": "3 4"}));
  svg.append(node("line", {x1: splitX, y1: topB, x2: splitX, y2: bottomB, stroke: GRID, "stroke-width": 1, "stroke-dasharray": "3 4"}));
  label(svg, "峰值已分配显存减少", left, topB - 13, {"font-size": 12, "font-weight": 500});
  axisTitle(svg, "相对 RG-LRU", 17, (topA + bottomA) / 2, true);
  axisTitle(svg, "相对 RG-LRU", 17, (topB + bottomB) / 2, true);
}

function drawScanEvidence(svg, width, height, scan) {
  clear(svg, "投影后的精确前向扫描", "相同 32 词元精确分块下，SAMU 扫描延迟低于 RG-LRU。");
  const summary = scan.training_scan?.summary || [];
  const raw = scan.training_scan?.rows || [];
  const lengths = [2048, 4096, 8192, 16384];
  const points = method => lengths.map(length => {
    const item = summary.find(row => row.method === method && row.length === length);
    const rows = raw.filter(row => row.method === method && row.length === length);
    return {
      length,
      y: item?.order_balanced_median_ms,
      low: Math.min(...rows.map(row => row.p10_ms).filter(Number.isFinite)),
      high: Math.max(...rows.map(row => row.p90_ms).filter(Number.isFinite)),
    };
  });
  const samu = points("samu_chunk32_exact"), rglru = points("rglru_chunk32_official16");
  const left = 68, right = width - 18, top = 58, bottom = height - 66;
  const x = point => left + lengths.indexOf(point.length) * (right - left) / (lengths.length - 1);
  const y = value => bottom - (Math.log10(value) + 1) / 2 * (bottom - top);
  frame(svg, left, top, right, bottom);
  horizontalGrid(svg, [.1, .2, .5, 1, 2, 5, 10], y, left, right, value => `${value} ms`);
  lengths.forEach((length, index) => label(svg, `${length / 1024}K`, x({length}), bottom + 23, {"text-anchor": "middle", fill: MUTED, "font-size": 11}));
  lineSeries(svg, samu, x, y, {color: SAMU, shape: "circle", error: true});
  lineSeries(svg, rglru, x, y, {color: RGLRU, shape: "square", error: true});
  legend(svg, [
    {label: "SAMU", color: SAMU, shape: "circle"},
    {label: "RG-LRU-16", color: RGLRU, shape: "square"},
  ], left, 25, right - left);
  samu.forEach((point, index) => {
    const ratio = rglru[index].y / point.y;
    label(svg, `${ratio.toFixed(2)}×`, x(point), y(point.y) + 20, {"text-anchor": "middle", fill: SAMU, "font-size": 11});
  });
  axisTitle(svg, "序列长度 L", (left + right) / 2, height - 15);
  axisTitle(svg, "前向扫描延迟（对数刻度）", 17, (top + bottom) / 2, true);
}

function drawDecodeEvidence(svg, width, height, inference) {
  clear(svg, "完整生成中的收益边界", "空提示和 4K 提示下，每个完整生成步的平均延迟几乎不随已生成长度改变。");
  const rows = inference.rows || [];
  const lengths = [128, 256, 512, 1024, 2048, 4096];
  const points = (architecture, prompt) => lengths.map(length => {
    const row = rows.find(item => item.architecture === architecture && item.workload === "continuous_decode_latency" && item.prompt_length === prompt && item.decode_length === length);
    return {length, y: row?.median_ms / length, low: row?.p10_ms / length, high: row?.p95_ms / length};
  });
  const series = [
    {label: "SAMU · 空提示", color: SAMU, shape: "circle", points: points("samu", 0)},
    {label: "SAMU · 4K 提示", color: SAMU, shape: "circle", dashed: true, hollow: true, points: points("samu", 4096)},
    {label: "RG-LRU · 空提示", color: RGLRU, shape: "square", points: points("rglru", 0)},
    {label: "RG-LRU · 4K 提示", color: RGLRU, shape: "square", dashed: true, hollow: true, points: points("rglru", 4096)},
  ];
  const values = series.flatMap(item => item.points.flatMap(point => [point.low, point.high])).filter(Number.isFinite);
  const minimum = Math.min(...values), maximum = Math.max(...values);
  const padding = Math.max((maximum - minimum) * .18, .01);
  const lowDomain = minimum - padding, highDomain = maximum + padding;
  const left = 68, right = width - 18, top = width < 470 ? 104 : 78, bottom = height - 66;
  const x = point => left + lengths.indexOf(point.length) * (right - left) / (lengths.length - 1);
  const y = value => bottom - (value - lowDomain) / (highDomain - lowDomain) * (bottom - top);
  const ticks = Array.from({length: 5}, (_, index) => lowDomain + index * (highDomain - lowDomain) / 4);
  frame(svg, left, top, right, bottom);
  horizontalGrid(svg, ticks, y, left, right, value => value.toFixed(3));
  lengths.forEach((length, index) => {
    if (width < 470 && ![0, 2, 4, 5].includes(index)) return;
    label(svg, length.toLocaleString(), x({length}), bottom + 23, {"text-anchor": "middle", fill: MUTED, "font-size": 11});
  });
  series.forEach(item => lineSeries(svg, item.points, x, y, {...item, error: true}));
  legend(svg, series, left, 25, right - left);
  const emptySamu = series[0].points, emptyRg = series[2].points;
  const gains = emptySamu.map((point, index) => (emptyRg[index].y / point.y - 1) * 100);
  label(svg, `SAMU 快 ${Math.min(...gains).toFixed(2)}%–${Math.max(...gains).toFixed(2)}%`, right - 4, top + 18, {"text-anchor": "end", "font-size": 11, "font-weight": 500});
  axisTitle(svg, "完整轨迹的生成词元数", (left + right) / 2, height - 15);
  axisTitle(svg, "平均单步延迟（毫秒/词元）", 17, (top + bottom) / 2, true);
}

export function renderAdvantageFigures({scan, paperScale, inference}) {
  responsive(document.querySelector("#control-scaling-chart"), 360, drawControlScaling);
  responsive(document.querySelector("#mixer-parameter-chart"), 360, (svg, width, height) => drawMixerParameters(svg, width, height, paperScale));
  responsive(document.querySelector("#training-evidence-chart"), 520, (svg, width, height) => drawTrainingEvidence(svg, width, height, paperScale));
  responsive(document.querySelector("#scan-evidence-chart"), 390, (svg, width, height) => drawScanEvidence(svg, width, height, scan));
  responsive(document.querySelector("#decode-evidence-chart"), 390, (svg, width, height) => drawDecodeEvidence(svg, width, height, inference));
}
