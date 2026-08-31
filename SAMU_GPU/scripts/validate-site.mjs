import {access, readFile} from "node:fs/promises";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const gpuRoot = join(root, "results", "gpu_optimization");
const failures = [];
const check = (condition, message) => { if (!condition) failures.push(message); };
const finite = value => Number.isFinite(Number(value));
const close = (actual, expected, tolerance = 0.002) => finite(actual) && Math.abs(Number(actual) - expected) <= tolerance;
const percentLower = (reference, candidate) => 100 * (Number(reference) - Number(candidate)) / Number(reference);
const oneDecimal = value => Number(value.toFixed(1));
const load = async name => JSON.parse(await readFile(join(gpuRoot, name), "utf8"));
const median = row => row?.forward_backward?.median_ms;
const byShape = (data, batch, length, width, architecture) => data.rows.find(row =>
  row.shape.batch === batch && row.shape.length === length && row.shape.width === width && row.architecture === architecture);

const html = await readFile(join(root, "index.html"), "utf8");
const script = await readFile(join(root, "src", "final-report.js"), "utf8");
const css = await readFile(join(root, "styles", "final-report.css"), "utf8");

const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map(match => match[1]);
const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
const internalTargets = [...html.matchAll(/href="#([^"]+)"/g)].map(match => match[1]);
check(duplicateIds.length === 0, `duplicate HTML ids: ${duplicateIds.join(", ")}`);
for (const target of internalTargets) check(ids.includes(target), `missing internal link target ${target}`);

check(/src\/final-report\.js\?v=20260831-5/.test(html), "final report module is missing or not cache-versioned");
check(/styles\/final-report\.css\?v=20260831-5/.test(html), "final report stylesheet is missing or not cache-versioned");
check(!/src\/app\.js|complete-analysis\.js|advantage-figures\.js|lessons\.js/.test(html), "legacy report modules are still mounted");
for (const target of ["conclusion", "terms", "measurements", "rglru", "backend", "memory", "sources"]) {
  check(html.includes(`href="#${target}"`), `missing navigation target ${target}`);
  check(html.includes(`id="${target}"`), `missing section id ${target}`);
}

const localAssets = [...html.matchAll(/(?:href|src)="([^"#]+)"/g)]
  .map(match => match[1])
  .filter(path => !/^(?:https?:|data:|mailto:)/.test(path))
  .map(path => path.split("?")[0]);
for (const asset of localAssets) {
  try { await access(join(root, asset)); }
  catch { failures.push(`missing local asset ${asset}`); }
}

for (const required of [
  "完整递归混合器：前向+反向",
  "快 24.6%–40.2%",
  "单个完整循环块：前向+反向",
  "快 5.8%–7.6%",
  "快 3.8%–6.5%",
  "慢 12.4%",
  "SAMU 与 RG-LRU 的 H800 训练性能对比",
  "共享特殊函数结果",
  "以 64 个分块为一组",
  "RG-LRU 的 GPU 实现",
  "按 \\(Q_c=L_c\\circ O_g\\) 求分块入口",
  "足以重放状态转移，但不足以单独恢复完整控制器导数",
  "P_{c,m}=e^{-\\nu_m G_c}",
]) check(html.includes(required), `missing scope-critical copy: ${required}`);

for (const stale of [
  /src\/app\.js/,
  /paper_scale_inference_h800\.json/,
  /完整模型生成/,
  /[“”]/,
  /帕累托/,
  /当前证据范围/,
  /结论不是/,
  /纯扫描|Hippogriff|Lingua|accelerated-scan/,
  /单一内核不可能覆盖所有形状/,
  /保留显存不是单向胜利/,
  /id="correctness"|id="lesson-6"/,
  /可以写|不能写/,
]) check(!stale.test(html), `stale or unsupported visible claim remains: ${stale}`);

check(/--viz-series-1/.test(css) && /\.paper-figure/.test(css) && /\.scientific-chart/.test(css), "scientific report styling is incomplete");
check(/MathJax/.test(html) && /tex-svg\.js/.test(html), "MathJax is not configured");
check((html.match(/architecture-figure/g) || []).length === 1, "the report must contain exactly one architecture figure");
check(/\.scope-split\s*\{[^}]*grid-template-columns:\s*1fr/.test(css), "wide block tables are not stacked vertically");
check(/<title>完整递归混合器前向加反向延迟<\/title>/.test(script), "mixer chart lacks an accessible Chinese title");
check(/<title>D=1024 时的序列长度转折<\/title>/.test(script), "length chart lacks an accessible Chinese title");
check(/<title>L=32768 时的状态宽度扩展<\/title>/.test(script), "width chart lacks an accessible Chinese title");
check((html.match(/\\\[/g) || []).length === (html.match(/\\\]/g) || []).length, "unbalanced display-math delimiters");
check((html.match(/\\\(/g) || []).length === (html.match(/\\\)/g) || []).length, "unbalanced inline-math delimiters");
check(/minimum_ms/.test(script) && /maximum_ms/.test(script), "figures do not render measured dispersion");
check(/selected_dispatch_very_long_hybrid_h800_v2\.json/.test(script), "very-long chart is not bound to the strict hybrid result");
check(/group\.shape\.length === 32768/.test(script), "length chart may be using rejected 65K/131K full-group rows");

const [mixer, counterexample, length, veryLong, width, block, optimizer, launches, fattori,
  hybrid65, hybrid131] = await Promise.all([
  load("selected_dispatch_grouped_k32_h800.json"),
  load("selected_dispatch_l8192_d1024_serial_k32_h800.json"),
  load("selected_dispatch_length_scaling_grouped_k32_h800.json"),
  load("selected_dispatch_very_long_hybrid_h800_v2.json"),
  load("selected_dispatch_width_scaling_extra_grouped_k32_h800.json"),
  load("block_dispatch_grouped_k32_h800.json"),
  load("optimizer_step_grouped_k32_h800.json"),
  load("selected_dispatch_launches_grouped_k32_h800.json"),
  load("public_fattori_h800.json"),
  load("samu_hybrid_prefix64_65536_d1024_correctness_v2.json"),
  load("samu_hybrid_prefix64_131072_d1024_correctness_v2.json"),
]);

const mixerExpected = [
  [4, 2048, 2048, 2.553, 1.924, 24.6, 22.0, 50, 70],
  [1, 8192, 2560, 3.130, 2.265, 27.6, 22.0, 50, 74],
  [1, 32768, 1024, 4.853, 3.041, 37.3, 23.4, 51, 73],
  [1, 32768, 2048, 8.763, 5.238, 40.2, 23.5, 51, 73],
];
for (const [batch, sequence, state, rgExpected, samuExpected, speedExpected, memoryExpected, rgLaunches, samuLaunches] of mixerExpected) {
  const rg = byShape(mixer, batch, sequence, state, "rglru");
  const samu = byShape(mixer, batch, sequence, state, "samu");
  check(close(median(rg), rgExpected) && close(median(samu), samuExpected), `mixer mismatch B${batch}/L${sequence}/D${state}`);
  check(oneDecimal(percentLower(median(rg), median(samu))) === speedExpected, `mixer speedup mismatch B${batch}/L${sequence}/D${state}`);
  check(oneDecimal(percentLower(rg?.peak_allocated_delta_bytes, samu?.peak_allocated_delta_bytes)) === memoryExpected,
    `mixer allocated-memory reduction mismatch B${batch}/L${sequence}/D${state}`);
  check(rg?.forward_backward?.samples_ms?.length === 10 && samu?.forward_backward?.samples_ms?.length === 10,
    `mixer row does not retain ten timed samples B${batch}/L${sequence}/D${state}`);
  const launchRg = byShape(launches, batch, sequence, state, "rglru");
  const launchSamu = byShape(launches, batch, sequence, state, "samu");
  check(launchRg?.cuda_kernel_events === rgLaunches && launchSamu?.cuda_kernel_events === samuLaunches,
    `launch count mismatch B${batch}/L${sequence}/D${state}`);
}

const counterRg = byShape(counterexample, 1, 8192, 1024, "rglru");
const counterSamu = byShape(counterexample, 1, 8192, 1024, "samu");
check(close(median(counterRg), 1.603) && close(median(counterSamu), 1.801), "8K/D1024 counterexample mismatch");
check(oneDecimal(percentLower(median(counterRg), median(counterSamu))) === -12.4, "8K/D1024 counterexample percentage mismatch");

const length32Rg = byShape(length, 1, 32768, 1024, "rglru");
const length32Samu = byShape(length, 1, 32768, 1024, "samu");
check(close(median(length32Rg), 4.844) && close(median(length32Samu), 3.051), "32K length-scaling mismatch");
for (const [sequence, rgExpected, samuExpected] of [[65536, 9.154, 6.666], [131072, 17.712, 13.163]]) {
  const rg = byShape(veryLong, 1, sequence, 1024, "rglru");
  const samu = byShape(veryLong, 1, sequence, 1024, "samu");
  check(close(median(rg), rgExpected) && close(median(samu), samuExpected), `strict very-long mismatch L${sequence}`);
}
check(oneDecimal(percentLower(median(byShape(veryLong, 1, 65536, 1024, "rglru")), median(byShape(veryLong, 1, 65536, 1024, "samu")))) === 27.2,
  "65K strict-dispatch percentage mismatch");
check(oneDecimal(percentLower(median(byShape(veryLong, 1, 131072, 1024, "rglru")), median(byShape(veryLong, 1, 131072, 1024, "samu")))) === 25.7,
  "131K strict-dispatch percentage mismatch");

for (const [state, rgExpected, samuExpected] of [[1536, 6.969, 4.123], [2560, 11.073, 6.637]]) {
  const rg = byShape(width, 1, 32768, state, "rglru");
  const samu = byShape(width, 1, 32768, state, "samu");
  check(close(median(rg), rgExpected) && close(median(samu), samuExpected), `width scaling mismatch D${state}`);
}

for (const [caseName, rgExpected, samuExpected] of [
  ["state2048_short", 15.764, 14.699],
  ["state2560_medium", 20.746, 19.544],
  ["state1024_long", 24.120, 22.276],
  ["400m_block_long", 43.202, 39.949],
]) {
  const item = block.cases.find(row => row.case === caseName);
  const rg = item?.rows?.find(row => row.architecture === "rglru");
  const samu = item?.rows?.find(row => row.architecture === "samu");
  check(close(median(rg), rgExpected) && close(median(samu), samuExpected), `one-block mismatch ${caseName}`);
}
check(JSON.stringify(block.cases.map(item => {
  const rg = item.rows.find(row => row.architecture === "rglru");
  const samu = item.rows.find(row => row.architecture === "samu");
  return oneDecimal(percentLower(median(rg), median(samu)));
})) === JSON.stringify([6.8, 5.8, 7.6, 7.5]), "one-block speedup vector mismatch");

for (const [caseName, rgExpected, samuExpected] of [
  ["state2048_short", 24.808, 23.728],
  ["state2560_medium", 34.394, 33.101],
  ["400m_block_long", 48.997, 45.796],
]) {
  const item = optimizer.cases.find(row => row.case === caseName);
  const rg = item?.rows?.find(row => row.architecture === "rglru");
  const samu = item?.rows?.find(row => row.architecture === "samu");
  check(close(rg?.optimizer_step?.median_ms, rgExpected) && close(samu?.optimizer_step?.median_ms, samuExpected), `one-block optimizer mismatch ${caseName}`);
}
check(JSON.stringify(optimizer.cases.map(item => {
  const rg = item.rows.find(row => row.architecture === "rglru");
  const samu = item.rows.find(row => row.architecture === "samu");
  return oneDecimal(percentLower(rg.optimizer_step.median_ms, samu.optimizer_step.median_ms));
})) === JSON.stringify([4.4, 3.8, 6.5]), "one-block optimizer speedup vector mismatch");

for (const [state, samuParameters, rgParameters] of [[1024, 3076, 134144], [2048, 6148, 530432], [2560, 7684, 826880]]) {
  const source = state === 2560 ? byShape(mixer, 1, 8192, state, "samu") : byShape(mixer, 1, 32768, state, "samu");
  const reference = state === 2560 ? byShape(mixer, 1, 8192, state, "rglru") : byShape(mixer, 1, 32768, state, "rglru");
  check(source?.parameters === samuParameters && reference?.parameters === rgParameters, `mixer parameter count mismatch D${state}`);
}

check(hybrid65.passed && hybrid131.passed, "strict very-long hybrid correctness gate did not pass");
check(hybrid65.rows?.[0]?.output?.relative_l2 === 0 && hybrid131.rows?.[0]?.output?.relative_l2 === 0,
  "strict very-long hybrid output is not bitwise identical");

const fattoriOriginal = fattori.rows.find(row => row.implementation === "fattori_original");
const oursRestricted = fattori.rows.find(row => row.implementation === "ours_restricted");
check(fattori.source_diff_empty && close(median(fattoriOriginal), 4.535) && close(median(oursRestricted), 2.260), "Fattori restricted reproduction mismatch");

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(JSON.stringify({
  reportVersion: "2026-08-31-5",
  sections: 7,
  localAssets: localAssets.length,
  primaryMixerRows: mixer.rows.length,
  mixerSamplesPerRow: 10,
  oneBlockCases: block.cases.length,
  oneBlockOptimizerCases: optimizer.cases.length,
  strictVeryLongRows: veryLong.rows.length,
  architectureFigures: 1,
  wideTablesStacked: true,
}, null, 2));
