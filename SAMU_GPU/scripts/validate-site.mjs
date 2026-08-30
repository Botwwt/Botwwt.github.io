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

check(/src\/final-report\.js\?v=20260831-2/.test(html), "final report module is missing or not cache-versioned");
check(/styles\/final-report\.css\?v=20260831-2/.test(html), "final report stylesheet is missing or not cache-versioned");
check(!/src\/app\.js|complete-analysis\.js|advantage-figures\.js|lessons\.js/.test(html), "legacy report modules are still mounted");
for (const target of ["conclusion", "measurements", "backend", "memory", "rglru", "correctness", "lesson-6", "sources"]) {
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
  "24.6%–40.2% faster",
  "5.8%–7.6% faster",
  "3.8%–6.5% faster",
  "12.4% slower",
  "单个完整 recurrent block",
  "不是 12/24 层完整语言模型 optimizer step",
  "grouped” 只表示 chunk summaries 的执行分组",
  "足以重放 transition，但不足以独立恢复完整 controller derivative",
  "P<sub>c,m</sub>=exp(−ν<sub>m</sub>G<sub>c</sub>)",
  "pure scan-only 最快的是 accelerated-scan/Hippogriff",
  "当前已证明 400M/1.3B 完整模型 optimizer step 更快",
]) check(html.includes(required), `missing scope-critical copy: ${required}`);

for (const stale of [
  /src\/app\.js/,
  /paper_scale_inference_h800\.json/,
  /完整模型生成/,
  /400M\s*\/\s*1\.3B 多层完整模型训练与生成属于当前/,
  /65K\/131K 仍有约 43% 优势[^<]*可以/,
  /SAMU 在所有形状都比 RG-LRU 快[^<]*证明/,
]) check(!stale.test(html), `stale or unsupported visible claim remains: ${stale}`);

check(/--viz-series-1/.test(css) && /\.paper-figure/.test(css) && /\.scientific-chart/.test(css), "scientific report styling is incomplete");
check(/<title>complete mixer F\+B latency<\/title>/.test(script), "mixer chart lacks an accessible title");
check(/<title>D=1024 length crossover<\/title>/.test(script), "length chart lacks an accessible title");
check(/<title>L=32768 width scaling<\/title>/.test(script), "width chart lacks an accessible title");
check(/minimum_ms/.test(script) && /maximum_ms/.test(script), "figures do not render measured dispersion");
check(/selected_dispatch_very_long_hybrid_h800_v2\.json/.test(script), "very-long chart is not bound to the strict hybrid result");
check(/group\.shape\.length === 32768/.test(script), "length chart may be using rejected 65K/131K full-group rows");

const [mixer, counterexample, length, veryLong, width, block, optimizer, launches, fattori, scanOnly,
  hybrid65, hybrid131, rejectedLong, compressed] = await Promise.all([
  load("selected_dispatch_grouped_k32_h800.json"),
  load("selected_dispatch_l8192_d1024_serial_k32_h800.json"),
  load("selected_dispatch_length_scaling_grouped_k32_h800.json"),
  load("selected_dispatch_very_long_hybrid_h800_v2.json"),
  load("selected_dispatch_width_scaling_extra_grouped_k32_h800.json"),
  load("block_dispatch_grouped_k32_h800.json"),
  load("optimizer_step_grouped_k32_h800.json"),
  load("selected_dispatch_launches_grouped_k32_h800.json"),
  load("public_fattori_h800.json"),
  load("public_accelerated_scan_h800.json"),
  load("samu_hybrid_prefix64_65536_d1024_correctness_v2.json"),
  load("samu_hybrid_prefix64_131072_d1024_correctness_v2.json"),
  load("samu_grouped_k32_long_correctness.json"),
  load("samu_compressed_grouped_k32_correctness.json"),
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
check(rejectedLong.passed === false, "rejected full-group very-long result is unexpectedly marked passed");
check(compressed.passed === false, "compressed grouped interaction is unexpectedly marked passed");
check(close(compressed.rows?.[1]?.output?.relative_l2, 6.375004431902198e-6, 1e-12), "compressed grouped rejection metric mismatch");

const fattoriOriginal = fattori.rows.find(row => row.implementation === "fattori_original");
const oursRestricted = fattori.rows.find(row => row.implementation === "ours_restricted");
check(fattori.source_diff_empty && close(median(fattoriOriginal), 4.535) && close(median(oursRestricted), 2.260), "Fattori restricted reproduction mismatch");
const hippogriff = scanOnly.rows.find(row => row.implementation === "hippogriff_accelerated_scan");
const lingua = scanOnly.rows.find(row => row.implementation === "lingua_original_wrapper");
const oursChunk32 = scanOnly.rows.find(row => row.implementation === "ours_materialized_chunk32");
check(close(median(hippogriff), 0.710) && close(median(lingua), 0.736) && close(median(oursChunk32), 0.858), "public scan-only reproduction mismatch");

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(JSON.stringify({
  reportVersion: "2026-08-31-2",
  sections: 8,
  localAssets: localAssets.length,
  primaryMixerRows: mixer.rows.length,
  mixerSamplesPerRow: 10,
  oneBlockCases: block.cases.length,
  oneBlockOptimizerCases: optimizer.cases.length,
  strictVeryLongRows: veryLong.rows.length,
  rejectedCandidatesRetained: ["full-group very-long", "compressed grouped K32"],
}, null, 2));
