import {access, readFile} from "node:fs/promises";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {lessons} from "../src/lessons.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const requiredCommit = "2efa84dac0e68e63547a27a18fa943c98f1c312e";
const html = await readFile(join(root, "index.html"), "utf8");
if (!/src\/app\.js\?v=20260830-3/.test(html)) failures.push("entry module is not cache-versioned");
if (!/styles\/course\.css\?v=20260830-3/.test(html)) failures.push("report stylesheet is not cache-versioned");
const load = async name => JSON.parse(await readFile(join(root, "benchmark_results_small_model", name), "utf8"));
const finite = value => Number.isFinite(Number(value));

if (lessons.length !== 16) failures.push(`expected 16 detailed report sections, found ${lessons.length}`);
for (const target of ["summary", "evidence", "training", "inference", "report", "sources"]) {
  if (!html.includes(`href="#${target}"`)) failures.push(`missing navigation target ${target}`);
  if (!html.includes(`id="${target}"`)) failures.push(`missing section id ${target}`);
}

const localAssets = [...html.matchAll(/(?:href|src)="([^"#]+)"/g)]
  .map(match => match[1])
  .filter(path => !path.startsWith("http") && !path.startsWith("data:"))
  .map(path => path.split("?")[0]);
for (const asset of localAssets) {
  try { await access(join(root, asset)); }
  catch { failures.push(`missing local asset ${asset}`); }
}

const appModule = await readFile(join(root, "src", "app.js"), "utf8");
const analysisModule = await readFile(join(root, "src", "complete-analysis.js"), "utf8");
if (!/complete-analysis\.js\?v=20260830-3/.test(appModule)) failures.push("analysis module is not cache-versioned");
if (!/advantage-figures\.js\?v=20260830-3/.test(analysisModule)) failures.push("figure module is not cache-versioned");

const [environment, correctness, officialBlock, equationAudit, deviceScan,
  backendAblation, paperScale, inference, roofline] = await Promise.all([
  load("environment.json"), load("correctness.json"), load("official_hawk_block_audit.json"),
  load("training_equation_audit.json"),
  load("training_on_device_complete.json"), load("paper_scale_backend_ablation.json"),
  load("paper_scale_h800.json"), load("paper_scale_inference_h800.json"),
  load("h800_roofline_decode.json"),
]);

if (!String(environment.gpu || "").includes("H800")) failures.push(`unexpected GPU ${environment.gpu}`);
if (environment.official_recurrentgemma_commit !== requiredCommit) failures.push("environment uses the wrong RecurrentGemma commit");
if (!officialBlock.passed) failures.push("official Hawk block audit failed");
if (officialBlock.official_commit !== requiredCommit) failures.push("official Hawk block audit uses the wrong commit");
if (!officialBlock.decode_consistency?.passed) failures.push("full-sequence versus token-decode audit failed");
for (const [architecture, backend] of [["samu", "packed"], ["rglru", "fused"], ["rglru", "bmm"]]) {
  if (!(officialBlock.decode_consistency?.rows || []).some(row => row.architecture === architecture && row.decode_backend === backend && row.passed)) {
    failures.push(`missing passing decode audit ${architecture}/${backend}`);
  }
}
if (!equationAudit.passed || !equationAudit.official_rglru_training?.passed) failures.push("official RG-LRU equation/training audit failed");
if (equationAudit.official_rglru_training?.reference_commit !== requiredCommit) failures.push("RG-LRU training audit uses the wrong commit");
for (const [name, row] of Object.entries(correctness)) {
  if (!["output_max_abs", "gradient_max_abs", "grad_a_max_abs", "state_max_abs"].some(key => finite(row[key]))) {
    failures.push(`correctness row ${name} has no numerical error`);
  }
}
for (const name of ["samu_decode_bfloat16", "rglru_decode_bfloat16", "rglru_decode_bmm_bfloat16"]) {
  if (!correctness[name] || !finite(correctness[name].output_max_abs) || !finite(correctness[name].state_max_abs)) {
    failures.push(`missing one-step decode correctness row ${name}`);
  }
}

for (const method of [
  "samu_framework_linear_reference", "samu_linear", "samu_chunk32",
  "samu_associative_bf16_reference", "samu_associative_fp32_reference",
  "rglru_framework_linear_reference", "rglru_linear", "rglru_chunk32",
  "rglru_associative_bf16_reference", "rglru_associative_fp32_reference",
]) {
  for (const length of [2048, 4096, 8192, 16384]) {
    if (!(deviceScan.summary || []).some(row => row.method === method && row.length === length)) {
      failures.push(`missing device-scan ${method} at L=${length}`);
    }
  }
}

for (const scale of ["400m", "1.3b"]) {
  for (const architecture of ["samu", "rglru"]) {
    for (const backend of ["triton", "framework_eager", "associative_bf16", "associative_fp32", "chunk16", "chunk32"]) {
      if (!(backendAblation.rows || []).some(row => row.scale === scale && row.architecture === architecture && row.backend === backend)) {
        failures.push(`missing backend ablation ${scale}/${architecture}/${backend}`);
      }
    }
    const record = (paperScale.records || []).find(row => row.scale === scale && row.architecture === architecture);
    if (!record) failures.push(`missing paper-scale record ${scale}/${architecture}`);
    for (const length of [2048, 4096, 8192]) {
      const row = record?.rows?.find(item => item.sequence_length === length);
      if (row?.status !== "measured" || !finite(row.order_balanced_median_step_ms)) {
        failures.push(`missing measured paper-scale step ${scale}/${architecture}/L${length}`);
      }
    }
  }
}
if (paperScale.official_recurrentgemma_commit !== requiredCommit) failures.push("paper-scale training uses the wrong commit");

if (inference.status !== "complete") failures.push(`paper-scale inference is not complete: ${inference.status}`);
if (inference.official_recurrentgemma_commit !== requiredCommit) failures.push("paper-scale inference uses the wrong commit");
if ((inference.measurement_passes || []).length !== 4) failures.push(`expected four pass/architecture inference records, found ${(inference.measurement_passes || []).length}`);
for (const passName of ["AB", "BA"]) {
  for (const architecture of ["samu", "rglru"]) {
    const pass = (inference.measurement_passes || []).find(row => row.measurement_pass === passName && row.architecture === architecture);
    if (!pass) failures.push(`missing inference pass ${passName}/${architecture}`);
    if ((pass?.batch_search || []).length !== 11) failures.push(`${passName}/${architecture} does not contain all eleven batch probes`);
  }
}
for (const architecture of ["samu", "rglru"]) {
  for (const prompt of [0, 4096]) for (const length of [128, 256, 512, 1024, 2048, 4096]) {
    const row = (inference.rows || []).find(item => item.architecture === architecture && item.workload === "continuous_decode_latency" && item.prompt_length === prompt && item.decode_length === length);
    if (!row || row.measurement_passes !== 2 || !finite(row.median_ms)) failures.push(`missing counterbalanced latency ${architecture}/P${prompt}/L${length}`);
  }
  for (const length of [512, 1024, 2048, 4096]) {
    const row = (inference.rows || []).find(item => item.architecture === architecture && item.workload === "maximum_throughput" && item.decode_length === length);
    if (!row || !finite(row.tokens_per_second)) failures.push(`missing completed maximum-throughput trajectory ${architecture}/L${length}`);
  }
}

if (!finite(roofline.measured_copy_bandwidth?.median_gb_per_second)) failures.push("missing measured H800 copy bandwidth");
if (!finite(roofline.measured_bf16_gemm?.median_tflops_per_second)) failures.push("missing measured H800 BF16 GEMM throughput");
if (!finite(roofline.measured_roofline?.ridge_flops_per_byte)) failures.push("missing measured H800 roofline ridge point");
for (const architecture of ["samu", "rglru"]) {
  if (!(roofline.paper_scale_decode_accounting?.[architecture]?.fixed_batch_latency || []).length) failures.push(`missing decode byte accounting for ${architecture}`);
}

const visibleText = [html, await readFile(join(root, "README.md"), "utf8"),
  await readFile(join(root, "BENCHMARK_METHODOLOGY.md"), "utf8"),
  await readFile(join(root, "GRIFFIN_PROTOCOL_STATUS.md"), "utf8"),
  await readFile(join(root, "src", "lessons.js"), "utf8"),
  await readFile(join(root, "src", "complete-analysis.js"), "utf8")].join("\n");
if (/mamba/i.test(visibleText)) failures.push("visible report still mentions Mamba");
if (/RTX 3090|3090 实测/i.test(visibleText)) failures.push("visible report still labels the experiment as RTX 3090");
if (/验证集训练曲线|测试集 BPB|enwik8 三随机种子|训练质量/.test(visibleText)) failures.push("visible report still contains the removed quality-training track");
if (/7B|7b|容量边界|显存不足/.test(visibleText)) failures.push("visible report still contains the removed large-model capacity track");
if (/旧微基准中的额外大投影|先通过官方方程与整块实现审计|这里统计逻辑值的数量/.test(visibleText)) failures.push("visible report still contains user-removed copy");
if (/id="correctness-audit"/.test(html)) failures.push("removed correctness audit block is still mounted");

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(JSON.stringify({
  sections: lessons.length,
  assets: localAssets.length,
  deviceScanRows: deviceScan.summary.length,
  backendRows: backendAblation.rows.length,
  paperScaleRecords: paperScale.records.filter(record => ["400m", "1.3b"].includes(record.scale)).length,
  inferencePasses: inference.measurement_passes.length,
  inferenceRows: inference.rows.length,
}, null, 2));
