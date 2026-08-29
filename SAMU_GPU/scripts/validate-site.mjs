import {readFile, access, readdir} from "node:fs/promises";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {lessons} from "../src/lessons.js";

const root=resolve(dirname(fileURLToPath(import.meta.url)),"..");
const html=await readFile(join(root,"index.html"),"utf8");
const failures=[];
if(lessons.length!==24)failures.push(`expected 24 report sections, found ${lessons.length}`);
for(const required of ["report","gpu-explorer","benchmarks","advantage","implementation","sources"]){
  if(!html.includes(`href="#${required}"`))failures.push(`missing navigation target ${required}`);
}
const localAssets=[...html.matchAll(/(?:href|src)="([^"#]+)"/g)].map(m=>m[1]).filter(x=>!x.startsWith("http")&&!x.startsWith("data:"));
for(const asset of localAssets){try{await access(join(root,asset));}catch{failures.push(`missing local asset ${asset}`);}}
const resultRoot=join(root,"benchmark_results_equal_kernel");
const summary=JSON.parse(await readFile(join(resultRoot,"summary.json"),"utf8"));
const raw=await readdir(join(resultRoot,"raw"));
if(!summary.generated_from_raw)failures.push("summary is not marked generated_from_raw");
if(!String(summary.environment?.gpu||"").includes("H800"))failures.push(`microbenchmark GPU is not H800: ${summary.environment?.gpu}`);
if(JSON.stringify(summary.model_orders)!==JSON.stringify(["rglru,samu","samu,rglru"]))failures.push(`microbenchmark execution orders are not counterbalanced: ${JSON.stringify(summary.model_orders)}`);
if(!summary.equation_audit?.passed)failures.push("official-equation audit did not pass");
if(summary.equation_audit?.rglru_reference_commit!=="2efa84dac0e68e63547a27a18fa943c98f1c312e")failures.push("unexpected RecurrentGemma source commit");
if(summary.rows.length!==raw.filter(x=>x.endsWith(".json")).length)failures.push(`summary/raw mismatch ${summary.rows.length}/${raw.length}`);
const measured=summary.rows.filter(r=>r.status==="measured");
if(!measured.length)failures.push("no measured benchmark rows");
if(summary.rows.some(row=>!['samu','rglru'].includes(row.model)))failures.push("public summary contains a non-SAMU/RG-LRU row");
for(const row of measured){
  for(const key of ["median_ms","p10_ms","p95_ms","raw_samples_ms","samples","requested_warmup_ms","requested_rep_ms","source_commit"]){
    if(row[key]===undefined||row[key]===null)failures.push(`${row.id} missing ${key}`);
  }
  if(row.samples!==row.raw_samples_ms.length)failures.push(`${row.id} sample count mismatch ${row.samples}/${row.raw_samples_ms.length}`);
}
const system=JSON.parse(await readFile(join(root,"benchmark_results_griffin_section5","summary.json"),"utf8"));
if(system.status!=="complete"||!system.correctness?.passed)failures.push("system experiment is incomplete or failed correctness");
if(!String(system.environment?.gpu||"").includes("H800"))failures.push(`system GPU is not H800: ${system.environment?.gpu}`);
const systemOrders=(system.configuration?.counterbalanced_runs||[]).map(run=>run.model_order).sort();
if(JSON.stringify(systemOrders)!==JSON.stringify(["rglru,samu","samu,rglru"]))failures.push(`system execution orders are not counterbalanced: ${JSON.stringify(systemOrders)}`);
const fullWinners=(system.rows||[]).filter(row=>row.workload==="maximum_throughput");
if(!fullWinners.length||fullWinners.some(row=>row.batch>128))failures.push("bounded full-trajectory throughput winners are missing or exceed B=128");
for(const model of ["samu","rglru"]){
  for(const length of [512,1024,2048,4096]){
    if(!fullWinners.some(row=>row.model===model&&row.decode_length===length))failures.push(`missing ${model} full-trajectory winner at length ${length}`);
  }
}
const visibleText=[html,await readFile(join(root,"README.md"),"utf8"),await readFile(join(root,"BENCHMARK_METHODOLOGY.md"),"utf8"),await readFile(join(root,"src","lessons.js"),"utf8")].join("\n");
if(/mamba/i.test(visibleText))failures.push("visible report still mentions Mamba");
if(/RTX 3090|3090 实测/i.test(visibleText))failures.push("visible report still labels the experiment as RTX 3090");
if(failures.length){console.error(failures.join("\n"));process.exit(1);}
console.log(JSON.stringify({sections:lessons.length,assets:localAssets.length,microRows:summary.rows.length,microMeasured:measured.length,systemRows:system.rows.length,fullTrajectoryWinners:fullWinners.length},null,2));
