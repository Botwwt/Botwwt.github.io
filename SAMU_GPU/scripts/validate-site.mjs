import {readFile, access, readdir} from "node:fs/promises";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {lessons} from "../src/lessons.js";

const root=resolve(dirname(fileURLToPath(import.meta.url)),"..");
const html=await readFile(join(root,"index.html"),"utf8");
const failures=[];
if(lessons.length!==24)failures.push(`expected 24 report sections, found ${lessons.length}`);
for(const required of ["summary","training","adaptation","section5","report","sources"]){
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
const completeRoot=join(root,"benchmark_results_griffin_complete");
const adaptation=JSON.parse(await readFile(join(completeRoot,"samu_gpu_adaptation.json"),"utf8"));
if(!String(adaptation.environment?.gpu||"").includes("H800"))failures.push(`adaptation GPU is not H800: ${adaptation.environment?.gpu}`);
if(!Object.values(adaptation.correctness||{}).every(audit=>audit.passed))failures.push("one or more SAMU adaptation correctness audits failed");
for(const batch of [1,4,16,64,128]){
  const row=(adaptation.comparisons||[]).find(item=>item.model==="grouped16_direct"&&item.batch===batch);
  if(!row||row.rglru_over_samu<=1)failures.push(`SAMU direct does not win the measured D2048 batch ${batch}`);
}
if(adaptation.hardware_counter_policy?.dram_l2_sfu_achieved_occupancy!==null)failures.push("blocked hardware counters must remain null");

const training=JSON.parse(await readFile(join(completeRoot,"training_on_device.json"),"utf8"));
if(training.paper_mapping?.timed_scope!=="post-projection forward scan/gate stage")failures.push("training scan timing scope is not explicit");
for(const method of ["samu_linear","rglru_linear","samu_chunk32","rglru_chunk32"]){
  for(const length of [2048,4096,8192,16384]){
    if(!(training.summary||[]).some(row=>row.method===method&&row.length===length))failures.push(`missing ${method} at length ${length}`);
  }
}

const system=JSON.parse(await readFile(join(completeRoot,"griffin_candidate_inference.json"),"utf8"));
if(!system.correctness?.passed)failures.push("candidate system correctness failed");
if(!String(system.environment?.gpu||"").includes("H800"))failures.push(`candidate system GPU is not H800: ${system.environment?.gpu}`);
if(system.environment?.rglru_commit!=="2efa84dac0e68e63547a27a18fa943c98f1c312e")failures.push("candidate system uses the wrong RecurrentGemma commit");
if(!String(system.fairness?.shared||"").includes("exact same embedding"))failures.push("candidate models do not declare exact shared shell tensors");
if(system.paper_protocol?.latency_repeats_per_order!==2||system.paper_protocol?.throughput_repeats_per_order!==2)failures.push("candidate system does not retain two repeats per order");
const systemRows=system.rows||[];
for(const model of ["samu_grouped16_direct","rglru_official16"]){
  for(const prompt of [0,4096])for(const order of ["AB","BA"])for(const length of [128,256,512,1024,2048,4096]){
    if(!systemRows.some(row=>row.model===model&&row.workload==="continuous_decode_latency"&&row.prompt_length===prompt&&row.measurement_order===order&&row.decode_length===length))failures.push(`missing latency ${model} prompt ${prompt} order ${order} length ${length}`);
  }
  for(const length of [512,1024,2048,4096]){
    if(!systemRows.some(row=>row.model===model&&row.workload==="bounded_maximum_throughput_candidate"&&row.decode_length===length))failures.push(`missing full trajectory ${model} length ${length}`);
  }
}
const forward=JSON.parse(await readFile(join(completeRoot,"griffin_full_forward_proxy.json"),"utf8"));
if(!forward.correctness?.passed)failures.push("full-forward proxy correctness failed");
if(forward.paper_mapping?.repeats_per_order!==5)failures.push("full-forward proxy does not retain five repeats per order");
const forwardRows=forward.rows||[];
for(const model of ["samu_grouped16_direct","rglru_official16"]){
  for(const order of ["AB","BA"])for(const length of [2048,4096,8192,16384]){
    const row=forwardRows.find(item=>item.model===model&&item.measurement_order===order&&item.sequence_length===length);
    if(!row)failures.push(`missing full forward ${model} order ${order} length ${length}`);
    else if(row.raw_samples_ms?.length!==5)failures.push(`full forward ${model} order ${order} length ${length} has the wrong sample count`);
  }
}
const visibleText=[html,await readFile(join(root,"README.md"),"utf8"),await readFile(join(root,"BENCHMARK_METHODOLOGY.md"),"utf8"),await readFile(join(root,"src","lessons.js"),"utf8")].join("\n");
if(/mamba/i.test(visibleText))failures.push("visible report still mentions Mamba");
if(/RTX 3090|3090 实测/i.test(visibleText))failures.push("visible report still labels the experiment as RTX 3090");
if(failures.length){console.error(failures.join("\n"));process.exit(1);}
console.log(JSON.stringify({sections:lessons.length,assets:localAssets.length,microRows:summary.rows.length,microMeasured:measured.length,adaptationRows:adaptation.final_rows.length,trainingRows:training.rows.length,forwardRows:forwardRows.length,systemRows:systemRows.length},null,2));
