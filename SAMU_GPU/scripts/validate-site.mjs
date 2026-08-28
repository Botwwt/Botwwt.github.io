import {readFile, access, readdir} from "node:fs/promises";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {lessons} from "../src/lessons.js";

const root=resolve(dirname(fileURLToPath(import.meta.url)),"..");
const html=await readFile(join(root,"index.html"),"utf8");
const failures=[];
if(lessons.length!==24)failures.push(`expected 24 lessons, found ${lessons.length}`);
for(const required of ["course","gpu-explorer","benchmarks","compare","implementation","sources"]){
  if(!html.includes(`href="#${required}"`))failures.push(`missing navigation target ${required}`);
}
const localAssets=[...html.matchAll(/(?:href|src)="([^"#]+)"/g)].map(m=>m[1]).filter(x=>!x.startsWith("http")&&!x.startsWith("data:"));
for(const asset of localAssets){try{await access(join(root,asset));}catch{failures.push(`missing local asset ${asset}`);}}
const summary=JSON.parse(await readFile(join(root,"benchmark_results","summary.json"),"utf8"));
const raw=await readdir(join(root,"benchmark_results","raw"));
if(!summary.generated_from_raw)failures.push("summary is not marked generated_from_raw");
if(summary.rows.length!==raw.filter(x=>x.endsWith(".json")).length)failures.push(`summary/raw mismatch ${summary.rows.length}/${raw.length}`);
const measured=summary.rows.filter(r=>r.status==="measured");
if(!measured.length)failures.push("no measured benchmark rows");
for(const row of measured){for(const key of ["median_ms","p10_ms","p95_ms","raw_samples_ms","source_commit"]){if(row[key]===undefined||row[key]===null)failures.push(`${row.id} missing ${key}`);}}
if(failures.length){console.error(failures.join("\n"));process.exit(1);}
console.log(JSON.stringify({lessons:lessons.length,assets:localAssets.length,rows:summary.rows.length,measured:measured.length,unsupported:summary.rows.filter(r=>r.status==="unsupported").length},null,2));
