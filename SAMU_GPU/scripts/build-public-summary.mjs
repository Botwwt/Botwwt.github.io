import {readFile, writeFile} from "node:fs/promises";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const resultRoot = join(root, "benchmark_results_griffin_complete");
const readJson = async name => JSON.parse(await readFile(join(resultRoot, name), "utf8"));

const [training, adaptation] = await Promise.all([
  readJson("training_on_device.json"),
  readJson("samu_gpu_adaptation.json"),
]);

const summary = {
  schema_version: 1,
  generated_from: ["training_on_device.json", "samu_gpu_adaptation.json"],
  training: {
    generated_utc: training.generated_utc,
    environment: training.environment,
    paper_mapping: training.paper_mapping,
    static_input_scalars_per_token: training.static_input_scalars_per_token,
    summary: training.summary,
  },
  adaptation: {
    generated_utc: adaptation.generated_utc,
    environment: adaptation.environment,
    scope: adaptation.scope,
    status: adaptation.status,
    counts: adaptation.counts,
    source_derived_special_functions_per_token: adaptation.source_derived_special_functions_per_token,
    correctness: adaptation.correctness,
    comparisons: adaptation.comparisons,
    compiler_resources: adaptation.compiler_resources,
    hardware_counter_policy: adaptation.hardware_counter_policy,
  },
};

await writeFile(
  join(resultRoot, "public_summary.json"),
  `${JSON.stringify(summary, null, 2)}\n`,
  "utf8",
);

console.log(JSON.stringify({
  output: join(resultRoot, "public_summary.json"),
  trainingRows: summary.training.summary.length,
  adaptationRows: summary.adaptation.comparisons.length,
}, null, 2));
