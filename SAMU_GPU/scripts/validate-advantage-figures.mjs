import {readFile} from "node:fs/promises";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {renderAdvantageFigures} from "../src/advantage-figures.js";

class FakeSvgNode {
  constructor(tagName, id = "") {
    this.tagName = tagName;
    this.id = id;
    this.children = [];
    this.attributes = new Map();
    this.textContent = "";
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = [...children];
  }

  getBoundingClientRect() {
    return {width: this.id === "training-evidence-chart" ? 1180 : 650};
  }
}

const ids = [
  "control-scaling-chart",
  "mixer-parameter-chart",
  "training-evidence-chart",
  "scan-evidence-chart",
  "decode-evidence-chart",
];
const svgs = new Map(ids.map(id => [id, new FakeSvgNode("svg", id)]));

Object.defineProperty(globalThis, "document", {value: {
  createElementNS: (_namespace, tagName) => new FakeSvgNode(tagName),
  querySelector: selector => svgs.get(selector.slice(1)) || null,
}});

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const load = async path => JSON.parse(await readFile(join(root, path), "utf8"));
const [scan, paperScale, inference] = await Promise.all([
  load("benchmark_results_griffin_complete/canonical_griffin_axes.json"),
  load("benchmark_results_small_model/paper_scale_h800.json"),
  load("benchmark_results_small_model/paper_scale_inference_h800.json"),
]);

renderAdvantageFigures({scan, paperScale, inference});

const failures = [];
for (const [id, svg] of svgs) {
  if (!svg.attributes.get("viewBox")) failures.push(`${id}: missing viewBox`);
  if (svg.children.length < 20) failures.push(`${id}: rendered only ${svg.children.length} SVG nodes`);
  if (!svg.children.some(child => child.tagName === "path" || child.tagName === "rect")) {
    failures.push(`${id}: no plotted geometry`);
  }
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(JSON.stringify(Object.fromEntries(
  [...svgs].map(([id, svg]) => [id, svg.children.length]),
), null, 2));
