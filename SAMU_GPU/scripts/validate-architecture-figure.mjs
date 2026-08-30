import {initArchitecture} from "../src/visualizations/core.js";

const labels = [];
let geometryCalls = 0;
let currentPath = [];
const strokedPaths = [];
const context = {
  setTransform() {}, clearRect() {}, save() {}, restore() {}, scale() {},
  beginPath() { geometryCalls += 1; currentPath = []; },
  moveTo(x, y) { currentPath.push([x, y]); },
  lineTo(x, y) { currentPath.push([x, y]); },
  closePath() {}, arc() { geometryCalls += 1; },
  roundRect() { geometryCalls += 1; },
  stroke() { if (currentPath.length > 1) strokedPaths.push([...currentPath]); },
  fill() {}, setLineDash() {},
  fillText(value) { labels.push(String(value)); },
};
const canvas = {
  width: 960,
  height: 520,
  getBoundingClientRect: () => ({width: 960, height: 520}),
  getContext: () => context,
};

Object.defineProperties(globalThis, {
  devicePixelRatio: {value: 1, configurable: true},
  getComputedStyle: {value: () => ({getPropertyValue: () => "#456b73"}), configurable: true},
  ResizeObserver: {value: class { observe() {} }, configurable: true},
  MutationObserver: {value: class { observe() {} }, configurable: true},
  document: {value: {
    querySelector: selector => selector === "#architecture-canvas" ? canvas : null,
    documentElement: {},
    fonts: {ready: Promise.resolve()},
  }, configurable: true},
});

initArchitecture();
await Promise.resolve();

const requiredLabels = [
  "Hawk 残差块",
  "门控前馈块",
  "SAMU 递推混合块",
  "因果 Conv1D",
  "SAMU",
  "Griffin 基线在此使用 RG-LRU",
];
const missing = requiredLabels.filter(value => !labels.includes(value));
const diagonalSegments = strokedPaths.flatMap(path => path.slice(1).map((point, index) => [path[index], point]))
  .filter(([from, to]) => from[0] !== to[0] && from[1] !== to[1]);
if (missing.length || geometryCalls < 30 || diagonalSegments.length) {
  console.error(`architecture figure failed: missing=${missing.join(",")} geometry=${geometryCalls} diagonal=${diagonalSegments.length}`);
  process.exit(1);
}

console.log(JSON.stringify({labels: labels.length, geometryCalls, routedPaths: strokedPaths.length, diagonalSegments: 0}, null, 2));
