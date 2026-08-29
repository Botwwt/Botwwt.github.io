import {activateWhenVisible, AnimationController, bindTransport, css, fitCanvas, lerp, roundedRect} from "../animation-controller.js";

const line = (ctx, a, b, color, width = 2) => {
  ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
};
const dot = (ctx, x, y, radius, color) => {
  ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
};
const label = (ctx, value, x, y, color = css("--ink"), size = 12, align = "center") => {
  ctx.fillStyle = color;
  ctx.font = `500 ${size}px "Noto Sans SC", sans-serif`;
  ctx.textAlign = align;
  ctx.fillText(value, x, y);
};
const arrow = (ctx, a, b, color, width = 1.5) => {
  line(ctx, a, b, color, width);
  const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const size = 7;
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - Math.cos(angle - .5) * size, b[1] - Math.sin(angle - .5) * size);
  ctx.lineTo(b[0] - Math.cos(angle + .5) * size, b[1] - Math.sin(angle + .5) * size);
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
};
const card = (ctx, x, y, width, height, title, subtitle, color, active = false) => {
  roundedRect(ctx, x, y, width, height, 8);
  ctx.fillStyle = active ? `${color}18` : css("--paper"); ctx.fill();
  ctx.strokeStyle = active ? color : css("--line-strong");
  ctx.lineWidth = active ? 2 : 1; ctx.stroke();
  label(ctx, title, x + width / 2, y + height * .43, active ? color : css("--ink"), 13);
  label(ctx, subtitle, x + width / 2, y + height * .68, css("--muted"), 10);
};

export function initArchitecture() {
  const canvas = document.querySelector("#architecture-canvas");
  if (!canvas) return;
  const explainer = document.querySelector("#architecture-explainer");
  const descriptions = [
    "当前输入先进入一次打包的 BF16 稠密投影。",
    "投影同时得到 2M 个写入实数和两个共享控制量 c、d。",
    "静态 ν、θ 与共享 c、d 组合，为每个复数模态重建衰减和旋转。",
    "不同模态并行更新；同一模态仍沿时间读取上一状态。",
    "新状态写回固定大小的 FP32 缓存，供下一个 token 使用。",
  ];
  const controller = new AnimationController({
    steps: descriptions.length,
    duration: 1900,
    onFrame: draw,
    onStep(step) { if (explainer) explainer.textContent = descriptions[step]; },
  });

  function draw(step, progress) {
    const {ctx, width: w, height: h} = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const dynamic = css("--dynamic");
    const state = css("--state");
    const write = css("--write");
    const motion = css("--motion");
    const lineColor = css("--line-strong");
    const nodes = [
      {x: w * .035, y: h * .39, w: w * .15, h: h * .2, title: "当前输入", sub: "[B,D] · BF16", color: dynamic},
      {x: w * .25, y: h * .18, w: w * .19, h: h * .2, title: "共享控制 c、d", sub: "每个 token 仅 2 个", color: dynamic},
      {x: w * .25, y: h * .63, w: w * .19, h: h * .2, title: "复数写入", sub: "[B,M,2] · BF16", color: write},
      {x: w * .53, y: h * .28, w: w * .25, h: h * .44, title: "模态状态更新", sub: "衰减 · 旋转 · 加写入", color: state},
      {x: w * .84, y: h * .39, w: w * .13, h: h * .2, title: "新状态", sub: "[B,M,2] · FP32", color: state},
    ];
    const links = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]];
    links.forEach(([from, to]) => {
      const a = nodes[from], b = nodes[to];
      arrow(ctx, [a.x + a.w, a.y + a.h / 2], [b.x, b.y + b.h / 2], lineColor);
    });
    nodes.forEach((node, index) => card(
      ctx, node.x, node.y, node.w, node.h,
      node.title, node.sub, node.color,
      [0, 1, 1, 3, 4][step] === index,
    ));
    label(ctx, "静态谱参数 ν、θ", nodes[3].x + nodes[3].w / 2, h * .1, css("--static"), 11);
    arrow(ctx, [nodes[3].x + nodes[3].w / 2, h * .13], [nodes[3].x + nodes[3].w / 2, nodes[3].y], css("--static"));
    label(ctx, "上一状态（固定大小缓存）", nodes[3].x + nodes[3].w / 2, h * .91, state, 11);
    arrow(ctx, [nodes[3].x + nodes[3].w / 2, h * .87], [nodes[3].x + nodes[3].w / 2, nodes[3].y + nodes[3].h], state);
    const [from, to] = links[Math.min(step, links.length - 1)];
    const a = nodes[from], b = nodes[to];
    dot(
      ctx,
      lerp(a.x + a.w, b.x, progress),
      lerp(a.y + a.h / 2, b.y + b.h / 2, progress),
      5,
      step === 1 ? write : step >= 3 ? state : motion,
    );
  }

  bindTransport(document.querySelector('[data-controller="architecture"]'), controller, {play: "播放", pause: "暂停"});
  controller.onStep?.(0);
  activateWhenVisible(canvas.closest(".hero-figure"), controller);
}

export function initCoherent(root) {
  const canvas = root.querySelector("[data-coherent-canvas]");
  const cInput = root.querySelector("[data-coherent-c]");
  const dInput = root.querySelector("[data-coherent-d]");
  const readout = root.querySelector("[data-coherent-readout]");
  if (!canvas || !cInput || !dInput || !readout) return;
  const modes = Array.from({length: 12}, (_, index) => ({
    nu: .025 + index * .028,
    theta: -Math.PI + index * Math.PI * 2 / 12,
  }));

  function draw() {
    const {ctx, width: w, height: h} = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const c = Number(cInput.value), d = Number(dInput.value), gain = Math.exp(c);
    const muted = css("--muted"), grid = css("--line"), dynamic = css("--dynamic");
    const state = css("--state"), motion = css("--motion"), baseline = css("--line-strong");

    const left = {x: 42, y: 95, w: w * .52 - 66, h: h - 155};
    const right = {x: w * .58, y: 95, w: w * .37, h: h - 155};
    label(ctx, `衰减控制 c = ${c.toFixed(2)}`, left.x, 32, dynamic, 14, "left");
    label(ctx, "同一个 c 经过不同 ν，得到不同保留率 ρ", left.x, 57, muted, 11, "left");
    for (let tick = 0; tick <= 4; tick += 1) {
      const yy = left.y + tick * left.h / 4;
      line(ctx, [left.x, yy], [left.x + left.w, yy], grid, 1);
      label(ctx, (1 - tick / 4).toFixed(2), left.x - 10, yy + 4, muted, 9, "right");
    }
    const values = modes.map((mode, index) => {
      const rho = Math.exp(-mode.nu * gain);
      const baseRho = Math.exp(-mode.nu);
      const x = left.x + index * left.w / (modes.length - 1);
      const y = left.y + (1 - rho) * left.h;
      const baseY = left.y + (1 - baseRho) * left.h;
      line(ctx, [x, baseY], [x, y], motion, 1.4);
      dot(ctx, x, baseY, 3.2, baseline);
      dot(ctx, x, y, 5, state);
      label(ctx, String(index), x, left.y + left.h + 20, muted, 9);
      return {...mode, rho, phase: mode.theta + d};
    });
    label(ctx, "模态编号", left.x + left.w / 2, left.y + left.h + 43, muted, 10);

    label(ctx, `相位控制 d = ${d.toFixed(2)} 弧度`, right.x, 32, dynamic, 14, "left");
    label(ctx, "同一个 d 为所有基础相位增加相同角度", right.x, 57, muted, 11, "left");
    const radius = Math.min(right.w, right.h) * .38;
    const cx = right.x + right.w / 2, cy = right.y + right.h / 2;
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.stroke();
    line(ctx, [cx - radius - 8, cy], [cx + radius + 8, cy], grid, 1);
    line(ctx, [cx, cy - radius - 8], [cx, cy + radius + 8], grid, 1);
    values.forEach((mode, index) => {
      const base = [cx + Math.cos(mode.theta) * radius, cy + Math.sin(mode.theta) * radius];
      const current = [cx + Math.cos(mode.phase) * radius, cy + Math.sin(mode.phase) * radius];
      dot(ctx, base[0], base[1], 3, baseline);
      arrow(ctx, base, current, motion, 1.2);
      dot(ctx, current[0], current[1], 4.5, dynamic);
      if (index % 3 === 0) label(ctx, `m${index}`, current[0], current[1] - 10, muted, 9);
    });
    label(ctx, "灰点：基础相位；彩色点：当前相位", cx, h - 22, muted, 10);

    const picks = [0, 3, 6, 9];
    readout.innerHTML = `<table><thead><tr><th>模态</th><th>静态 ν</th><th>当前保留率 ρ</th><th>静态 θ</th><th>当前相位 θ+d</th></tr></thead><tbody>${picks.map(index => `<tr><td>${index}</td><td>${values[index].nu.toFixed(3)}</td><td>${values[index].rho.toFixed(3)}</td><td>${values[index].theta.toFixed(2)}</td><td>${values[index].phase.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
  }
  cInput.addEventListener("input", draw);
  dInput.addEventListener("input", draw);
  draw();
}

export function initCoreFigure(root) {
  if (root.dataset.figure === "coherent") initCoherent(root);
}
