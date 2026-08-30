import {css, fitCanvas, roundedRect} from "../animation-controller.js?v=20260831-1";

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
  function draw() {
    const {ctx, width: w, height: h} = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.scale(w / 960, h / 520);
    const dynamic = css("--dynamic");
    const state = css("--state");
    const write = css("--write");
    const motion = css("--motion");
    const ink = css("--ink");
    const muted = css("--muted");
    const paper = css("--paper");
    const panel = css("--panel");
    const lineColor = css("--line-strong");

    const panelBox = (x, y, width, height, title, subtitle = "") => {
      roundedRect(ctx, x, y, width, height, 18);
      ctx.fillStyle = panel; ctx.fill();
      ctx.strokeStyle = lineColor; ctx.lineWidth = 1.2; ctx.stroke();
      label(ctx, title, x + 18, y + 27, ink, 13, "left");
      if (subtitle) label(ctx, subtitle, x + 18, y + 47, muted, 10, "left");
    };
    const block = (x, y, width, height, title, color, subtitle = "") => {
      roundedRect(ctx, x, y, width, height, 7);
      ctx.fillStyle = `${color}22`; ctx.fill();
      ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke();
      label(ctx, title, x + width / 2, y + (subtitle ? height * .43 : height * .58), ink, 11);
      if (subtitle) label(ctx, subtitle, x + width / 2, y + height * .70, muted, 8.5);
    };
    const junction = (x, y) => {
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = paper; ctx.fill(); ctx.strokeStyle = lineColor; ctx.stroke();
      label(ctx, "×", x, y + 4, ink, 12);
    };
    const residual = (x, y) => {
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = paper; ctx.fill(); ctx.strokeStyle = lineColor; ctx.stroke();
      label(ctx, "+", x, y + 4, ink, 13);
    };
    const routedArrow = (points, color, width = 1.35, dashed = false) => {
      if (points.length < 2) return;
      ctx.save();
      ctx.lineCap = "square";
      ctx.lineJoin = "miter";
      if (dashed) ctx.setLineDash([7, 6]);
      ctx.beginPath();
      ctx.moveTo(...points[0]);
      points.slice(1).forEach(point => ctx.lineTo(...point));
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
      ctx.setLineDash([]);
      const end = points.at(-1), previous = points.at(-2);
      const angle = Math.atan2(end[1] - previous[1], end[0] - previous[0]);
      const size = 6;
      ctx.beginPath();
      ctx.moveTo(...end);
      ctx.lineTo(end[0] - Math.cos(angle - .48) * size, end[1] - Math.sin(angle - .48) * size);
      ctx.lineTo(end[0] - Math.cos(angle + .48) * size, end[1] - Math.sin(angle + .48) * size);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
    };

    panelBox(26, 38, 220, 440, "Hawk 残差块", "重复堆叠 N 层");
    block(72, 388, 128, 42, "RMSNorm", write);
    block(61, 306, 150, 58, "递推混合块", dynamic, "Temporal mixing");
    residual(136, 278);
    block(72, 211, 128, 42, "RMSNorm", write);
    block(61, 129, 150, 58, "门控前馈块", motion, "Gated MLP");
    residual(136, 101);
    routedArrow([[136, 468], [136, 430]], lineColor);
    routedArrow([[136, 388], [136, 364]], lineColor);
    routedArrow([[136, 306], [136, 287]], lineColor);
    routedArrow([[136, 269], [136, 253]], lineColor);
    routedArrow([[136, 211], [136, 187]], lineColor);
    routedArrow([[136, 129], [136, 110]], lineColor);
    routedArrow([[136, 92], [136, 60]], lineColor);
    routedArrow([[136, 451], [48, 451], [48, 278], [127, 278]], lineColor, 1.15);
    routedArrow([[136, 269], [48, 269], [48, 101], [127, 101]], lineColor, 1.15);
    label(ctx, "输入", 136, 496, muted, 10);
    label(ctx, "输出", 136, 28, muted, 10);

    panelBox(292, 38, 260, 225, "门控前馈块", "双方完全相同");
    block(326, 189, 82, 34, "Linear", motion);
    block(326, 137, 82, 34, "GeLU", write);
    block(437, 189, 82, 34, "Linear", motion);
    junction(423, 111);
    block(382, 58, 82, 34, "Linear", motion);
    line(ctx, [423, 248], [423, 229], lineColor, 1.2);
    routedArrow([[423, 229], [367, 229], [367, 223]], lineColor, 1.2);
    routedArrow([[423, 229], [478, 229], [478, 223]], lineColor, 1.2);
    routedArrow([[367, 189], [367, 171]], lineColor);
    routedArrow([[367, 137], [367, 111], [414, 111]], lineColor);
    routedArrow([[478, 189], [478, 111], [432, 111]], lineColor);
    routedArrow([[423, 102], [423, 92]], lineColor);

    panelBox(586, 38, 348, 440, "SAMU 递推混合块", "对应 Griffin 架构中的 RG-LRU 位置");
    block(618, 382, 92, 38, "Linear", motion, "输出门分支");
    block(618, 318, 92, 38, "GeLU", write);
    block(810, 382, 92, 38, "Linear", motion, "递推分支");
    block(793, 310, 126, 48, "因果 Conv1D", dynamic, "局部时间混合");
    block(793, 232, 126, 54, "SAMU", state, "复数模态状态更新");
    junction(760, 186);
    block(714, 105, 92, 38, "Linear", motion, "输出投影");
    label(ctx, "共享控制 cₜ、dₜ", 770, 263, state, 9.5, "right");
    label(ctx, "Griffin 基线在此使用 RG-LRU", 760, 454, muted, 9.5);
    line(ctx, [760, 465], [760, 430], lineColor, 1.2);
    routedArrow([[760, 430], [664, 430], [664, 420]], lineColor, 1.2);
    routedArrow([[760, 430], [856, 430], [856, 420]], lineColor, 1.2);
    routedArrow([[664, 382], [664, 356]], lineColor);
    routedArrow([[664, 318], [664, 186], [751, 186]], lineColor);
    routedArrow([[856, 382], [856, 358]], lineColor);
    routedArrow([[856, 310], [856, 286]], lineColor);
    routedArrow([[856, 232], [856, 186], [769, 186]], lineColor);
    routedArrow([[776, 259], [793, 259]], state, 1.25);
    routedArrow([[760, 177], [760, 143]], lineColor);
    routedArrow([[760, 105], [760, 76]], lineColor);
    label(ctx, "输出", 760, 66, muted, 10);

    routedArrow([[211, 158], [292, 158]], motion, 1.4, true);
    routedArrow([[211, 335], [586, 335]], dynamic, 1.4, true);
    label(ctx, "展开", 251, 146, motion, 9.5);
    label(ctx, "展开", 394, 323, dynamic, 9.5);
    ctx.restore();
  }

  draw();
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(draw).observe(canvas);
  if (typeof MutationObserver !== "undefined") {
    new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
  }
  document.fonts?.ready.then(draw);
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
