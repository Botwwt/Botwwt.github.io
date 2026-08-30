import {lessons, parts} from "./lessons.js";
import {initArchitecture, initCoreFigure} from "./visualizations/core.js";
import {initCompleteAnalysis} from "./complete-analysis.js";
import {renderMathWhenReady} from "./math.js";

const content = document.querySelector("#lesson-content");
const nav = document.querySelector("#lesson-links");

function lessonHTML(lesson, index) {
  const part = parts[index] ? `<section class="part-divider" id="${parts[index].id}"><p class="part-label">第 ${parts[index].roman} 部分</p><h2>${parts[index].title}</h2><p>${parts[index].text}</p></section>` : "";
  return `${part}<section class="lesson" id="lesson-${index}" data-lesson="${index}"><header class="lesson-header"><span class="lesson-number">${String(index).padStart(2, "0")}</span><h3>${lesson.title}</h3></header><p class="lead">${lesson.lead}</p>${lesson.body}${index < lessons.length - 1 ? `<a class="lesson-next" href="#lesson-${index + 1}">下一节 · ${lessons[index + 1].title} ↓</a>` : ""}</section>`;
}

content.innerHTML = lessons.map(lessonHTML).join("");
nav.innerHTML = lessons.map((lesson, index) => `<li><a href="#lesson-${index}"><span>${String(index).padStart(2, "0")}</span> ${lesson.title}</a></li>`).join("");
renderMathWhenReady(document);
initArchitecture();
document.querySelectorAll("[data-figure]").forEach(initCoreFigure);
initCompleteAnalysis().catch(error => {
  console.error("H800 result loading failed", error);
  document.querySelector("#training-result").innerHTML = "<p><b>训练结果文件读取失败。</b> 请刷新页面或直接查看下方原始数据链接。</p>";
  document.querySelector("#inference-result").innerHTML = "<p><b>推理结果文件读取失败。</b> 请刷新页面或直接查看下方原始数据链接。</p>";
});

const navItems = [...nav.children];
const sections = [...document.querySelectorAll(".lesson")];
const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  const index = Number(visible.target.dataset.lesson);
  navItems.forEach((item, itemIndex) => item.classList.toggle("active", index === itemIndex));
}, {rootMargin: "-18% 0px -62% 0px", threshold: [0, .15, .5]});
sections.forEach(section => observer.observe(section));

function progress() {
  const doc = document.documentElement;
  const maximum = doc.scrollHeight - innerHeight;
  const fraction = maximum > 0 ? Math.min(1, scrollY / maximum) : 0;
  document.querySelector("#progress-bar").style.width = `${fraction * 100}%`;
  document.querySelector("#progress-value").textContent = `${Math.round(fraction * 100)}%`;
}
addEventListener("scroll", progress, {passive: true});
progress();

const theme = document.querySelector("#theme-toggle");
const saved = localStorage.getItem("samu-theme");
if (saved) document.documentElement.dataset.theme = saved;
theme.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("samu-theme", next);
});

console.info(`SAMU GPU report ready: ${lessons.length} sections.`);
