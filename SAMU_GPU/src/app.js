import {lessons,parts} from "./lessons.js";
import {initArchitecture,initCoreFigure} from "./visualizations/core.js";
import {initSystemFigure} from "./visualizations/systems.js";
import {initBenchmarkLab} from "./benchmark-lab.js";

const content=document.querySelector("#lesson-content"),nav=document.querySelector("#lesson-links");

function lessonHTML(lesson,i){
  const part=parts[i]?`<section class="part-divider" id="${parts[i].id}"><p class="part-label">Part ${parts[i].roman}</p><h2>${parts[i].title}</h2><p>${parts[i].text}</p></section>`:"";
  return `${part}<section class="lesson" id="lesson-${i}" data-lesson="${i}" data-context='${JSON.stringify(lesson.context)}'><header class="lesson-header"><span class="lesson-number">${String(i).padStart(2,"0")}</span><h3>${lesson.title}</h3></header><p class="lead">${lesson.lead}</p>${lesson.body}${i<lessons.length-1?`<a class="lesson-next" href="#lesson-${i+1}">下一课 · ${lessons[i+1].title} ↓</a>`:""}</section>`;
}

content.innerHTML=lessons.map(lessonHTML).join("");
nav.innerHTML=lessons.map((l,i)=>`<li><a href="#lesson-${i}"><span>${String(i).padStart(2,"0")}</span> ${l.title}</a></li>`).join("");

initArchitecture();
document.querySelectorAll("[data-figure]").forEach(root=>{initCoreFigure(root);initSystemFigure(root);});
initBenchmarkLab();

document.addEventListener("click",e=>{
  const answer=e.target.closest("[data-answer]");if(!answer)return;const q=answer.closest(".quiz"),correct=answer.dataset.answer===q.dataset.correct;q.querySelectorAll("[data-answer]").forEach(b=>{b.classList.toggle("primary",b===answer);b.setAttribute("aria-pressed",b===answer);});q.querySelector(".quiz-feedback").innerHTML=`<b>${correct?"正确。":"再想一步。"}</b> ${q.querySelector("template").innerHTML}`;
});

const live=document.querySelector("#live-context"),navItems=[...nav.children],sections=[...document.querySelectorAll(".lesson")];
const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(!visible)return;const i=+visible.target.dataset.lesson,ctx=JSON.parse(visible.target.dataset.context);navItems.forEach((li,j)=>li.classList.toggle("active",i===j));live.innerHTML=`<span class="context-label">Lesson ${String(i).padStart(2,"0")}</span><strong>${ctx[0]}</strong><p>${ctx.slice(1).join("<br>")}</p>`;},{rootMargin:"-18% 0px -62% 0px",threshold:[0,.15,.5]});sections.forEach(s=>observer.observe(s));

document.addEventListener("samu:context",e=>{live.innerHTML=`<span class="context-label">Interactive selection</span><strong>${e.detail.title}</strong><p>${e.detail.lines.join("<br>")}</p>`;});

function progress(){const doc=document.documentElement,max=doc.scrollHeight-innerHeight,p=max>0?Math.min(1,scrollY/max):0;document.querySelector("#progress-bar").style.width=`${p*100}%`;document.querySelector("#progress-value").textContent=`${Math.round(p*100)}%`;}
addEventListener("scroll",progress,{passive:true});progress();

const theme=document.querySelector("#theme-toggle");
const saved=localStorage.getItem("samu-theme");if(saved)document.documentElement.dataset.theme=saved;
theme.addEventListener("click",()=>{const next=document.documentElement.dataset.theme==="dark"?"light":"dark";document.documentElement.dataset.theme=next;localStorage.setItem("samu-theme",next);});

const tooltip=document.querySelector("#tooltip");
document.addEventListener("pointerover",e=>{const target=e.target.closest("[data-tip]");if(!target)return;tooltip.textContent=target.dataset.tip;tooltip.hidden=false;});
document.addEventListener("pointermove",e=>{if(tooltip.hidden)return;tooltip.style.left=`${Math.min(innerWidth-290,e.clientX+14)}px`;tooltip.style.top=`${Math.min(innerHeight-100,e.clientY+14)}px`;});
document.addEventListener("pointerout",e=>{if(e.target.closest("[data-tip]"))tooltip.hidden=true;});

console.info(`SAMU GPU course ready: ${lessons.length} lessons; benchmark figures read from JSON.`);
