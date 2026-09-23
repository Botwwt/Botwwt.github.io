'use strict';

document.querySelectorAll('.publication-filter').forEach((filter) => {
  const section = filter.closest('.publications-section');
  const papers = section.querySelectorAll('.paper');
  const buttons = filter.querySelectorAll('button');
  const status = section.querySelector('.filter-status');
  filter.hidden = false;
  buttons.forEach((button) => {
    button.addEventListener('click', () => {
      let visible = 0;
      buttons.forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle('is-active', active);
        candidate.setAttribute('aria-pressed', String(active));
      });
      papers.forEach((paper) => {
        paper.hidden = button.dataset.filter !== 'all' && paper.dataset.selected !== 'true';
        if (!paper.hidden) visible += 1;
      });
      status.textContent = document.documentElement.lang === 'zh'
        ? `显示 ${visible} 篇论文` : `${visible} publications shown`;
    });
  });
});

const navigation = [...document.querySelectorAll('.main-nav a')].map((link) => ({
  link,
  section: document.querySelector(new URL(link.href).hash)
})).filter((item) => item.section);
let navigationFrame = null;
function updateNavigation() {
  navigationFrame = null;
  const marker = document.querySelector('.topbar').getBoundingClientRect().bottom + 80;
  let current = navigation[0];
  navigation.forEach((item) => {
    if (item.section.getBoundingClientRect().top <= marker) current = item;
  });
  navigation.forEach((item) => {
    if (item === current) item.link.setAttribute('aria-current', 'location');
    else item.link.removeAttribute('aria-current');
  });
}
function scheduleNavigation() {
  if (navigationFrame === null) navigationFrame = requestAnimationFrame(updateNavigation);
}
window.addEventListener('scroll', scheduleNavigation, { passive: true });
window.addEventListener('resize', scheduleNavigation);
window.addEventListener('load', updateNavigation);
updateNavigation();
