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

if ('IntersectionObserver' in window) {
  const links = [...document.querySelectorAll('.main-nav a')];
  const sections = links.map((link) => document.querySelector(new URL(link.href).hash)).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      links.forEach((link) => {
        const active = new URL(link.href).hash === `#${entry.target.id}`;
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-12% 0px -65% 0px', threshold: 0 });
  sections.forEach((section) => observer.observe(section));
}
