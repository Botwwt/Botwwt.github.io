'use strict';

const menuButton = document.querySelector('.navbar-toggler');
const menu = document.querySelector('#site-navigation');
menuButton.hidden = false;
menuButton.addEventListener('click', () => {
  const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
  menuButton.setAttribute('aria-expanded', String(!isOpen));
});
menu.addEventListener('click', (event) => {
  if (event.target.closest('a')) menuButton.setAttribute('aria-expanded', 'false');
});
document.querySelector('.navbar').addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') {
    menuButton.setAttribute('aria-expanded', 'false');
    menuButton.focus();
  }
});

const themeButton = document.querySelector('.theme-toggle');
const colorPreference = window.matchMedia('(prefers-color-scheme: dark)');
const themeColor = document.querySelector('meta[name="theme-color"]');
let theme;
try {
  theme = localStorage.getItem('academic-theme');
} catch {
  // The theme can still be changed for this page when storage is unavailable.
}

function updateTheme() {
  const dark = theme === 'dark' || (theme !== 'light' && colorPreference.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  themeButton.setAttribute('aria-pressed', String(dark));
  themeColor.content = dark ? '#1c1c1d' : '#ffffff';
}

themeButton.addEventListener('click', () => {
  theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  try {
    localStorage.setItem('academic-theme', theme);
  } catch {
    // The current page does not depend on storage.
  }
  updateTheme();
});
colorPreference.addEventListener('change', updateTheme);
themeButton.hidden = false;
updateTheme();

document.querySelectorAll('.publication-filter').forEach((filter) => {
  const section = filter.closest('.publications-section');
  const papers = section.querySelectorAll('.paper');
  const groups = section.querySelectorAll('.publication-group');
  const buttons = filter.querySelectorAll('button');
  const status = section.querySelector('.filter-status');
  filter.hidden = false;
  function applyFilter(value) {
    let visible = 0;
    buttons.forEach((button) => {
      const active = button.dataset.filter === value;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    papers.forEach((paper) => {
      paper.hidden = value !== 'all' && paper.dataset.selected !== 'true';
      if (!paper.hidden) visible += 1;
    });
    groups.forEach((group) => {
      const visiblePapers = [...group.querySelectorAll('.paper')].filter((paper) => !paper.hidden).length;
      group.hidden = visiblePapers === 0;
      group.querySelector('[data-group-count]').textContent = visiblePapers;
    });
    status.textContent = document.documentElement.lang === 'zh'
      ? `显示 ${visible} 篇论文` : `${visible} publications shown`;
  }
  buttons.forEach((button) => {
    button.addEventListener('click', () => applyFilter(button.dataset.filter));
  });
  window.addEventListener('hashchange', () => {
    const paper = document.getElementById(window.location.hash.slice(1));
    if (paper?.hidden && section.contains(paper)) {
      applyFilter('all');
      paper.scrollIntoView();
    }
  });
});

const navigation = [...menu.querySelectorAll('a')].map((link) => {
  const url = new URL(link.href);
  const section = url.pathname === location.pathname && document.getElementById(url.hash.slice(1));
  return { link, section };
}).filter((item) => item.section);
let navigationFrame = null;
function updateNavigation() {
  navigationFrame = null;
  const marker = document.querySelector('.navbar').getBoundingClientRect().bottom + 30;
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
