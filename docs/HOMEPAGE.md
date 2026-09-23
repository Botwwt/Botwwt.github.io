# Wentao Wang's academic homepage

The September 2026 redesign keeps the existing Jekyll / GitHub Pages deployment and the independent `/analysis/`, `/rtu/`, and `/SAMU_GPU/` projects.

## Editing content

- `_data/profile.yml`: bilingual biography, incoming Ph.D. information, education labels, research interests, news, experience, awards, and service.
- `_data/publications.yml`: shared publication records for the homepage and full publication pages. `selected: false` excludes a paper from the Selected filter, but the default All view includes every paper. Keep submitted / accepted / published states distinct.
- `_pages/about.md` and `_pages/about_zh.md`: English and Chinese homepage routes.
- `_pages/publications.md` and `_pages/publications_zh.md`: full publication routes.
- `_layouts/academic.html`, `_includes/academic-*.html`, `assets/css/academic.css`, and `assets/js/academic.js`: shared templates and responsive styling. The new layout has no external font, JavaScript, or stylesheet dependencies.
- `files/Wentao_Wang_CV.pdf`: downloadable Chinese CV. Its LaTeX source and portrait are in the same directory; run `xelatex Wentao_Wang_CV.tex` from `files/` to rebuild it. The source uses CTeX and Font Awesome.

The profile describes Peking University as **incoming** and keeps the undergraduate graduation date as **June 2027 (expected)**. No Ph.D. starting date is assumed. GPA and weighted average come from the latest CV; major rank is updated to **7/99**. LinearARD retains **second author (student second)** at the owner's explicit request.

## Design references

The layout was independently implemented, drawing on the compact academic navigation and profile organization of [Xutao Mao's homepage](https://henrymao2004.github.io/) and the readable research rows of [Jon Barron's homepage](https://jonbarron.info/). The repository's original AcadHomepage license and attribution remain in place.

## Verification and deployment

The main branch automatically builds and deploys via `.github/workflows/`. Before publishing changes, check both languages and both publication routes, local links and anchors, paper filtering (6 All / 4 Selected as of September 2026), images, keyboard focus, and narrow mobile screens. The page remains readable with JavaScript disabled. Verify the GitHub Pages build and deployed page after each push.
