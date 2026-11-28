# Wentao Wang's academic homepage

The academic pages use an al-folio layout on Jekyll and deploy to GitHub Pages. The repository also hosts the independent `/analysis/`, `/rtu/`, and `/SAMU_GPU/` projects.

## Editing content

- `_data/profile.yml`: bilingual biography, 2027 Ph.D. entry information, education labels, research interests, news, experience, awards, and service.
- `_data/publications.yml`: shared publication records for the homepage and full publication pages. `selected: false` excludes a paper from the Selected filter, but the default All view includes every paper. Keep submitted / accepted / published states distinct.
- `_pages/about.md` and `_pages/about_zh.md`: English and Chinese homepage routes.
- `_pages/publications.md` and `_pages/publications_zh.md`: full publication routes.
- `_layouts/academic.html`, `_includes/academic-*.html`, `assets/css/academic.css`, and `assets/js/academic.js`: shared templates, theme switching, and responsive styling. The Roboto font is served from the repository.
- `files/Wentao_Wang_CV.pdf`: downloadable Chinese CV. Its LaTeX source and portrait are in the same directory; run `xelatex Wentao_Wang_CV.tex` from `files/` to rebuild it. The source uses CTeX and Font Awesome.

The profile uses **2027年入学直博生 / incoming Ph.D. student (2027 entry)** at Peking University, as requested by the owner, and keeps the undergraduate graduation date as **June 2027 (expected)**. GPA and weighted average come from the latest CV, and major rank is **7/99**. LinearARD is **accepted to NeurIPS 2026 (CCF-A)**. Every publication displays its full author list with **Wentao Wang** in bold. Superscript `*` indicates equal contribution, and `†` identifies an advisor listed as first author.

Publications are grouped by the explicit `authorship_role` in each record. `first` and `cofirst` appear in the first-author group, with a role label and the full figure-led layout. The collaboration group uses a compact layout with smaller figures on the right. Place `second` and `advisor_first` records before other `coauthor` papers within that group. Equal-contribution marks belonging only to other authors do not make Wentao a co-first author. The All view currently contains **4 first-author papers (including 1 co-first-author paper)** and **5 collaborative papers**. The Selected view contains 3 and 2 respectively; empty groups are hidden when filtering.

## Design references

The navigation, right-aligned profile, news table, bibliography rows, and theme colors are adapted from [al-folio v0.16.3](https://github.com/alshedivat/al-folio/tree/v0.16.3) using the existing bilingual YAML data. See [the al-folio MIT license](LICENSE-al-folio.txt) and [the Roboto Open Font License](../assets/fonts/OFL-Roboto.txt).

## Verification and deployment

The main branch automatically builds and deploys via `.github/workflows/`. Before publishing changes, check both languages and both publication routes, local links and anchors, paper filtering (9 All / 5 Selected as of November 2026), images, keyboard focus, and narrow mobile screens. The page remains readable with JavaScript disabled. Verify the GitHub Pages build and deployed page after each push.

Optional `talks` in a publication record appear beneath that paper on both homepages and publication pages. The Global Credit Assignment record includes Guozhang Chen's June 13, 2026 BAAI Conference talk, with the [official agenda](https://2026.baai.ac.cn/schedule), [conference recap](https://hub.baai.ac.cn/view/55652), and [replay](https://event.baai.ac.cn/live/1033).

The BAAI talk preview uses [the official recap's photograph](https://simg.baai.ac.cn/hub-detail/60e8e3654e49d551e4d330e0198a26641781842825210.webp). Clicking the image opens the official forum replay.

Journal Impact Factors display their metric year and link to the publisher's source. The 2025 values, verified in November 2026, are COMST **50.6** ([IEEE title list](https://open.ieee.org/wp-content/uploads/IEEE-Title-List-September-2026-CAPES.pdf#page=1)), TMC **8.8** ([IEEE Computer Society](https://www.computer.org/csdl/journal/tm/)), and Applied Intelligence **3.5** ([Springer Nature](https://link.springer.com/journal/10489)).

The ESI Highly Cited Paper (Top 1%) designation for the Decision-Making survey was confirmed by the site owner on November 28, 2026. Its public ESI record was not independently located during this update.

MC-TRCM uses the architecture diagram (Figure 2) from the final ICONIP 2026 submission package, rendered from `figures/mctrcm_architecture_new.pdf`. The original manuscript figure is preserved without alteration.

The PASB preview uses the overview diagram (Figure 1), cropped from page 2 of the [arXiv v5 manuscript](https://arxiv.org/pdf/2607.10526v5#page=2).

The VisAlloc preview uses the comparison diagram (Figure 1), cropped from page 2 of the [arXiv v1 manuscript](https://arxiv.org/pdf/2611.25069v1#page=2).

The YOTO preview uses the failure-head examples (Figure 2), cropped from page 6 of the [arXiv v1 manuscript](https://arxiv.org/pdf/2611.28100v1#page=6). This paper has no equal-contribution or advisor-first annotation in its author list.

The Dalian University of Technology emblem is a transparent PNG from [the university logo resource page](https://www.urongda.com/logos/4121010141).
