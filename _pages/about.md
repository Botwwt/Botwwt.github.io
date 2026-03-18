---
permalink: /
title: ""
excerpt: ""
lang: en
lang_switch_label: "中文"
lang_switch_url: "/zh/"
author_profile: true
author_bio: "B.S. student in Foundational Mathematical Sciences at Dalian University of Technology"
sidebar_intro: "Take joy in discovery, stay true to what matters. Remain curious in exploration and reflective in work."
redirect_from:
  - /about/
  - /about.html
---

<span class='anchor' id='about'></span>

<div class="homepage-hero">
  <p class="section-kicker">Academic Homepage</p>
  <h1>Wentao Wang</h1>
  <p>I am a B.S. student in Foundational Mathematical Sciences at Dalian University of Technology. My research focuses on efficient and decision-oriented machine learning, spanning long-context large language models, reinforcement learning, neural combinatorial optimization, and biologically inspired credit assignment.</p>
  <p>I currently work with the Institute of Automation, Chinese Academy of Sciences, and Peking University. My recent outputs include a survey published in <em>IEEE Communications Surveys & Tutorials</em>, two ICML 2026 submissions under review, and ongoing manuscripts under review at <em>IEEE Transactions on Mobile Computing</em> and <em>Applied Intelligence</em>.</p>
  <div class="link-pills">
    <a class="link-pill" href="mailto:shiyanxi1@mail.dlut.edu.cn">Email</a>
    <a class="link-pill" href="https://scholar.google.com/citations?user=tF1l1S0AAAAJ&hl=zh-CN">Google Scholar</a>
    <a class="link-pill" href="https://github.com/Botwwt">GitHub</a>
    <a class="link-pill" href="/publications/">Full Publications</a>
  </div>
</div>

<div class="highlight-grid">
  <div class="highlight-card">
    <h3>Profile</h3>
    <p>B.S. in Foundational Mathematical Sciences, Dalian University of Technology</p>
    <p>GPA 4.22/5.00, Rank 12/102, expected Jun. 2027</p>
  </div>
  <div class="highlight-card">
    <h3>Current Positions</h3>
    <p>Research Intern at the Institute of Automation, Chinese Academy of Sciences</p>
    <p>Research Intern at Peking University</p>
  </div>
  <div class="highlight-card">
    <h3>Selected Outputs</h3>
    <p>IEEE COMST publication on decision-making LLMs for wireless systems</p>
    <p>ICML 2026 x2, IEEE TMC, and Applied Intelligence manuscripts under review</p>
  </div>
  <div class="highlight-card">
    <h3>Research Stack</h3>
    <p>PyTorch, Transformers, TRL, Unsloth, LoRA/QLoRA, GRPO, and large-scale experiment tooling</p>
  </div>
</div>

<span class='anchor' id='research'></span>

## Research Interests

<div class="chip-row">
  <span class="chip">Large Language Models</span>
  <span class="chip">Reinforcement Learning</span>
  <span class="chip">Neural Combinatorial Optimization</span>
  <span class="chip">Wireless Systems</span>
  <span class="chip">Long-context Modeling</span>
  <span class="chip">Credit Assignment</span>
  <span class="chip">Brain-Inspired Learning</span>
</div>

My recent work follows three connected directions: decision-making large language models for wireless systems and edge networks, structure-aware learning for routing and combinatorial optimization, and online/local learning rules for recurrent or spiking systems under near-critical dynamics. I am particularly interested in methods that improve memory efficiency, training stability, and theoretical grounding without giving up empirical performance.

<span class='anchor' id='news'></span>

## News

<ul class="news-list">
  <li><strong>Mar. 2026:</strong> Two ICML 2026 manuscripts, <em>LinearARD</em> and <em>Global Credit Assignment via Dynamical Criticality</em>, are under review.</li>
  <li><strong>Feb. 2026:</strong> <em>Cooperative Edge Caching with Large Language Model in Wireless Networks</em> is available on <a href="https://arxiv.org/abs/2602.13307">arXiv</a> and under review at <em>IEEE Transactions on Mobile Computing</em>.</li>
  <li><strong>Jan. 2026:</strong> Completed the <em>LinearARD</em> manuscript on linear-memory attention distillation for RoPE restoration.</li>
  <li><strong>Oct. 2025:</strong> Joined Peking University as a research intern to study online learning and global credit assignment in recurrent systems.</li>
  <li><strong>Dec. 2024:</strong> Joined the Institute of Automation, Chinese Academy of Sciences, as a research intern.</li>
</ul>

<span class='anchor' id='publications'></span>

{% assign publications = site.data.publications.items %}
{% assign publication_limit = site.data.publications.home_limit | default: 5 %}

## Publications

<p class="section-note">{{ site.data.publications.quartile_note.en }}</p>

{% if publications.size <= publication_limit %}
  {% for publication in publications %}
    {% include publication-card.html publication=publication lang='en' variant='home' %}
  {% endfor %}
{% endif %}

{% if publications.size > publication_limit %}
  {% for publication in publications limit:publication_limit %}
    {% include publication-card.html publication=publication lang='en' variant='home' %}
  {% endfor %}
  <details class="publication-collapse">
    <summary>Show remaining publications and manuscripts</summary>
    {% for publication in publications offset:publication_limit %}
      {% include publication-card.html publication=publication lang='en' variant='home' %}
    {% endfor %}
  </details>
{% endif %}

<div class="section-actions">
  <a class="link-pill" href="/publications/">View Full Publication List</a>
</div>

<span class='anchor' id='experience'></span>

## Research Experience

<div class="timeline-card">
  <h3>Institute of Automation, Chinese Academy of Sciences</h3>
  <p class="timeline-meta">Research Intern | Dec. 2024 - Present | Beijing, China</p>
  <p>I work on decision-making LLMs for wireless communication, cooperative multi-base-station edge caching, and efficient long-context restoration for large language models. My recent work covers both algorithm design and full training/evaluation pipelines based on supervised fine-tuning, GRPO, LoRA/QLoRA, TRL, and Unsloth.</p>
</div>

<div class="timeline-card">
  <h3>Peking University</h3>
  <p class="timeline-meta">Research Intern | Oct. 2025 - Present | Beijing, China</p>
  <p>I investigate temporal credit assignment in recurrent and spiking systems, with a focus on online and biologically inspired learning rules that approximate global gradient propagation at much lower memory cost. A central theme is how near-critical dynamics can support stable long-range sequence learning.</p>
</div>

<div class="timeline-card">
  <h3>Dalian University of Technology</h3>
  <p class="timeline-meta">Undergraduate Researcher | Oct. 2024 - Mar. 2025 | Dalian, China</p>
  <p>I led a project on neural combinatorial optimization for the pickup and delivery problem, covering model design, reinforcement learning training, benchmark construction, empirical analysis, and manuscript preparation.</p>
</div>

<span class='anchor' id='education'></span>

## Education

<div class="timeline-card">
  <h3>Dalian University of Technology</h3>
  <p class="timeline-meta">B.S. in Foundational Mathematical Sciences | Sep. 2023 - Jun. 2027 (expected) | Dalian, China</p>
  <p>GPA: 4.22/5.00 | Rank: 12/102</p>
  <p><strong>Selected coursework:</strong> Mathematical Analysis (99), Advanced Algebra (99), Probability Theory (99), Data Structures and Algorithms (99), Optimization Methods (99), Programming and Algorithms (100), Analytic Geometry (100)</p>
  <p><strong>English:</strong> CET-6 460, CET-4 522</p>
</div>

## Technical Skills

<div class="chip-row">
  <span class="chip">Python</span>
  <span class="chip">MATLAB</span>
  <span class="chip">Java</span>
  <span class="chip">PyTorch</span>
  <span class="chip">Transformers</span>
  <span class="chip">TRL</span>
  <span class="chip">Unsloth</span>
  <span class="chip">Git</span>
  <span class="chip">Linux</span>
  <span class="chip">LaTeX</span>
</div>

## Academic Service

<div class="timeline-card">
  <h3>Reviewer</h3>
  <p class="timeline-meta">Journal on Wireless Communications and Networking | CAS major-category Q4 | JCR Q2/Q3</p>
  <p>Peer reviewer for manuscripts on wireless communications and networking.</p>
</div>

<span class='anchor' id='honors'></span>

## Honors and Awards

- **Dec. 2024:** Academic Excellence Scholarship, Dalian University of Technology (Top 15%)
- **Nov. 2025:** Academic Excellence Scholarship, Dalian University of Technology (Top 15%)
- **Sep. 2024:** Provincial First Prize, Contemporary Undergraduate Mathematical Contest in Modeling
- **Nov. 2024:** National First Prize, Asia and Pacific Mathematical Contest in Modeling (APMCM)
- **Feb. 2024:** Honorable Mention, Mathematical Contest in Modeling / Interdisciplinary Contest in Modeling (MCM/ICM)
- **Nov. 2025:** National Second Prize, National English Translation Competition

## Activities

- **2023:** Starting player, runner-up, Freshman Basketball Tournament, Dalian University of Technology
- **Feb. 2024:** Participant, National University of Singapore Summer Academic Exchange
