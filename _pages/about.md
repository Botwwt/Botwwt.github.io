---
permalink: /
title: ""
excerpt: ""
lang: en
lang_switch_label: "中文"
lang_switch_url: "/zh/"
author_profile: false
profile_cover: true
author_bio: "B.S. student in Foundational Mathematical Sciences at Dalian University of Technology"
sidebar_intro: "Take joy in discovery, stay true to what matters. Remain curious in exploration and reflective in work."
redirect_from:
  - /about/
  - /about.html
---

<span class='anchor' id='bio'></span>

## Biography

I am a B.S. student in Foundational Mathematical Sciences at Dalian University of Technology, expected to graduate in June 2027. My current GPA is 4.22/5.00, with a weighted average score of 91.91/100 and a major rank of 12/102.

My research focuses on efficient and decision-oriented machine learning. I work across brain-inspired credit assignment, large language models, reinforcement learning, long-context modeling, wireless systems, and neural combinatorial optimization. I am currently a research intern at Peking University and the Institute of Automation, Chinese Academy of Sciences.

<div class="highlight-grid">
  <div class="highlight-card">
    <h3>Education</h3>
    <p>B.S. in Foundational Mathematical Sciences, Dalian University of Technology</p>
    <p>GPA 4.22/5.00, rank 12/102, expected Jun. 2027</p>
  </div>
  <div class="highlight-card">
    <h3>Current Research</h3>
    <p>Peking University: online/local credit assignment for recurrent and spiking systems</p>
    <p>CASIA: decision-making LLMs for wireless systems and long-context restoration</p>
  </div>
  <div class="highlight-card">
    <h3>Selected Outputs</h3>
    <p>ICML 2026 accepted first-author paper, IEEE COMST published survey, IEEE TMC accepted paper, and NeurIPS 2026 submission</p>
  </div>
</div>

<span class='anchor' id='research'></span>

## Research Interests

<div class="chip-row">
  <span class="chip">Brain-Inspired Learning</span>
  <span class="chip">Credit Assignment</span>
  <span class="chip">Large Language Models</span>
  <span class="chip">Reinforcement Learning</span>
  <span class="chip">Long-Context Modeling</span>
  <span class="chip">Wireless Systems</span>
  <span class="chip">Neural Combinatorial Optimization</span>
</div>

My recent work follows three connected directions: criticality-driven local learning rules for temporal credit assignment, LLM-native decision-making for wireless communication and edge caching, and structure-aware learning methods for routing and pickup-and-delivery problems. I care about methods that reduce memory cost and improve stability while keeping strong empirical performance.

<span class='anchor' id='news'></span>

## News

<ul class="news-list">
  <li><strong>2026:</strong> <em>Global Credit Assignment via Dynamical Criticality</em> was accepted to ICML 2026 (CCF-A). The paper is available on OpenReview, and the COLA codebase is public on GitHub.</li>
  <li><strong>2026:</strong> <em>Cooperative Edge Caching with Large Language Model in Wireless Networks</em> was accepted by <em>IEEE Transactions on Mobile Computing</em>.</li>
  <li><strong>2026:</strong> <em>LinearARD: Linear-Memory Attention Distillation for RoPE Restoration</em> is under review at NeurIPS 2026 (CCF-A).</li>
  <li><strong>2025:</strong> <em>Decision-Making Large Language Model for Wireless Communication</em> was published in <em>IEEE Communications Surveys & Tutorials</em>.</li>
  <li><strong>Oct. 2025:</strong> Joined Peking University as a research intern on temporal credit assignment and online local learning.</li>
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
  <a class="link-pill" href="/publications/">Full Publication List</a>
</div>

<span class='anchor' id='experience'></span>

## Research Experience

<div class="timeline-card">
  <h3>Peking University</h3>
  <p class="timeline-meta">Research Intern | Oct. 2025 - Present | Beijing, China</p>
  <p>I study temporal credit assignment beyond standard BPTT, focusing on online, biologically inspired, and memory-efficient local learning rules. My recent work analyzes approximation error, stability, and scalability in recurrent, convolutional recurrent, and spiking neural systems.</p>
</div>

<div class="timeline-card">
  <h3>Institute of Automation, Chinese Academy of Sciences</h3>
  <p class="timeline-meta">Research Intern | Dec. 2024 - Present | Beijing, China</p>
  <p>I work on collaborative edge caching, long-context restoration, and decision-making LLMs for wireless communication. I contribute to environment construction, training pipelines, experiment organization, visualization, technical writing, and follow-up patent materials.</p>
</div>

<div class="timeline-card">
  <h3>Dalian University of Technology</h3>
  <p class="timeline-meta">Undergraduate Researcher | Oct. 2024 - Mar. 2025 | Dalian, China</p>
  <p>I led a neural combinatorial optimization project for pickup-and-delivery problems, including model design, reinforcement learning training, benchmark construction, comparative experiments, and manuscript writing.</p>
</div>

<div class="timeline-card">
  <h3>Eastern Institute of Technology</h3>
  <p class="timeline-meta">Research Intern | Aug. 2025 - Nov. 2025 | Ningbo, China</p>
  <p>I reviewed representative literature on world models and vision-language models and summarized major technical routes.</p>
</div>

<span class='anchor' id='education'></span>

## Education

<div class="timeline-card">
  <h3>Dalian University of Technology</h3>
  <p class="timeline-meta">B.S. in Foundational Mathematical Sciences | Sep. 2023 - Jun. 2027 (expected) | Dalian, China</p>
  <p>GPA: 4.22/5.00 | Weighted average: 91.91/100 | Rank: 12/102</p>
  <p><strong>Selected coursework:</strong> Mathematical Analysis (99), Advanced Algebra (99), Probability Theory (99), Data Structures and Algorithms (99), Optimization Methods (99), Programming and Algorithms (100), Analytic Geometry (100).</p>
  <p><strong>English:</strong> CET-6 460, CET-4 522.</p>
</div>

## Technical Skills

<div class="chip-row">
  <span class="chip">Python</span>
  <span class="chip">MATLAB</span>
  <span class="chip">Java</span>
  <span class="chip">PyTorch</span>
  <span class="chip">Hugging Face Transformers</span>
  <span class="chip">TRL</span>
  <span class="chip">Unsloth</span>
  <span class="chip">Git</span>
  <span class="chip">Linux</span>
  <span class="chip">LaTeX</span>
</div>

<span class='anchor' id='service'></span>

## Academic Service and Activities

<div class="timeline-card">
  <h3>Reviewer</h3>
  <p class="timeline-meta">Journal on Wireless Communications and Networking</p>
  <p>Peer reviewer for manuscripts on wireless communications and networking.</p>
</div>

<div class="timeline-card">
  <h3>Kaggle and Academic Exchange</h3>
  <p class="timeline-meta">Kaggle Competitions Expert | National University of Singapore summer academic exchange participant</p>
  <p>I participate in applied machine learning competitions and academic exchange activities alongside research work.</p>
</div>

<span class='anchor' id='honors'></span>

## Honors and Awards

- **Jun. 2026:** Kaggle BirdCLEF+ 2026 Bronze Medal
- **Mar. 2026:** Kaggle March Machine Learning Mania 2026 Bronze Medal
- **Mar. 2026:** Second Prize, Climbing Cup, Dalian University of Technology
- **Feb. 2026:** Meritorious Winner, Mathematical Contest in Modeling (MCM)
- **Nov. 2025:** National Second Prize, National English Translation Competition
- **Nov. 2025:** Academic Excellence Scholarship, Dalian University of Technology
- **Dec. 2024:** Key member, Liaoning provincial undergraduate innovation project
- **Dec. 2024:** Academic Excellence Scholarship, Dalian University of Technology
- **Nov. 2024:** National First Prize, Asia and Pacific Mathematical Contest in Modeling
- **Sep. 2024:** Provincial First Prize, Contemporary Undergraduate Mathematical Contest in Modeling
- **Jan. 2024:** Honorable Mention, Mathematical Contest in Modeling / Interdisciplinary Contest in Modeling
