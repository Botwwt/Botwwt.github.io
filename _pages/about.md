---
permalink: /
title: ""
excerpt: ""
lang: en
lang_switch_label: "中文"
lang_switch_url: "/zh/"
author_profile: false
profile_cover: true
author_bio: "B.S. Student in Foundational Mathematical Sciences"
profile_intro: "I study efficient, decision-oriented machine learning, with current work spanning brain-inspired credit assignment, large language models, and reinforcement learning."
redirect_from:
  - /about/
  - /about.html
---

<div class="academic-homepage">
  <main class="content-shell">
    <section id="bio" class="site-section biography-section">
      <div class="section-heading">
        <div>
          <span class="section-index">01</span>
          <h2>Biography</h2>
        </div>
        <p>Mathematics, learning systems, and intelligent decision-making.</p>
      </div>

      <div class="biography-grid">
        <div class="biography-copy">
          <p>I am a B.S. student in <strong>Foundational Mathematical Sciences</strong> at <a href="https://www.dlut.edu.cn/" target="_blank" rel="noopener noreferrer">Dalian University of Technology</a>, expected to graduate in June 2027. My current GPA is <strong>4.22/5.00</strong>, with a weighted average of <strong>91.91/100</strong> and a major rank of <strong>12/102</strong>.</p>
          <p>My research centers on efficient and decision-oriented machine learning. I am currently a research intern at <strong>Peking University</strong> and the <strong>Institute of Automation, Chinese Academy of Sciences</strong>, working on biologically inspired credit assignment, decision-making LLMs, long-context modeling, and learning-based optimization.</p>
        </div>

        <aside class="profile-facts" aria-label="Profile highlights">
          <div><span>Current</span><strong>Research Intern</strong></div>
          <div><span>Focus</span><strong>Machine Learning &amp; AI</strong></div>
          <div><span>Graduation</span><strong>June 2027</strong></div>
        </aside>
      </div>

      <div class="institution-row education-row">
        <div class="institution-logo">
          <img src="{{ '/images/institutions/dlut.png' | relative_url }}" alt="Dalian University of Technology logo" loading="lazy">
        </div>
        <div class="institution-main">
          <p class="institution-kicker">Education</p>
          <h3><a href="https://www.dlut.edu.cn/" target="_blank" rel="noopener noreferrer">Dalian University of Technology</a></h3>
          <p>B.S. in Foundational Mathematical Sciences · Sep. 2023 – Jun. 2027 (expected)</p>
        </div>
        <dl class="education-stats">
          <div><dt>GPA</dt><dd>4.22 / 5.00</dd></div>
          <div><dt>Average</dt><dd>91.91 / 100</dd></div>
          <div><dt>Rank</dt><dd>12 / 102</dd></div>
        </dl>
      </div>
    </section>

    <section id="research" class="site-section research-section">
      <div class="section-heading">
        <div>
          <span class="section-index">02</span>
          <h2>Research Focus</h2>
        </div>
        <p>Methods that learn efficiently and act under real constraints.</p>
      </div>

      <div class="focus-grid">
        <article class="focus-item">
          <span class="focus-number">01</span>
          <div>
            <h3>Brain-Inspired Credit Assignment</h3>
            <p>Online, biologically plausible, and memory-efficient alternatives to standard backpropagation through time.</p>
            <div class="inline-links"><a href="https://openreview.net/forum?id=jI2TM7Kzlu" target="_blank" rel="noopener noreferrer">COLA</a><a href="https://github.com/Criticality-Cognitive-Computation-Lab/COLA" target="_blank" rel="noopener noreferrer">Code</a></div>
          </div>
        </article>
        <article class="focus-item">
          <span class="focus-number">02</span>
          <div>
            <h3>Decision-Making LLMs</h3>
            <p>Language models as policy and reasoning systems for wireless resource allocation and cooperative edge caching.</p>
            <div class="inline-links"><a href="https://ieeexplore.ieee.org/document/11180008/" target="_blank" rel="noopener noreferrer">COMST Survey</a><a href="https://arxiv.org/abs/2602.13307" target="_blank" rel="noopener noreferrer">Edge Caching</a></div>
          </div>
        </article>
        <article class="focus-item">
          <span class="focus-number">03</span>
          <div>
            <h3>Long-Context Modeling</h3>
            <p>Linear-memory distillation and restoration methods for extending the usable context of large language models.</p>
            <div class="inline-links"><a href="mailto:shiyanxi1@mail.dlut.edu.cn?subject=LinearARD%20Preprint%20Request">LinearARD</a></div>
          </div>
        </article>
        <article class="focus-item">
          <span class="focus-number">04</span>
          <div>
            <h3>Neural Combinatorial Optimization</h3>
            <p>Deep reinforcement learning for structured routing and pickup-and-delivery problems.</p>
            <div class="inline-links"><a href="{{ '/files/papers/caadrl-pdp.pdf' | relative_url }}" target="_blank">CAADRL</a><a href="https://github.com/Botwwt/CluPDTSP" target="_blank" rel="noopener noreferrer">Code</a></div>
          </div>
        </article>
      </div>
    </section>

    <section id="publications" class="site-section publications-section">
      <div class="section-heading section-heading--actions">
        <div>
          <span class="section-index">03</span>
          <h2>Publications</h2>
        </div>
        <div class="publication-filter" role="group" aria-label="Filter publications">
          <button type="button" data-filter="selected" aria-pressed="false">Selected</button>
          <button type="button" class="is-active" data-filter="all" aria-pressed="true">All</button>
        </div>
      </div>
      <p class="section-note">{{ site.data.publications.quartile_note.en }}</p>
      <div class="publication-list">
        {% for publication in site.data.publications.items %}
          {% include publication-card.html publication=publication lang='en' variant='home' %}
        {% endfor %}
      </div>
      <a class="section-more" href="{{ '/publications/' | relative_url }}">Full publication details <i class="fas fa-arrow-right" aria-hidden="true"></i></a>
    </section>

    <section id="experience" class="site-section experience-section">
      <div class="section-heading">
        <div>
          <span class="section-index">04</span>
          <h2>Research Experience</h2>
        </div>
        <p>Current and recent research appointments.</p>
      </div>

      <div class="experience-list">
        <article class="experience-item">
          <div class="institution-logo"><img src="{{ '/images/institutions/pku.png' | relative_url }}" alt="Peking University logo" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">Research Intern</p><h3><a href="https://www.pku.edu.cn/" target="_blank" rel="noopener noreferrer">Peking University</a></h3></div><time>Oct. 2025 – Present</time></div>
            <p>Study temporal credit assignment beyond standard BPTT through online, biologically inspired, and memory-efficient local learning rules; analyze approximation error, stability, and scalability in recurrent and spiking systems.</p>
          </div>
        </article>

        <article class="experience-item">
          <div class="institution-logo"><img src="{{ '/images/institutions/casia.jpg' | relative_url }}" alt="Institute of Automation, Chinese Academy of Sciences logo" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">Research Intern</p><h3><a href="https://ia.cas.cn/" target="_blank" rel="noopener noreferrer">Institute of Automation, Chinese Academy of Sciences</a></h3></div><time>Dec. 2024 – Present</time></div>
            <p>Work on decision-making LLMs for wireless communication, cooperative edge caching, and long-context restoration; contribute to training pipelines, experiments, visualization, technical writing, and patent materials.</p>
          </div>
        </article>

        <article class="experience-item">
          <div class="institution-logo"><img src="{{ '/images/institutions/dlut.png' | relative_url }}" alt="Dalian University of Technology logo" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">Undergraduate Researcher</p><h3><a href="https://www.dlut.edu.cn/" target="_blank" rel="noopener noreferrer">Dalian University of Technology</a></h3></div><time>Oct. 2024 – Mar. 2025</time></div>
            <p>Led a neural combinatorial optimization project for pickup-and-delivery problems, including model design, reinforcement learning training, benchmark construction, comparative experiments, and manuscript writing.</p>
          </div>
        </article>

        <article class="experience-item">
          <div class="institution-logo institution-logo--dark"><img src="{{ '/images/institutions/eit.jpg' | relative_url }}" alt="Eastern Institute of Technology, Ningbo logo" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">Research Intern</p><h3><a href="https://www.eitech.edu.cn/en/" target="_blank" rel="noopener noreferrer">Eastern Institute of Technology, Ningbo</a></h3></div><time>Aug. 2025 – Nov. 2025</time></div>
            <p>Reviewed representative literature on world models and vision-language models and summarized major technical routes.</p>
          </div>
        </article>
      </div>
    </section>

    <section id="codebase" class="site-section code-section">
      <div class="section-heading">
        <div>
          <span class="section-index">05</span>
          <h2>Open Source</h2>
        </div>
        <a class="heading-link" href="https://github.com/Botwwt" target="_blank" rel="noopener noreferrer">GitHub profile <i class="fas fa-external-link-alt" aria-hidden="true"></i></a>
      </div>

      <div class="repo-grid">
        <article class="repo-card"><div class="repo-card__top"><i class="fab fa-github" aria-hidden="true"></i><span>ICML 2026</span></div><h3><a href="https://github.com/Criticality-Cognitive-Computation-Lab/COLA" target="_blank" rel="noopener noreferrer">COLA</a></h3><p>Criticality-driven local learning for global temporal credit assignment.</p><a class="repo-card__link" href="https://openreview.net/forum?id=jI2TM7Kzlu" target="_blank" rel="noopener noreferrer">OpenReview</a></article>
        <article class="repo-card"><div class="repo-card__top"><i class="fab fa-github" aria-hidden="true"></i><span>IEEE TMC</span></div><h3><a href="https://github.com/gracefulning/CoopLLM-Cache" target="_blank" rel="noopener noreferrer">CoopLLM-Cache</a></h3><p>SFT and GRPO training for LLM-based cooperative edge caching.</p><a class="repo-card__link" href="https://arxiv.org/abs/2602.13307" target="_blank" rel="noopener noreferrer">arXiv</a></article>
        <article class="repo-card"><div class="repo-card__top"><i class="fab fa-github" aria-hidden="true"></i><span>CAADRL</span></div><h3><a href="https://github.com/Botwwt/CluPDTSP" target="_blank" rel="noopener noreferrer">CluPDTSP</a></h3><p>Cluster-aware deep reinforcement learning for pickup-and-delivery problems.</p><a class="repo-card__link" href="{{ '/files/papers/caadrl-pdp.pdf' | relative_url }}" target="_blank">Paper</a></article>
      </div>
    </section>

    <section id="honors" class="site-section honors-section">
      <div class="section-heading">
        <div>
          <span class="section-index">06</span>
          <h2>Selected Honors</h2>
        </div>
      </div>
      <ol class="honors-list">
        <li><time>Jun. 2026</time><span>Kaggle BirdCLEF+ 2026 Bronze Medal</span></li>
        <li><time>Mar. 2026</time><span>Kaggle March Machine Learning Mania 2026 Bronze Medal</span></li>
        <li><time>Mar. 2026</time><span>Second Prize, Climbing Cup, Dalian University of Technology</span></li>
        <li><time>Feb. 2026</time><span>Meritorious Winner, Mathematical Contest in Modeling</span></li>
        <li><time>Nov. 2025</time><span>National Second Prize, National English Translation Competition</span></li>
        <li><time>2024 &amp; 2025</time><span>Academic Excellence Scholarship, Dalian University of Technology</span></li>
        <li><time>Nov. 2024</time><span>National First Prize, Asia and Pacific Mathematical Contest in Modeling</span></li>
        <li><time>Sep. 2024</time><span>Provincial First Prize, Contemporary Undergraduate Mathematical Contest in Modeling</span></li>
      </ol>
    </section>

    <section id="service" class="site-section service-section">
      <div class="section-heading">
        <div>
          <span class="section-index">07</span>
          <h2>Academic Service</h2>
        </div>
      </div>
      <div class="service-grid">
        <div><i class="fas fa-pen-nib" aria-hidden="true"></i><span>Journal Reviewer</span><strong>Journal on Wireless Communications and Networking</strong></div>
        <div><i class="fas fa-chart-line" aria-hidden="true"></i><span>Community</span><strong>Kaggle Competitions Expert</strong></div>
        <div><i class="fas fa-globe-asia" aria-hidden="true"></i><span>Academic Exchange</span><strong>National University of Singapore Summer Program</strong></div>
      </div>
    </section>
  </main>
</div>