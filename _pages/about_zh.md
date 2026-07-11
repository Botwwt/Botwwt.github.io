---
permalink: /zh/
title: ""
excerpt: ""
lang: zh
lang_switch_label: "English"
lang_switch_url: "/"
author_profile: false
profile_cover: true
author_name: "王文韬"
author_bio: "大连理工大学数理基础科学本科生"
profile_intro: "我的研究聚焦高效且面向决策的机器学习，目前主要探索类脑信用分配、大语言模型与强化学习。"
---

<div class="academic-homepage">
  <main class="content-shell">
    <section id="bio" class="site-section biography-section">
      <div class="section-heading">
        <div>
          <span class="section-index">01</span>
          <h2>个人简介</h2>
        </div>
        <p>数学基础、学习系统与智能决策。</p>
      </div>

      <div class="biography-grid">
        <div class="biography-copy">
          <p>我目前就读于<a href="https://www.dlut.edu.cn/" target="_blank" rel="noopener noreferrer">大连理工大学</a><strong>数理基础科学</strong>专业，预计 2027 年 6 月毕业。当前学分绩为 <strong>4.22/5.00</strong>，加权平均分为 <strong>91.91/100</strong>，专业排名 <strong>12/102</strong>。</p>
          <p>我的研究聚焦于高效且面向决策的机器学习。目前我在<strong>北京大学</strong>和<strong>中国科学院自动化研究所</strong>开展科研实习，研究方向包括类脑信用分配、决策型大语言模型、长上下文建模与学习驱动优化。</p>
        </div>

        <aside class="profile-facts" aria-label="个人概况">
          <div><span>当前身份</span><strong>科研实习生</strong></div>
          <div><span>研究方向</span><strong>机器学习与人工智能</strong></div>
          <div><span>预计毕业</span><strong>2027 年 6 月</strong></div>
        </aside>
      </div>

      <div class="institution-row education-row">
        <div class="institution-logo">
          <img src="{{ '/images/institutions/dlut.png' | relative_url }}" alt="大连理工大学校徽" loading="lazy">
        </div>
        <div class="institution-main">
          <p class="institution-kicker">教育背景</p>
          <h3><a href="https://www.dlut.edu.cn/" target="_blank" rel="noopener noreferrer">大连理工大学</a></h3>
          <p>数理基础科学本科 · 2023 年 9 月 – 2027 年 6 月（预计）</p>
        </div>
        <dl class="education-stats">
          <div><dt>学分绩</dt><dd>4.22 / 5.00</dd></div>
          <div><dt>加权均分</dt><dd>91.91 / 100</dd></div>
          <div><dt>专业排名</dt><dd>12 / 102</dd></div>
        </dl>
      </div>
    </section>

    <section id="research" class="site-section research-section">
      <div class="section-heading">
        <div>
          <span class="section-index">02</span>
          <h2>研究方向</h2>
        </div>
        <p>面向真实约束，构建高效学习与决策方法。</p>
      </div>

      <div class="focus-grid">
        <article class="focus-item">
          <span class="focus-number">01</span>
          <div>
            <h3>类脑信用分配</h3>
            <p>探索在线、具生物合理性且低内存的时序信用分配方法，作为标准时间反向传播的替代方案。</p>
            <div class="inline-links"><a href="https://openreview.net/forum?id=jI2TM7Kzlu" target="_blank" rel="noopener noreferrer">COLA</a><a href="https://github.com/Criticality-Cognitive-Computation-Lab/COLA" target="_blank" rel="noopener noreferrer">代码</a></div>
          </div>
        </article>
        <article class="focus-item">
          <span class="focus-number">02</span>
          <div>
            <h3>决策型大语言模型</h3>
            <p>研究大语言模型在无线资源分配和协同边缘缓存中的策略建模、推理与决策能力。</p>
            <div class="inline-links"><a href="https://ieeexplore.ieee.org/document/11180008/" target="_blank" rel="noopener noreferrer">COMST 综述</a><a href="https://arxiv.org/abs/2602.13307" target="_blank" rel="noopener noreferrer">边缘缓存</a></div>
          </div>
        </article>
        <article class="focus-item">
          <span class="focus-number">03</span>
          <div>
            <h3>长上下文建模</h3>
            <p>通过线性内存蒸馏与恢复方法，提升大语言模型对超长上下文的有效利用能力。</p>
            <div class="inline-links"><a href="mailto:shiyanxi1@mail.dlut.edu.cn?subject=LinearARD%20Preprint%20Request">LinearARD</a></div>
          </div>
        </article>
        <article class="focus-item">
          <span class="focus-number">04</span>
          <div>
            <h3>神经组合优化</h3>
            <p>使用深度强化学习求解具有结构约束的路径规划与取送货优化问题。</p>
            <div class="inline-links"><a href="{{ '/files/papers/caadrl-pdp.pdf' | relative_url }}" target="_blank">CAADRL</a><a href="https://github.com/Botwwt/CluPDTSP" target="_blank" rel="noopener noreferrer">代码</a></div>
          </div>
        </article>
      </div>
    </section>

    <section id="publications" class="site-section publications-section">
      <div class="section-heading section-heading--actions">
        <div>
          <span class="section-index">03</span>
          <h2>论文成果</h2>
        </div>
        <div class="publication-filter" role="group" aria-label="筛选论文">
          <button type="button" data-filter="selected" aria-pressed="false">代表性论文</button>
          <button type="button" class="is-active" data-filter="all" aria-pressed="true">全部论文</button>
        </div>
      </div>
      <p class="section-note">{{ site.data.publications.quartile_note.zh }}</p>
      <div class="publication-list">
        {% for publication in site.data.publications.items %}
          {% include publication-card.html publication=publication lang='zh' variant='home' %}
        {% endfor %}
      </div>
      <a class="section-more" href="{{ '/zh/publications/' | relative_url }}">查看完整论文信息 <i class="fas fa-arrow-right" aria-hidden="true"></i></a>
    </section>

    <section id="experience" class="site-section experience-section">
      <div class="section-heading">
        <div>
          <span class="section-index">04</span>
          <h2>研究经历</h2>
        </div>
        <p>当前及近期科研任职。</p>
      </div>

      <div class="experience-list">
        <article class="experience-item">
          <div class="institution-logo"><img src="{{ '/images/institutions/pku.png' | relative_url }}" alt="北京大学校徽" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">科研实习生</p><h3><a href="https://www.pku.edu.cn/" target="_blank" rel="noopener noreferrer">北京大学</a></h3></div><time>2025 年 10 月 – 至今</time></div>
            <p>研究标准 BPTT 之外的时序信用分配方法，设计在线、具生物启发且低内存的局部学习规则，并分析其在循环网络与脉冲神经网络中的近似误差、稳定性和可扩展性。</p>
          </div>
        </article>

        <article class="experience-item">
          <div class="institution-logo"><img src="{{ '/images/institutions/casia.jpg' | relative_url }}" alt="中国科学院自动化研究所标识" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">科研实习生</p><h3><a href="https://ia.cas.cn/" target="_blank" rel="noopener noreferrer">中国科学院自动化研究所</a></h3></div><time>2024 年 12 月 – 至今</time></div>
            <p>参与无线通信决策型大语言模型、协同边缘缓存与长上下文恢复研究，负责训练流程、实验组织、图表绘制、技术写作与后续专利材料整理。</p>
          </div>
        </article>

        <article class="experience-item">
          <div class="institution-logo"><img src="{{ '/images/institutions/dlut.png' | relative_url }}" alt="大连理工大学校徽" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">本科科研</p><h3><a href="https://www.dlut.edu.cn/" target="_blank" rel="noopener noreferrer">大连理工大学</a></h3></div><time>2024 年 10 月 – 2025 年 3 月</time></div>
            <p>主导取送货问题神经组合优化项目，完成模型设计、强化学习训练、基准构建、对比实验与论文撰写。</p>
          </div>
        </article>

        <article class="experience-item">
          <div class="institution-logo institution-logo--dark"><img src="{{ '/images/institutions/eit.jpg' | relative_url }}" alt="宁波东方理工大学校徽" loading="lazy"></div>
          <div class="experience-content">
            <div class="experience-title-row"><div><p class="institution-kicker">科研实习生</p><h3><a href="https://www.eitech.edu.cn/" target="_blank" rel="noopener noreferrer">宁波东方理工大学</a></h3></div><time>2025 年 8 月 – 2025 年 11 月</time></div>
            <p>阅读世界模型与视觉语言大模型相关代表性文献，梳理主要方法和技术路线。</p>
          </div>
        </article>
      </div>
    </section>

    <section id="codebase" class="site-section code-section">
      <div class="section-heading">
        <div>
          <span class="section-index">05</span>
          <h2>开源项目</h2>
        </div>
        <a class="heading-link" href="https://github.com/Botwwt" target="_blank" rel="noopener noreferrer">GitHub 主页 <i class="fas fa-external-link-alt" aria-hidden="true"></i></a>
      </div>

      <div class="repo-grid">
        <article class="repo-card"><div class="repo-card__top"><i class="fab fa-github" aria-hidden="true"></i><span>ICML 2026</span></div><h3><a href="https://github.com/Criticality-Cognitive-Computation-Lab/COLA" target="_blank" rel="noopener noreferrer">COLA</a></h3><p>基于临界动力学的局部学习与全局时序信用分配。</p><a class="repo-card__link" href="https://openreview.net/forum?id=jI2TM7Kzlu" target="_blank" rel="noopener noreferrer">OpenReview</a></article>
        <article class="repo-card"><div class="repo-card__top"><i class="fab fa-github" aria-hidden="true"></i><span>IEEE TMC</span></div><h3><a href="https://github.com/gracefulning/CoopLLM-Cache" target="_blank" rel="noopener noreferrer">CoopLLM-Cache</a></h3><p>用于大语言模型协同边缘缓存的 SFT 与 GRPO 训练流程。</p><a class="repo-card__link" href="https://arxiv.org/abs/2602.13307" target="_blank" rel="noopener noreferrer">arXiv</a></article>
        <article class="repo-card"><div class="repo-card__top"><i class="fab fa-github" aria-hidden="true"></i><span>CAADRL</span></div><h3><a href="https://github.com/Botwwt/CluPDTSP" target="_blank" rel="noopener noreferrer">CluPDTSP</a></h3><p>面向取送货问题的聚类感知深度强化学习方法。</p><a class="repo-card__link" href="{{ '/files/papers/caadrl-pdp.pdf' | relative_url }}" target="_blank">论文</a></article>
      </div>
    </section>

    <section id="honors" class="site-section honors-section">
      <div class="section-heading">
        <div>
          <span class="section-index">06</span>
          <h2>代表性荣誉</h2>
        </div>
      </div>
      <ol class="honors-list">
        <li><time>2026 年 6 月</time><span>Kaggle BirdCLEF+ 2026 铜牌</span></li>
        <li><time>2026 年 3 月</time><span>Kaggle March Machine Learning Mania 2026 铜牌</span></li>
        <li><time>2026 年 3 月</time><span>大连理工大学攀登杯二等奖</span></li>
        <li><time>2026 年 2 月</time><span>美国大学生数学建模竞赛（MCM）Meritorious Winner</span></li>
        <li><time>2025 年 11 月</time><span>全国英语翻译大赛国家二等奖</span></li>
        <li><time>2024、2025 年</time><span>大连理工大学学业优秀奖学金</span></li>
        <li><time>2024 年 11 月</time><span>亚太地区大学生数学建模竞赛国家一等奖</span></li>
        <li><time>2024 年 9 月</time><span>全国大学生数学建模竞赛省级一等奖</span></li>
      </ol>
    </section>

    <section id="service" class="site-section service-section">
      <div class="section-heading">
        <div>
          <span class="section-index">07</span>
          <h2>学术服务</h2>
        </div>
      </div>
      <div class="service-grid">
        <div><i class="fas fa-pen-nib" aria-hidden="true"></i><span>期刊审稿</span><strong>Journal on Wireless Communications and Networking</strong></div>
        <div><i class="fas fa-chart-line" aria-hidden="true"></i><span>社区活动</span><strong>Kaggle Competitions Expert</strong></div>
        <div><i class="fas fa-globe-asia" aria-hidden="true"></i><span>学术交流</span><strong>新加坡国立大学暑期学术交流项目</strong></div>
      </div>
    </section>
  </main>
</div>