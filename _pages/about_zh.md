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
sidebar_intro: "乐于趣，敏于义。在探索中永葆好奇，在工作中常思反省。"
---

<span class='anchor' id='bio'></span>

## 个人简介

我目前就读于大连理工大学数理基础科学专业，预计 2027 年 6 月毕业。当前学分绩为 4.22/5.00，加权平均分为 91.91/100，专业排名 12/102。

我的研究兴趣聚焦于高效且面向决策的机器学习，覆盖类脑信用分配、大语言模型、强化学习、长上下文建模、无线系统与神经组合优化。当前我在北京大学和中国科学院自动化研究所开展科研实习。

<div class="highlight-grid">
  <div class="highlight-card">
    <h3>教育背景</h3>
    <p>大连理工大学数理基础科学本科生</p>
    <p>GPA 4.22/5.00，专业排名 12/102，预计 2027 年 6 月毕业</p>
  </div>
  <div class="highlight-card">
    <h3>当前研究</h3>
    <p>北京大学：循环与脉冲系统的在线局部信用分配</p>
    <p>中科院自动化所：无线系统决策型大语言模型与长上下文恢复</p>
  </div>
  <div class="highlight-card">
    <h3>代表成果</h3>
    <p>ICML 2026 第一作者录用论文、IEEE COMST 已发表综述、IEEE TMC 已录用论文，以及 NeurIPS 2026 在审投稿</p>
  </div>
</div>

<span class='anchor' id='research'></span>

## 研究方向

<div class="chip-row">
  <span class="chip">类脑学习</span>
  <span class="chip">信用分配</span>
  <span class="chip">大语言模型</span>
  <span class="chip">强化学习</span>
  <span class="chip">长上下文建模</span>
  <span class="chip">无线系统</span>
  <span class="chip">神经组合优化</span>
</div>

近阶段我的研究主要集中在三个方向：面向时序信用分配的临界动力学局部学习规则、面向无线通信与边缘缓存的大语言模型序列决策方法，以及利用结构先验的路径规划与取送货问题学习方法。我尤其关注如何在保持经验性能的同时降低内存开销、提升训练稳定性，并增强方法的理论解释性。

<span class='anchor' id='news'></span>

## 最新动态

<ul class="news-list">
  <li><strong>2026 年：</strong><em>Global Credit Assignment via Dynamical Criticality</em> 已被 ICML 2026（CCF-A）录用；论文链接已更新为 OpenReview，COLA 代码仓库已公开。</li>
  <li><strong>2026 年：</strong><em>Cooperative Edge Caching with Large Language Model in Wireless Networks</em> 已被 <em>IEEE Transactions on Mobile Computing</em> 录用。</li>
  <li><strong>2026 年：</strong><em>LinearARD: Linear-Memory Attention Distillation for RoPE Restoration</em> 正在 NeurIPS 2026（CCF-A）审稿中。</li>
  <li><strong>2025 年：</strong><em>Decision-Making Large Language Model for Wireless Communication</em> 已发表于 <em>IEEE Communications Surveys & Tutorials</em>。</li>
  <li><strong>2025 年 10 月：</strong>加入北京大学，开展时序信用分配与在线局部学习研究。</li>
  <li><strong>2024 年 12 月：</strong>加入中国科学院自动化研究所开展科研实习。</li>
</ul>

<span class='anchor' id='publications'></span>

{% assign publications = site.data.publications.items %}
{% assign publication_limit = site.data.publications.home_limit | default: 5 %}

## 论文与在研稿件

<p class="section-note">{{ site.data.publications.quartile_note.zh }}</p>

{% if publications.size <= publication_limit %}
  {% for publication in publications %}
    {% include publication-card.html publication=publication lang='zh' variant='home' %}
  {% endfor %}
{% endif %}

{% if publications.size > publication_limit %}
  {% for publication in publications limit:publication_limit %}
    {% include publication-card.html publication=publication lang='zh' variant='home' %}
  {% endfor %}
  <details class="publication-collapse">
    <summary>展开剩余论文与稿件</summary>
    {% for publication in publications offset:publication_limit %}
      {% include publication-card.html publication=publication lang='zh' variant='home' %}
    {% endfor %}
  </details>
{% endif %}

<div class="section-actions">
  <a class="link-pill" href="/zh/publications/">完整论文列表</a>
</div>

<span class='anchor' id='experience'></span>

## 研究经历

<div class="timeline-card">
  <h3>北京大学</h3>
  <p class="timeline-meta">科研实习生 | 2025 年 10 月至今 | 北京</p>
  <p>围绕现有 BPTT 更新方式的时序信用分配问题开展研究，关注在线、具生物启发且低内存的局部学习规则，并验证其近似误差、稳定性与可扩展性。</p>
</div>

<div class="timeline-card">
  <h3>中国科学院自动化研究所</h3>
  <p class="timeline-meta">科研实习生 | 2024 年 12 月至今 | 北京</p>
  <p>参与协同边缘缓存、长上下文恢复与无线通信大模型综述等研究，负责环境与训练流程搭建、实验组织、图表绘制、技术写作，并参与后续专利材料整理。</p>
</div>

<div class="timeline-card">
  <h3>大连理工大学</h3>
  <p class="timeline-meta">本科科研 | 2024 年 10 月至 2025 年 3 月 | 大连</p>
  <p>围绕取送货问题神经组合优化开展本科科研，完成模型设计、强化学习训练、基准构建、对比实验与论文撰写。</p>
</div>

<div class="timeline-card">
  <h3>东方理工大学</h3>
  <p class="timeline-meta">科研实习生 | 2025 年 8 月至 2025 年 11 月 | 宁波</p>
  <p>阅读世界模型与视觉语言大模型相关文献，梳理代表性方法与技术路线。</p>
</div>

<span class='anchor' id='education'></span>

## 教育背景

<div class="timeline-card">
  <h3>大连理工大学</h3>
  <p class="timeline-meta">数理基础科学本科生 | 2023 年 9 月至 2027 年 6 月（预计） | 大连</p>
  <p>GPA: 4.22/5.00 | 加权平均分: 91.91/100 | 专业排名: 12/102</p>
  <p><strong>核心课程：</strong>数学分析（99）、高等代数（99）、概率论（99）、数据结构与算法（99）、最优化方法（99）、程序设计与算法（100）、解析几何（100）。</p>
  <p><strong>英语成绩：</strong>CET-6 460，CET-4 522。</p>
</div>

## 技术技能

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

## 学术服务与活动

<div class="timeline-card">
  <h3>审稿服务</h3>
  <p class="timeline-meta">Journal on Wireless Communications and Networking 审稿人</p>
  <p>参与无线通信与网络方向稿件评审。</p>
</div>

<div class="timeline-card">
  <h3>Kaggle 与学术交流</h3>
  <p class="timeline-meta">Kaggle Competitions Expert | 新加坡国立大学暑期学术交流项目参与者</p>
  <p>在科研之外参与应用机器学习竞赛与学术交流活动。</p>
</div>

<span class='anchor' id='honors'></span>

## 荣誉奖项

- **2026 年 6 月：** Kaggle BirdCLEF+ 2026 铜牌
- **2026 年 3 月：** Kaggle March Machine Learning Mania 2026 铜牌
- **2026 年 3 月：** 大连理工大学攀登杯二等奖
- **2026 年 2 月：** 美国大学生数学建模竞赛（MCM）Meritorious Winner
- **2025 年 11 月：** 全国英语翻译大赛国家二等奖
- **2025 年 11 月：** 大连理工大学学业优秀奖学金
- **2024 年 12 月：** 辽宁省级大创项目主力队员
- **2024 年 12 月：** 大连理工大学学业优秀奖学金
- **2024 年 11 月：** 亚太地区大学生数学建模竞赛国家一等奖
- **2024 年 9 月：** 全国大学生数学建模竞赛省级一等奖
- **2024 年 1 月：** 美国大学生数学建模竞赛（MCM）Honorable Mention
