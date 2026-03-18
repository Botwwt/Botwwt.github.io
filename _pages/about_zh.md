---
permalink: /zh/
title: ""
excerpt: ""
lang: zh
lang_switch_label: "English"
lang_switch_url: "/"
author_profile: true
author_name: "王文涛"
author_bio: "大连理工大学数理基础科学本科生"
sidebar_intro: "乐于趣，敏于义。在探索中永葆好奇，在工作中常思反省。"
---

<span class='anchor' id='about'></span>

<div class="homepage-hero">
  <p class="section-kicker">Academic Homepage</p>
  <h1>王文涛</h1>
  <p>我目前就读于大连理工大学数理基础科学专业，研究兴趣聚焦于高效且面向决策的机器学习，包括长上下文大语言模型、强化学习、神经组合优化以及具有生物启发的信用分配学习。</p>
  <p>当前我在中国科学院自动化研究所和北京大学开展科研实习。近期成果包括 1 篇发表于 <em>IEEE Communications Surveys & Tutorials</em> 的综述、2 篇投稿 ICML 2026 且处于在审状态的论文，以及投向 <em>IEEE Transactions on Mobile Computing</em> 和 <em>Applied Intelligence</em> 的在审工作。</p>
  <div class="link-pills">
    <a class="link-pill" href="mailto:shiyanxi1@mail.dlut.edu.cn">邮箱</a>
    <a class="link-pill" href="https://scholar.google.com/citations?user=tF1l1S0AAAAJ&hl=zh-CN">谷歌学术</a>
    <a class="link-pill" href="https://github.com/Botwwt">GitHub</a>
    <a class="link-pill" href="/zh/publications/">完整论文列表</a>
  </div>
</div>

<div class="highlight-grid">
  <div class="highlight-card">
    <h3>教育背景</h3>
    <p>大连理工大学数理基础科学本科生</p>
    <p>GPA 4.22/5.00，专业排名 12/102，预计 2027 年 6 月毕业</p>
  </div>
  <div class="highlight-card">
    <h3>当前科研岗位</h3>
    <p>中国科学院自动化研究所科研实习生</p>
    <p>北京大学科研实习生</p>
  </div>
  <div class="highlight-card">
    <h3>代表性成果</h3>
    <p>IEEE COMST 综述论文，聚焦无线系统中的决策型大语言模型</p>
    <p>ICML 2026 两篇在审稿件，以及 IEEE TMC、Applied Intelligence 在审工作</p>
  </div>
  <div class="highlight-card">
    <h3>研究工具链</h3>
    <p>PyTorch、Transformers、TRL、Unsloth、LoRA/QLoRA、GRPO 及大规模实验流程搭建</p>
  </div>
</div>

<span class='anchor' id='research'></span>

## 研究方向

<div class="chip-row">
  <span class="chip">大语言模型</span>
  <span class="chip">强化学习</span>
  <span class="chip">神经组合优化</span>
  <span class="chip">无线系统</span>
  <span class="chip">长上下文建模</span>
  <span class="chip">信用分配</span>
  <span class="chip">类脑学习</span>
</div>

近阶段我的研究主要集中在三个方向：面向无线系统与边缘网络的决策型大语言模型、利用结构先验的路径规划与组合优化学习方法，以及基于临界动力学的循环/脉冲系统在线局部学习规则。我尤其关注如何在不牺牲经验性能的前提下，提高方法的内存效率、训练稳定性与理论解释性。

<span class='anchor' id='news'></span>

## 最新动态

<ul class="news-list">
  <li><strong>2026 年 3 月：</strong> <em>LinearARD</em> 与 <em>Global Credit Assignment via Dynamical Criticality</em> 两篇 ICML 2026 稿件正在审稿中。</li>
  <li><strong>2026 年 2 月：</strong> <em>Cooperative Edge Caching with Large Language Model in Wireless Networks</em> 发布于 <a href="https://arxiv.org/abs/2602.13307">arXiv</a>，并投至 <em>IEEE Transactions on Mobile Computing</em>。</li>
  <li><strong>2026 年 1 月：</strong> 完成 <em>LinearARD</em> 论文撰写，聚焦 RoPE 恢复中的线性内存注意力蒸馏。</li>
  <li><strong>2025 年 10 月：</strong> 加入北京大学，开展循环系统在线学习与全局信用分配研究。</li>
  <li><strong>2024 年 12 月：</strong> 加入中国科学院自动化研究所开展科研实习。</li>
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
  <a class="link-pill" href="/zh/publications/">查看完整论文列表</a>
</div>

<span class='anchor' id='experience'></span>

## 研究经历

<div class="timeline-card">
  <h3>中国科学院自动化研究所</h3>
  <p class="timeline-meta">科研实习生 | 2024 年 12 月至今 | 北京</p>
  <p>主要研究无线通信中的决策型大语言模型、协同边缘缓存以及长上下文恢复。近期工作既包括算法设计，也包括基于监督微调、GRPO、LoRA/QLoRA、TRL 和 Unsloth 的完整训练与评测流程搭建。</p>
</div>

<div class="timeline-card">
  <h3>北京大学</h3>
  <p class="timeline-meta">科研实习生 | 2025 年 10 月至今 | 北京</p>
  <p>围绕循环与脉冲系统中的时序信用分配问题开展研究，关注如何设计内存开销更低、更加符合生物机制的在线学习规则，并探索临界动力学对长程序列学习稳定性的作用。</p>
</div>

<div class="timeline-card">
  <h3>大连理工大学</h3>
  <p class="timeline-meta">本科科研 | 2024 年 10 月至 2025 年 3 月 | 大连</p>
  <p>独立主导取送货问题上的神经组合优化研究，完成模型设计、强化学习训练、基准构建、实验分析与论文撰写。</p>
</div>

<span class='anchor' id='education'></span>

## 教育背景

<div class="timeline-card">
  <h3>大连理工大学</h3>
  <p class="timeline-meta">数理基础科学本科生 | 2023 年 9 月至 2027 年 6 月（预计） | 大连</p>
  <p>GPA: 4.22/5.00 | 专业排名: 12/102</p>
  <p><strong>主要课程：</strong>数学分析（99）、高等代数（99）、概率论（99）、数据结构与算法（99）、最优化方法（99）、程序设计与算法（100）、解析几何（100）</p>
  <p><strong>英语成绩：</strong>CET-6 460，CET-4 522</p>
</div>

## 技术技能

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

## 学术服务

<div class="timeline-card">
  <h3>审稿服务</h3>
  <p class="timeline-meta">Journal on Wireless Communications and Networking 审稿人 | 中科院大类 4 区 | JCR Q2/Q3</p>
  <p>参与无线通信与网络方向稿件评审。</p>
</div>

<span class='anchor' id='honors'></span>

## 荣誉奖项

- **2024 年 12 月：** 大连理工大学学业优秀奖学金（前 15%）
- **2025 年 11 月：** 大连理工大学学业优秀奖学金（前 15%）
- **2024 年 9 月：** 全国大学生数学建模竞赛省一等奖
- **2024 年 11 月：** 亚太地区大学生数学建模竞赛国家一等奖
- **2024 年 2 月：** 美国大学生数学建模竞赛 Honorable Mention
- **2025 年 11 月：** 全国英语翻译大赛国家二等奖

## 活动经历

- **2023 年：** 大连理工大学新生篮球赛主力队员、亚军
- **2024 年 2 月：** 新加坡国立大学暑期学术交流项目参与者
