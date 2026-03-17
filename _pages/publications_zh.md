---
permalink: /zh/publications/
title: "论文成果"
excerpt: ""
lang: zh
lang_switch_label: "English"
lang_switch_url: "/publications/"
author_profile: true
author_name: "王文涛"
author_bio: "大连理工大学基础数学拔尖计划本科生"
sidebar_intro: "本科阶段主要从事大语言模型、强化学习与类脑学习方向的研究。"
---

# 论文与在研稿件

<p class="page-lead">本页汇总我当前公开发表、公开预印本以及仍在审稿阶段的研究工作。对于尚未公开的稿件，页面保留按需索取方式。</p>

<div class="link-pills">
  <a class="link-pill" href="/zh/">返回主页</a>
  <a class="link-pill" href="https://scholar.google.com/citations?user=tF1l1S0AAAAJ&hl=zh-CN">谷歌学术</a>
  <a class="link-pill" href="https://github.com/Botwwt">GitHub</a>
</div>

## 期刊论文与公开预印本

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">Applied Intelligence | 在审</div>
      <img src="{{ '/images/publications/caadrl-pdp.png' | relative_url }}" alt="Cluster-aware attention model for pickup and delivery problems" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### [Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems]({{ '/files/papers/caadrl-pdp.pdf' | relative_url }})

**Wentao Wang**, Lifeng Han, Guangyu Zou  
第一作者 | 大连理工大学 | 2024 年 10 月 - 2025 年 3 月

- 提出 CAADRL 框架，用于建模带簇结构的取送货路径优化问题。
- 将全局自注意力、簇内注意力与带门控的动态双解码器结合，实现宏观簇间决策与微观簇内决策协同。
- 在簇状基准上取得领先结果，同时显著降低相对神经协同搜索方法的推理开销。

<div class="pub-links">
  <a href="{{ '/files/papers/caadrl-pdp.pdf' | relative_url }}">论文</a>
  <a href="https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=tF1l1S0AAAAJ&citation_for_view=tF1l1S0AAAAJ:qjMakFHDy7sC">学术主页</a>
  <a href="https://github.com/Botwwt/CluPDTSP">代码</a>
</div>
  </div>
</div>

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">IEEE COMST | 已发表</div>
      <img src="{{ '/images/publications/wireless-llm-survey.png' | relative_url }}" alt="Survey of decision-making large language models for wireless communication" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### [Decision-Making Large Language Model for Wireless Communication: A Comprehensive Survey on Key Techniques](https://ieeexplore.ieee.org/document/11180008/)

Ning Yang, Mingrui Fan, **Wentao Wang**, Haijun Zhang  
第三作者（学生二作） | 中国科学院自动化研究所 | 2024 年 12 月 - 2025 年 8 月

- 系统梳理无线通信场景下 LLM 决策研究，覆盖数据、模型、推理、控制与多智能体协同等关键维度。
- 我主要负责数据生成与增强、架构适配以及开放问题等部分的撰写。
- 论文已发表于 <em>IEEE Communications Surveys & Tutorials</em>。

<div class="pub-links">
  <a href="https://ieeexplore.ieee.org/document/11180008/">IEEE Xplore</a>
</div>
  </div>
</div>

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">IEEE TMC | 在审</div>
      <img src="{{ '/images/publications/coop-llm-cache.jpg' | relative_url }}" alt="Cooperative edge caching with large language models" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### [Cooperative Edge Caching with Large Language Model in Wireless Networks](https://arxiv.org/abs/2602.13307)

Ning Yang, **Wentao Wang**, Lingtao Ouyang, Haijun Zhang  
第二作者（学生一作） | 中国科学院自动化研究所 | 2025 年 5 月 - 2025 年 11 月

- 构建多基站、多用户协同边缘缓存环境，并将任务表述为 LLM 原生的序列决策问题。
- 基于严格约束的文本到动作接口，实现 SFT+GRPO 两阶段训练流程，并结合 TRL、Unsloth 与 QLoRA。
- 设计机会感知奖励函数，用于刻画未来协同命中收益并惩罚不可行动作。

<div class="pub-links">
  <a href="https://arxiv.org/abs/2602.13307">arXiv</a>
  <a href="https://github.com/gracefulning/CoopLLM-Cache">代码</a>
</div>
  </div>
</div>

## 在研与投稿稿件

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">ICML 2026 | 在审</div>
      <img src="{{ '/images/publications/linearard.png' | relative_url }}" alt="LinearARD for long-context RoPE restoration" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### LinearARD: Linear-Memory Attention Distillation for RoPE Restoration

第二作者（学生二作） | 中国科学院自动化研究所 | 2025 年 9 月 - 2025 年 12 月

- 提出一种面向长上下文 LLM 的恢复蒸馏框架，对齐原生 RoPE 教师模型与缩放 RoPE 学生模型之间的行级自关系分布。
- 推导精确的线性内存 KL 核，避免显式构造完整关系矩阵。
- 在 LLaMA2-7B 从 4K 扩展到 32K 的设定下，仅用 4.25M 训练 token 即恢复 98.3% 的短上下文性能。

<div class="pub-links">
  <a href="mailto:shiyanxi1@mail.dlut.edu.cn?subject=LinearARD%20Preprint%20Request">按需索取预印本</a>
</div>
  </div>
</div>

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">ICML 2026 | 在审</div>
      <img src="{{ '/images/publications/global-credit-criticality.png' | relative_url }}" alt="Global credit assignment via dynamical criticality" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### Global Credit Assignment via Dynamical Criticality

第一作者 | 北京大学 | 2025 年 10 月至今

- 聚焦循环神经网络与脉冲神经系统中的时序信用分配问题，提出基于临界动力学的在线局部学习规则。
- 利用临界边缘的长程时空关联，推导具有常数激活内存开销的闭式在线教学信号。
- 结合理论分析与基准实验研究近似误差、稳定性与可扩展性。

<div class="pub-links">
  <a href="mailto:shiyanxi1@mail.dlut.edu.cn?subject=Global%20Credit%20Assignment%20Preprint%20Request">按需索取预印本</a>
</div>
  </div>
</div>
