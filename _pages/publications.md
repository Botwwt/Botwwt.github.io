---
permalink: /publications/
title: "Publications"
excerpt: ""
lang: en
lang_switch_label: "中文"
lang_switch_url: "/zh/publications/"
author_profile: true
author_bio: "B.S. student in Foundational Mathematical Sciences at Dalian University of Technology"
sidebar_intro: "Undergraduate researcher working on large language models, reinforcement learning, and brain-inspired learning."
---

# Publications and Manuscripts

<p class="page-lead">This page collects my current publications and ongoing manuscripts. It includes public links when available and request-based access for submissions that remain under review.</p>

<div class="link-pills">
  <a class="link-pill" href="/">Back to Homepage</a>
  <a class="link-pill" href="https://scholar.google.com/citations?user=tF1l1S0AAAAJ&hl=zh-CN">Google Scholar</a>
  <a class="link-pill" href="https://github.com/Botwwt">GitHub</a>
</div>

## Journal Publications and Preprints

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">Applied Intelligence | Under Review</div>
      <img src="{{ '/images/publications/caadrl-pdp.png' | relative_url }}" alt="Cluster-aware attention model for pickup and delivery problems" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### [Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems]({{ '/files/papers/caadrl-pdp.pdf' | relative_url }})

**Wentao Wang**, Lifeng Han, Guangyu Zou  
First author | Dalian University of Technology | Oct. 2024 - Mar. 2025

- Proposes CAADRL, a cluster-aware deep reinforcement learning framework for pickup and delivery problems with clustered structure.
- Integrates global self-attention, intra-cluster attention, and a dynamic dual-decoder with a learnable gate.
- Achieves state-of-the-art performance on clustered benchmarks while reducing inference latency relative to neural collaborative-search baselines.

<div class="pub-links">
  <a href="{{ '/files/papers/caadrl-pdp.pdf' | relative_url }}">Paper</a>
  <a href="https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=tF1l1S0AAAAJ&citation_for_view=tF1l1S0AAAAJ:qjMakFHDy7sC">Scholar</a>
  <a href="https://github.com/Botwwt/CluPDTSP">Code</a>
</div>
  </div>
</div>

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">IEEE COMST | Published</div>
      <img src="{{ '/images/publications/wireless-llm-survey.png' | relative_url }}" alt="Survey of decision-making large language models for wireless communication" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### [Decision-Making Large Language Model for Wireless Communication: A Comprehensive Survey on Key Techniques](https://ieeexplore.ieee.org/document/11180008/)

Ning Yang, Mingrui Fan, **Wentao Wang**, Haijun Zhang  
Third author (student second) | Institute of Automation, Chinese Academy of Sciences | Dec. 2024 - Aug. 2025

- Surveys LLM-enabled decision-making for wireless communication across data, modeling, reasoning, inference, learning-based control, and multi-agent settings.
- I independently contributed the sections on data generation and augmentation, architecture adaptation, and open challenges.
- The paper was published in <em>IEEE Communications Surveys & Tutorials</em>.

<div class="pub-links">
  <a href="https://ieeexplore.ieee.org/document/11180008/">IEEE Xplore</a>
</div>
  </div>
</div>

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">IEEE TMC | Under Review</div>
      <img src="{{ '/images/publications/coop-llm-cache.jpg' | relative_url }}" alt="Cooperative edge caching with large language models" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### [Cooperative Edge Caching with Large Language Model in Wireless Networks](https://arxiv.org/abs/2602.13307)

Ning Yang, **Wentao Wang**, Lingtao Ouyang, Haijun Zhang  
Second author (student first) | Institute of Automation, Chinese Academy of Sciences | May 2025 - Nov. 2025

- Builds a multi-BS, multi-user cooperative edge-caching environment and formulates it as an LLM-native sequential decision problem.
- Implements a two-stage SFT+GRPO training pipeline with TRL, Unsloth, and QLoRA under a strictly validated text-to-action interface.
- Designs an opportunity-aware reward to quantify future cooperative hit-rate gains while penalizing infeasible actions.

<div class="pub-links">
  <a href="https://arxiv.org/abs/2602.13307">arXiv</a>
  <a href="https://github.com/gracefulning/CoopLLM-Cache">Code</a>
</div>
  </div>
</div>

## Manuscripts in Progress

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">ICML 2026 | Under Review</div>
      <img src="{{ '/images/publications/linearard.png' | relative_url }}" alt="LinearARD for long-context RoPE restoration" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### LinearARD: Linear-Memory Attention Distillation for RoPE Restoration

Second author (student second) | Institute of Automation, Chinese Academy of Sciences | Sep. 2025 - Dec. 2025

- Proposes a restoration-distillation framework that aligns row-wise dense self-relations between a native-RoPE teacher and a RoPE-scaled student.
- Derives an exact linear-memory KL kernel, avoiding explicit construction of full relation matrices on long sequences.
- On LLaMA2-7B scaled from 4K to 32K, the method recovers 98.3% of short-context performance using only 4.25M training tokens.

<div class="pub-links">
  <a href="mailto:shiyanxi1@mail.dlut.edu.cn?subject=LinearARD%20Preprint%20Request">Preprint upon request</a>
</div>
  </div>
</div>

<div class='paper-box'>
  <div class='paper-box-image'>
    <div>
      <div class="badge">ICML 2026 | Under Review</div>
      <img src="{{ '/images/publications/global-credit-criticality.png' | relative_url }}" alt="Global credit assignment via dynamical criticality" width="100%">
    </div>
  </div>
  <div class='paper-box-text' markdown="1">

### Global Credit Assignment via Dynamical Criticality

First author | Peking University | Oct. 2025 - Present

- Studies temporal credit assignment in recurrent and spiking neural systems through a criticality-driven online local learning rule.
- Uses long-range spatiotemporal correlations near the edge of chaos to derive a closed-form online teaching signal with constant activation memory.
- Combines theoretical analysis and benchmark experiments to study approximation error, stability, and scalability.

<div class="pub-links">
  <a href="mailto:shiyanxi1@mail.dlut.edu.cn?subject=Global%20Credit%20Assignment%20Preprint%20Request">Preprint upon request</a>
</div>
  </div>
</div>
