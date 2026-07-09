---
permalink: /zh/publications/
title: "论文成果"
excerpt: ""
lang: zh
lang_switch_label: "English"
lang_switch_url: "/publications/"
author_profile: false
profile_cover: true
author_name: "王文韬"
author_bio: "大连理工大学数理基础科学本科生"
sidebar_intro: "乐于趣，敏于义。在探索中永葆好奇，在工作中常思反省。"
---

# 论文与在研稿件

<p class="page-lead">本页汇总我当前公开发表、已录用、公开预印本以及仍在审稿阶段的研究工作。COLA / ICML 2026 论文已按要求放在第一位。</p>

<div class="link-pills">
  <a class="link-pill" href="/zh/">主页</a>
  <a class="link-pill" href="https://scholar.google.com/citations?user=tF1l1S0AAAAJ&hl=zh-CN">谷歌学术</a>
  <a class="link-pill" href="https://github.com/Botwwt">GitHub</a>
  <a class="link-pill" href="https://www.linkedin.com/in/%E6%96%87%E9%9F%AC-%E7%8E%8B-235ab637b/">领英</a>
</div>

<p class="section-note">{{ site.data.publications.quartile_note.zh }}</p>

{% assign publications = site.data.publications.items %}

## 论文与公开预印本

{% for publication in publications %}
  {% if publication.section == 'journal' %}
    {% include publication-card.html publication=publication lang='zh' variant='full' %}
  {% endif %}
{% endfor %}

## 在研与投稿稿件

{% for publication in publications %}
  {% if publication.section == 'ongoing' %}
    {% include publication-card.html publication=publication lang='zh' variant='full' %}
  {% endif %}
{% endfor %}
