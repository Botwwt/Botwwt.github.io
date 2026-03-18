---
permalink: /zh/publications/
title: "论文成果"
excerpt: ""
lang: zh
lang_switch_label: "English"
lang_switch_url: "/publications/"
author_profile: true
author_name: "王文涛"
author_bio: "大连理工大学数理基础科学本科生"
sidebar_intro: "本科阶段主要从事大语言模型、强化学习与类脑学习方向的研究。"
---

# 论文与在研稿件

<p class="page-lead">本页汇总我当前公开发表、公开预印本以及仍在审稿阶段的研究工作。对于尚未公开的稿件，页面保留按需索取方式。</p>

<div class="link-pills">
  <a class="link-pill" href="/zh/">返回主页</a>
  <a class="link-pill" href="https://scholar.google.com/citations?user=tF1l1S0AAAAJ&hl=zh-CN">谷歌学术</a>
  <a class="link-pill" href="https://github.com/Botwwt">GitHub</a>
</div>

<p class="section-note">{{ site.data.publications.quartile_note.zh }}</p>

{% assign publications = site.data.publications.items %}

## 期刊论文与公开预印本

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
