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
profile_intro: "汇总高效学习与智能决策方向的已发表论文、录用论文、公开预印本和在研稿件。"
---

<div class="standalone-publications">
  <main class="content-shell">
    <section class="site-section publications-section">
      <div class="section-heading section-heading--actions">
        <div>
          <span class="section-index">科研成果</span>
          <h2>论文成果</h2>
        </div>
        <div class="publication-filter" role="group" aria-label="筛选论文">
          <button type="button" data-filter="selected" aria-pressed="false">代表性论文</button>
          <button type="button" class="is-active" data-filter="all" aria-pressed="true">全部论文</button>
        </div>
      </div>
      <p class="publications-intro">COLA 已被 ICML 2026 录用并置于首位。本页汇总已发表、已录用、公开预印本以及当前在审稿件。</p>
      <p class="section-note">{{ site.data.publications.quartile_note.zh }}</p>
      <div class="publication-list">
        {% for publication in site.data.publications.items %}
          {% include publication-card.html publication=publication lang='zh' variant='full' %}
        {% endfor %}
      </div>
      <a class="section-more" href="{{ '/zh/' | relative_url }}"><i class="fas fa-arrow-left" aria-hidden="true"></i> 返回主页</a>
    </section>
  </main>
</div>