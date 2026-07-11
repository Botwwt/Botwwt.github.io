---
permalink: /publications/
title: "Publications"
excerpt: ""
lang: en
lang_switch_label: "中文"
lang_switch_url: "/zh/publications/"
author_profile: false
profile_cover: true
author_bio: "B.S. Student in Foundational Mathematical Sciences"
profile_intro: "Publications, accepted papers, public preprints, and ongoing manuscripts across efficient learning and intelligent decision-making."
---

<div class="standalone-publications">
  <main class="content-shell">
    <section class="site-section publications-section">
      <div class="section-heading section-heading--actions">
        <div>
          <span class="section-index">Research Output</span>
          <h2>Publications</h2>
        </div>
        <div class="publication-filter" role="group" aria-label="Filter publications">
          <button type="button" data-filter="selected" aria-pressed="false">Selected</button>
          <button type="button" class="is-active" data-filter="all" aria-pressed="true">All</button>
        </div>
      </div>
      <p class="publications-intro">COLA, accepted at ICML 2026, is listed first. This page includes published work, accepted papers, public preprints, and manuscripts currently under review.</p>
      <p class="section-note">{{ site.data.publications.quartile_note.en }}</p>
      <div class="publication-list">
        {% for publication in site.data.publications.items %}
          {% include publication-card.html publication=publication lang='en' variant='full' %}
        {% endfor %}
      </div>
      <a class="section-more" href="{{ '/' | relative_url }}"><i class="fas fa-arrow-left" aria-hidden="true"></i> Back to homepage</a>
    </section>
  </main>
</div>