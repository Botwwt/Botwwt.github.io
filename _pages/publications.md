---
permalink: /publications/
title: "Publications"
excerpt: ""
lang: en
lang_switch_label: "中文"
lang_switch_url: "/zh/publications/"
author_profile: false
profile_cover: true
author_bio: "B.S. student in Foundational Mathematical Sciences at Dalian University of Technology"
sidebar_intro: "Take joy in discovery, stay true to what matters. Remain curious in exploration and reflective in work."
---

# Publications and Manuscripts

<p class="page-lead">This page collects my publications, accepted papers, public preprints, and ongoing manuscripts. The COLA/ICML 2026 paper is listed first as requested.</p>

<div class="link-pills">
  <a class="link-pill" href="/">Homepage</a>
  <a class="link-pill" href="https://scholar.google.com/citations?user=tF1l1S0AAAAJ&hl=zh-CN">Google Scholar</a>
  <a class="link-pill" href="https://github.com/Botwwt">GitHub</a>
  <a class="link-pill" href="https://www.linkedin.com/in/%E6%96%87%E9%9F%AC-%E7%8E%8B-235ab637b/">LinkedIn</a>
</div>

<p class="section-note">{{ site.data.publications.quartile_note.en }}</p>

{% assign publications = site.data.publications.items %}

## Publications and Preprints

{% for publication in publications %}
  {% if publication.section == 'journal' %}
    {% include publication-card.html publication=publication lang='en' variant='full' %}
  {% endif %}
{% endfor %}

## Manuscripts in Progress

{% for publication in publications %}
  {% if publication.section == 'ongoing' %}
    {% include publication-card.html publication=publication lang='en' variant='full' %}
  {% endif %}
{% endfor %}
