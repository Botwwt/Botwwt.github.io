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

<p class="section-note">{{ site.data.publications.quartile_note.en }}</p>

{% assign publications = site.data.publications.items %}

## Journal Publications and Preprints

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
