---
title: "ESL Notes"
sitemap: false
permalink: /Notes/ESL
---

<div id="notes"></div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<script>
  // MathJax config
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$']]
    }
  };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<script>
  fetch("https://raw.githubusercontent.com/Raihaen/Study-Notes/main/Stat%20learning/Elements-of-Statistical-Learning.md")
    .then(response => response.text())
    .then(text => {
      document.getElementById("notes").innerHTML = marked.parse(text);
      MathJax.typesetPromise(); // Render LaTeX after HTML
    })
    .catch(err => console.error("Failed to fetch notes:", err));
</script>