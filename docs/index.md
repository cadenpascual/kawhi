---
layout: default
title: Identifying Topological Defensive Signatures
---

<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']]
    }
  };
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Identifying "Topological Defensive Signatures" in the NBA via Gromov-Wasserstein

**Authors:** Takafumi Matsui, Caden Pascual, Rayan Saab, Alex Cloninger  
**Institution:** UC San Diego (DSC180B)

---

## 1. Abstract

This project develops a unified geometric framework for modeling NBA team defense as a dynamic, spatially organized system rather than a collection of isolated events. Leveraging spatiotemporal tracking data from the SportVU era, we adopt a “team-as-entity” perspective in which defensive behavior emerges from structured interactions among players.

At the micro level, we construct a physics-inspired potential energy model that governs defender positioning through cohesion forces, occupancy penalties, collective ball pressure, anisotropic offender attraction fields, basket anchoring, and boundary constraints. These interacting potentials produce coordinated defensive formations that adapt to ball movement and offensive threat levels.

At the macro level, we represent resulting defensive configurations as spatial metric-measure structures and compare teams using Gromov-Wasserstein optimal transport, enabling topology-aware similarity measurement independent of absolute court alignment. By integrating dynamical systems modeling with geometric optimal transport, this work establishes a principled approach to uncovering latent defensive archetypes and formally characterizing what we define as a team’s “topological defensive signature.”
---

## 2. Methodology: The Potential Energy Model

Our model assumes that defensive positioning is governed by a series of interacting potential fields that push and pull defenders into optimal configurations.

![NBA Potential Fields](assets/potential_fields_diagram.png)
*(Placeholder: Upload a plot from your tracking notebooks showing a frame of players with their potential field heatmaps)*

### 2.1 Basket Anchor
This potential attracts defenders towards the basket, acting as a "safe home" or a baseline defensive position. It creates a strong pull as the defender gets further away from the basket.

$$E_{gravity}(p_{d})=-W_{bg}\cdot e^{-\frac{||p_{d}-p_{b}||^{2}}{2\sigma_{bg}^{2}}}$$

**Where:**
* **$P_{d}$**: Current position of the defender.
* **$p_{b}$**: Position of the basket.
* **$W_{bg}$**: Weight of the basket gravity potential.
* **$\sigma_{bg}$**: Standard deviation, controlling the range/spread of the basket's influence.

### 2.2 Boundary Penalty Potential
This potential penalizes defenders for getting too close to the court boundaries, ensuring they stay within playable areas. It acts as a "soft spring" pushing them away from the edges.

For each defender $p_{d}=(x,y)$:

$$E_{boundary}(p_{d})=W_{boundary}\cdot[\max(0,B-x)^{2}+\max(0,B-(X_{court}-x))^{2}+\max(0,B-y)^{2}+\max(0,B-(Y_{court}-y))^{2}]$$

---

## 3. Instantaneous Shot Threat (IST) and Expected Field Goal (xFG)

To quantify defensive effectiveness, we calculate the Instantaneous Shot Threat (IST) using the features derived from our tracking data. As defined in our pipeline, IST is the product of Shot Quality ($Q$), Openness ($O$), Shootability ($S$), and a Ball Factor ($B$).

$$IST = Q \cdot O \cdot S \cdot B$$

By computing this alongside Expected Field Goal percentage ($xFG$), we can map exactly how a team's topological signature suppresses offensive efficiency.

![xFG vs IST Distribution](assets/xfg_distribution.png)
*(Placeholder: Export a plot from `03_calculate_xfg.ipynb` showing the relationship or distribution of xFG across different defensive alignments)*

---

## 4. Results: Topological Signatures via Gromov-Wasserstein

By mapping these defensive formations as spatial point clouds, we applied Gromov-Wasserstein optimal transport to measure the structural similarity between different possessions, regardless of their absolute rotation or translation on the court.

![Gromov-Wasserstein Transport Map](assets/gw_transport_map.png)
*(Placeholder: Insert a visualization showing the optimal transport mapping lines between two different team defensive setups)*

### 4.1 Defensive Archetypes
*(Insert text summarizing the distinct defensive clusters/archetypes you discovered in the data)*

---

## 5. Conclusion

By integrating dynamical systems modeling with geometric optimal transport, we can mathematically classify NBA defenses beyond traditional box score metrics. This topological approach opens new avenues for scouting, allowing teams to identify opponents with similar defensive DNA and optimize offensive strategies accordingly.