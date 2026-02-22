---
layout: default
title: Identifying Topological Defensive Signatures
---

<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Identifying "Topological Defensive Signatures" in the NBA via Gromov-Wasserstein

**Authors:** Takafumi Matsui, Caden Pascual, Rayan Saab, Alex Cloninger  
**Institution:** UC San Diego (DSC180B)

---

## 1. Abstract / Overview

This project develops a unified geometric framework for modeling NBA team defense as a dynamic, spatially organized system rather than a collection of isolated events. Leveraging spatiotemporal tracking data from the SportVU era, we adopt a "team-as-entity" perspective in which defensive behavior emerges from structured interactions among players. 

At the micro level, we construct a physics-inspired potential energy model that governs defender positioning through cohesion forces, occupancy penalties, collective ball pressure, anisotropic offender attraction fields, basket anchoring, and boundary constraints. These interacting potentials produce coordinated defensive formations that adapt to ball movement and offensive threat levels. 

At the macro level, we represent resulting defensive configurations as spatial metric-measure structures and compare teams using Gromov-Wasserstein optimal transport, enabling topology-aware similarity measurement independent of absolute court alignment. By integrating dynamical systems modeling with geometric optimal transport, this work establishes a principled approach to uncovering latent defensive archetypes and formally characterizing what we define as a team's "topological defensive signature."

---

## 2. Methodology: The Potential Energy Model

Our model assumes that defensive positioning is governed by a series of interacting potential fields. Below are the core components of this system:

*(Placeholder for 5.1 Cohesion, 5.2 Occupancy, and 5.3 Offender Attraction Fields)*

### 5.4 Basket Anchor
This potential attracts defenders towards the basket, acting as a "safe home" or a baseline defensive position. It's an exponential function, creating a strong pull as the defender gets further away from the basket.

$$E_{gravity}(p_{d})=-W_{bg}\cdot e^{-\frac{||p_{d}-p_{b}||^{2}}{2\sigma_{bg}^{2}}}$$

**Where:**
* **$P_{d}$**: Current position of the defender.
* **$p_{b}$**: Position of the basket.
* **$W_{bg}$**: Weight of the basket gravity potential.
* **$\sigma_{bg}$**: Standard deviation, controlling the range/spread of the basket's influence.

### 5.5 Boundary Penalty Potential
This potential penalizes defenders for getting too close to the court boundaries, ensuring they stay within playable areas. It acts as a "soft spring" pushing them away from the edges.

For each defender $p_{d}=(x,y)$:

$$E_{boundary}(p_{d})=W_{boundary}\cdot[\max(0,B-x)^{2}+\max(0,B-(X_{court}-x))^{2}+\max(0,B-y)^{2}+\max(0,B-(Y_{court}-y))^{2}]$$

**Where:**
* **B (2.0)**: Buffer distance from the boundary.
* **$X_{court}$ (94)**: Court width.
* **$Y_{court}$ (50)**: Court height.
* **$W_{boundary}$ (10.0)**: Weight of the boundary penalty.

---

## 3. Results

*(Placeholder: We will add your Gromov-Wasserstein optimal transport results, visual maps, and defensive archetype findings here).*

---

## 4. Conclusion

*(Placeholder: We will summarize how characterizing these defensive signatures can impact NBA scouting and defensive strategy).*