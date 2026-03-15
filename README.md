# K.A.W.H.I. - Kinetic Adaptation via Wasserstein Heuristics and Identity

**Modeling NBA Team Dynamics via Wasserstein Gradient Flows**

This document provides a comprehensive overview of the `kawhi` repository. This project focuses on modeling NBA team dynamics via Wasserstein Gradient Flows, utilizing player tracking data, play-by-play data, and advanced spatial-temporal modeling to simulate and evaluate defensive positioning.

---

## 🔗 Project Deliverables
* **🌐 [Interactive Website](https://takafumim.github.io/Wasserstein-Gradient-Flows/)** – Explore our interactive models and spatial-temporal visualizations.
* **📄 [Final Technical Report](https://github.com/cadenpascual/kawhi-artifact-directory/blob/main/report.pdf)** – Detailed methodology, mathematical framework, and results analysis.
* **🖼️ [Research Poster](./docs/DSC180_Poster.pdf)** – A visual summary of our project's findings and Wasserstein Gradient Flow application.
* **📦 [Artifact Repository](https://github.com/cadenpascual/kawhi-artifact-directory)** – The centralized directory for all data artifacts and supplemental materials.


---

## 🛠️ Environment Setup Instructions

To ensure reproducibility, please follow these steps to set up the Python environment required to run this codebase. 

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cadenpascual/kawhi.git
   cd kawhi
   ```

2. **Create a virtual environment (Recommended: Conda):**
   ```bash
   # Using Conda (Preferred for Data Science packages like JAX & SciPy)
   conda create -n kawhi_env python=3.10 -y
   conda activate kawhi_env
   ```

   *Alternative using standard Python venv:*

      ```bash
      python -m venv kawhi_env
      source kawhi_env/bin/activate  # On Windows use: kawhi_env\Scripts\activate
      ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---
## 🚀 How to Run the Project Pipeline

To ensure the data is processing correctly, follow this iterative pipeline: **Maps ➔ Trajectories ➔ Optimization ➔ Simulation**. 

### 🧪 The 5-Minute Capstone Demo
To evaluate the Wasserstein physics engine and feature pipeline without processing 82 games of tracking data, use the `--demo` flags. This routes the pipeline to a 1-game subset (`01.18.2016 GSW at CLE`). 

**Step 1: Compute Player Spatial Maps**
Generates foundational xPPS (Expected Points Per Shot) maps for offensive players.
```bash
python -m src.pipelines.compute_player_maps --demo
```
> **🔍 Verification:** Open **`00_Quickstart_Demo.ipynb`** to verify that the spatial maps for key players (e.g., Stephen Curry, LeBron James) have loaded correctly.

**Step 2: Compute Trajectories & Features**
Extracts synchronized trajectories and defensive configurations.
```bash
python -m src.pipelines.compute_real_traj --demo
```
> **🔍 Verification:** Go back to the notebook and use the animation widget to view the **Real NBA Defense**. Ensure the player coordinates are properly aligned with the court boundaries.

**Step 3: JKO Defensive Optimization (Threat Tuning)**
Runs the JAX physics engine to find the optimal balance between Threat Reduction and Kinematic Smoothness. 
```bash
python src/gradient_flows/optimize.py --demo --trials 20
```
> **🔍 Verification:** Once the Pareto Front chart appears, identify the **Trial Number** that provides your desired balance of threat reduction and smoothness. You can view the parameters for this trial in the notebook's Optuna summary section.

**Step 4: Generate Final Simulations**
Applies your optimized parameters to the dataset to create the final simulated movements. Replace `12` with your chosen trial number.
```bash
python -m src.pipelines.compute_ist_traj --demo --trial 12
```

**Step 5: Final Interactive Audit**
Return to **`00_Quickstart_Demo.ipynb`**. The notebook will now detect your simulated data, allowing you to run side-by-side animations and statistical reports comparing the **Real Defense** to your **Optimal JKO Defense**.

---

### 🌍 Full Season Execution (For Complete Analysis)
To run the pipeline on the full 2015-16 season data, simply omit the `--demo` flags.

**Step 1: Compute Full Maps**
```bash
python -m src.pipelines.compute_player_maps
```

**Step 2: Compute Full Trajectories**
```bash
python -m src.pipelines.compute_real_traj
```

**Step 3: Full Season Optimization**
```bash
python src/gradient_flows/optimize.py --trials 100
```

**Step 4: Full Season Simulation Generation**
```bash
python -m src.pipelines.compute_ist_traj --trial [YOUR_BEST_TRIAL]
```
---
## 📊 Data Sources & References

### Primary Data Sources
We combined publicly available NBA datasets with original data processing, alignment, and clustering methods.
- **NBA SportVU Tracking Data (2015–16)**: [sealneaward/nba-movement-data](https://github.com/sealneaward/nba-movement-data) - Raw player/ball tracking data used to construct per-play tracking events.
- **NBA Shot Chart Data / Stats API**: [swar/nba_api](https://github.com/swar/nba_api) - Used for shot context, shot locations, and auxiliary features.

### Related Research & Inspiration
- **Collective Motion and Team Structure in Professional Basketball** (*Scientific Reports*, 2025): [Article Link](https://www.nature.com/articles/s41598-025-04953-x). This work inspired our treatment of basketball teams as coordinated dynamical systems. *(No data or code from this paper was used directly).*

### Conceptual References & Benchmarks
- **Expected Field Goal Percentage (xFG%)**: [NBA Intro to xFG%](https://www.nba.com/news/intro-to-expected-field-goal-percentage). Consulted as a conceptual benchmark. We do not replicate the NBA’s exact xFG% methodology; instead, we developed an independent expected-value framework based on tracking-derived defensive features.

---

## 📁 Directory Structure

### `data/`
Holds all data required and generated by the project. *(Note: Large files are ignored by git).*
* `raw/`: Unmodified datasets (Play-by-play logs, `.7z` tracking data, basic defense stats).
* `processed/`: Cleaned, transformed, and feature-engineered data (Parquet features, aggregated `.csv` IST tables, `.npz` shot maps, and SQLite `.db` optimization results).


### `src/` (Codebase Architecture)
Core Python source code modules organized by domain:
* `data_io/`: I/O handlers for parsing `.7z` archives and standardizing save/load formats.
* `data_sources/`: Scripts for pulling official `nba_api` data.
* `features/`: Deriving defensive spatial stats and Initial Spatial-Temporal (IST) states.
* `gradient_flows/`: The core math/physics engine using Wasserstein Gradient Flows.
* `metrics/`: Evaluation scripts to grade defensive success.
* `pipelines/` & `scripts/`: Executable files for end-to-end batch processing.
* `processing/`: Complex logic merging tracking frames with Play-by-Play (PBP) logs.
* `spatial/`: Matrices and dimension mapping for the court layout.
* `tracking/`: Data structures for ball/player movement and event detection.
* `viz/`: Generation of court visualizations and gradient flow animations.

### `images/` & `docs/`
Contains the generated charts (e.g., `harden_density.png`, expected FG heatmaps) and markdown files to render project reports.

## 🗺️ Forward Roadmap & Future Work

While the core Wasserstein Gradient Flow models and Expected Field Goal (xFG) pipelines are fully functional, we have identified several areas for future improvement:

- **[ ] Integrate Playmaking Gravity:** Current IST models are shot-based. Future iterations should incorporate pass-probability and advantage-creation metrics.
- **[ ] Off-Ball Screening Evaluation:** Expand the repulsive potentials in the gradient flow to account for off-ball screens and off-ball movement gravity.
- **[ ] Optimization Speed:** Migrate the core numerical solver in `solver.py` to leverage JAX's `vmap` and `@jit` decorators for faster simulation across full 82-game seasons.
- **[ ] Real-Time API:** Package the pipeline into a Flask/FastAPI endpoint for real-time defensive evaluations.

While the core JKO simulation demonstrates significant improvements in defensive positioning, we have identified several physical and structural constraints for future optimization:

* **[ ] Velocity Control (Speed Constraints):** Current simulated defenders occasionally surpass human speed limits to escape steep Gaussian gravity wells. Future work involves refining the kinetic penalty to enforce strict, realistic velocity caps.
* **[ ] Posture-Based Offender Threat:** Currently, defenders evaluate off-ball threats purely via distance to the ball. Future extensions will utilize Graph Convolutional Neural Networks (GCN) to adjust offensive potentials based on player momentum and bodily posture.
* **[ ] Heterogeneous Defensive Profiles:** The current model assumes defensive homogeneity. Future iterations will incorporate individual impact metrics (e.g., wingspan, lateral quickness) to differentiate between elite interior protectors (like Victor Wembanyama) and high-pressure perimeter defenders (like Alex Caruso).
