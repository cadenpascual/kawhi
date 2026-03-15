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

### 🧪 The 3-Minute Capstone Demo (Recommended for Reviewers)
To evaluate the Wasserstein physics engine and feature pipeline without processing 5 hours of 82-game tracking data, use the `--demo` flags. This routes the pipeline to a lightweight 1-game subset (`01.18.2016 GSW at CLE`).

**Step 1: Compute Player Spatial Maps**
Generates the foundational xPPS (Expected Points Per Shot) spatial maps for the offensive players based on the demo subset.
```bash
python -m src.pipelines.compute_player_maps
```

**Step 2: Compute Trajectories & Features**
Parses the compressed SportVU tracking data, synchronizes it with play-by-play logs, and extracts the tracking trajectories of all 10 players.
```bash
python -m src.pipelines.compute_real_traj --demo
```

**Step 3: JKO Defensive Optimization (Threat Tuning)**
Runs the core JAX-based Optimal Transport physics engine. It trains on a subset of the demo plays to find the optimal balance between Threat Reduction (IST) and Kinematic Smoothness. 
```bash
python src/gradient_flows/optimize.py --demo --trials 30
```

> **Flag Guide:**
> * `--demo`: Forces the script to use the lightweight datasets in `data/demo/`.
> * `--trials [int]`: (Optional) Controls how many Bayesian search iterations Optuna runs. Default is 20, but you can increase it (e.g., 50 or 100) for a more refined Pareto Front.

---

### 🌍 Full Season Execution (For Complete Analysis)
If you want to run the pipeline on the complete raw 7z archives and execute the full 100-trial optimization across the entire dataset, simply omit the `--demo` flag.

**Step 1: Compute Full Player Spatial Maps**
```bash
python -m src.pipelines.compute_player_maps
```

**Step 2: Compute Full Trajectories**
*(Note: Depending on your CPU cores, this multiprocessing step takes roughly 1-2 hours).*
```bash
python -m src.pipelines.compute_real_traj
```

**Step 3: Full Threat Optimization**
```bash
python src/gradient_flows/optimize.py --trials 100
```

---

## 🧠 Methodology & Notes

### Defensive xFG & IST
We model defense not as a direct predictor of missed shots, but as a contextual modifier of expected shot quality. The learned defense feature represents the **marginal risk induced by giving a shooter additional space**, enabling defender-specific guarding decisions rather than raw outcome prediction.

*Note on Pre-Shot Application:* Our xFG (Expected Field Goal) and xPPS (Expected Points Per Shot) models were trained at shot time, but we utilize them pre-shot. This is valid because we interpret our Initial Spatial-Temporal (IST) metric as a latent, continuous threat, rather than a literal immediate probability of a shot.

### Current Limitations (Deficiencies)
Because our current "impact" metric is shot-based, it currently does not capture:
* **Playmaking Gravity:** Passing and advantage creation.
* **Screening:** Off-ball screening value and off-ball gravity.
* **Foul-drawn Rim Pressure:** Missed shots that result in fouls are traditionally omitted from NBA shot charts, masking true rim-pressure value.

### Core Variables Used
The modeling utilizes the following extracted features:
`GAME_ID`, `SHOT_EVENT_ID`, `tracking_event_id`, `release_frame_idx`, `event_list_idx`, `PERIOD`, `game_clock`, `PLAYER_ID`, `TEAM_ID`, `x_ft`, `y_ft`, `xFG_offense`, `xPPS_offense`, `SHOT_MADE_FLAG`, `close_def_dist_release`, `closest_def_dist`, `close_def_id`, `num_defenders_tracked`, `w0_close_def_dist_mean`, `w0_close_def_dist_min`, `w0_shooter_speed_mean`, `w0_shooter_speed_max`, `w0_def_speed_mean`, `w0_closing_speed_mean`, `w1_close_def_dist_mean`, `w1_close_def_dist_min`, `w1_shooter_speed_mean`, `w1_shooter_speed_max`, `w1_def_speed_mean`, `w1_closing_speed_mean`, `shooter_speed`, `game_clock_tracking`, `shot_clock_tracking`, `w0_shooter_accel_mean`, `w1_shooter_accel_mean`, `Real_IST`, `Real_Q`, `Real_O`, `Real_S`, `Empirical_IST`

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

### `notebooks/`
Jupyter notebooks detailing the data science pipeline:
* `01_data_exploration.ipynb` - Initial EDA.
* `02_label_events.ipynb` - Labeling specific basketball events.
* `03_calculate_xfg.ipynb` - Calculating xFG probabilities.
* `04_defensive_features.ipynb` - Extracting spatial features for defenders.
* `05_bigdata_pipeline.ipynb` - Scaling the data processing pipeline.
* `06` & `07_calculate_ist...` - Computing IST distributions.
* `08_optimize_variables.ipynb` - Preparing boundary variables.
* `09_run_simulated_defense.ipynb` - Core Wasserstein Gradient Flow simulations.
* `10` & `11_agg_data...` - Post-simulation aggregations.
* `12_report_visualizations.ipynb` - Final plots and figures.

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
