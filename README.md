# K.A.W.H.I. - Kinetic Adaptation via Wasserstein Heuristics and Identity

**Modeling NBA Team Dynamics via Wasserstein Gradient Flows**

This document provides a comprehensive overview of the `kawhi` repository. This project focuses on modeling NBA team dynamics via Wasserstein Gradient Flows, utilizing player tracking data, play-by-play data, and advanced spatial-temporal modeling to simulate and evaluate defensive positioning.

---

## 🛠️ Environment Setup Instructions

To ensure reproducibility, please follow these steps to set up the Python environment required to run this codebase. 

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/kawhi.git
   cd kawhi
   ```

2. **Create a virtual environment (Recommended: Conda or standard venv):**
   ```bash
   # Using standard Python venv
   python -m venv kawhi_env
   source kawhi_env/bin/activate  # On Windows use: kawhi_env\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Exact Commands to Run Experiments

The K.A.W.H.I. pipeline is broken down into specific modules. To reproduce our defensive feature extraction and Wasserstein Gradient Flow optimizations, run the following commands in order from the root directory:

**1. Data Parsing & Ingestion**
Extracts the raw 7z Second Spectrum tracking files and converts them into structured JSON events.
```bash
python src/scripts/7z_to_json.py
```

**2. Feature Engineering**
Generates the defensive spatial statistics and Initial Spatial-Temporal (IST) states.
```bash
python src/pipelines/defense_parquet.py
```

**3. Wasserstein Optimization Simulations**
Runs the core math/physics engine to simulate optimal defensive movements.
```bash
python src/gradient_flows/optimize.py
```

**4. Generate Final Visualizations**
To view the final spatial expected value maps, defensive pressure heatmaps, and results, execute the final visualization notebook:
```bash
jupyter nbconvert --to notebook --execute notebooks/12_report_visualizations.ipynb
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