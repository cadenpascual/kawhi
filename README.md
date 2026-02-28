# Modeling NBA Team Dynamics via Wasserstein Gradient Flows

## Dataset Details
For this project we combined publicly available NBA datasets with original data processing,
alignment, and clustering methods.

### Data 


## 📊 Data Sources & References

### Primary Data Sources
These data sources were directly used or processed in this project.

- **NBA SportVU Tracking Data (2015–16)**  
  https://github.com/sealneaward/nba-movement-data  
  Raw player and ball tracking data used to construct per-play tracking events.

- **NBA Play-by-Play Data**  
  https://github.com/sumitrodatta/nba-alt-awards  
  Official NBA play-by-play logs used to label and align tracking events.

- **NBA Shot Chart Data / Stats API**  
  https://github.com/swar/nba_api  
  Used for shot context, shot locations, and auxiliary features.

### Related Research & Inspiration

- **Collective Motion and Team Structure in Professional Basketball**  
  *Scientific Reports* (2025).  
  https://www.nature.com/articles/s41598-025-04953-x  

  This work inspired our treatment of basketball teams as coordinated dynamical systems
  and informed our use of spatial structure and collective motion metrics. No data or code
  from this paper was used directly.


### Conceptual References & Benchmarks

- **Expected Field Goal Percentage (xFG%)**  
  https://www.nba.com/news/intro-to-expected-field-goal-percentage  

  This article was consulted as a conceptual benchmark for shot-quality metrics used in
  basketball analytics. While our project also models expected shooting outcomes, we do
  not replicate the NBA’s xFG% formulation, inputs, or methodology. Instead, we develop
  an independent expected-value framework based on tracking-derived defensive features.



## Repository Structure

```yaml
DSC180B_FinalProject/
│
├── data/
│   ├── raw/
│   │   ├── json/                # Raw SportVU game JSON (one file per game)
│   │   ├── 7z/                  # Compressed SportVU archives (optional)
│   │   └── 2015-16_pbp.csv      # League-wide play-by-play
│   │
│   └── processed/
│       ├── *_tracking_raw.json
│       └── *_labeled.json       # Tracking events with start_type labels
│
├── src/
│   ├── pipelines/
│   │   └── label_events.py      # End-to-end pipeline: raw → labeled events
│   │
│   ├── processing/
│   │   ├── pbp/
│   │   │   ├── restart_detection.py   # Detects play restarts (missed FT, TO, etc.)
│   │   │   └── alignment.py           # Aligns PBP rows to tracking by game clock
│   │   │
│   │   ├── tracking/
│   │   │   ├── cleaning.py            # Deduplication / normalization of tracking
│   │   │   └── summaries.py           # Event-level summaries (clock span, ball pos)
│   │   │
│   │   ├── indexing.py                # Builds time-based tracking index
│   │   └── play_start_classifier.py   # Classifies play start type
│   │
│   ├── data_io/
│   │   └── save_load.py               # Safe JSON load/save utilities
│   │
│   └── utils/
│       └── casting.py                 # safe_int, safe_float, time parsing helpers
│
├── notebooks/
│   └── cluster_events.ipynb           # Thin driver notebook for experimentation
│
└── README.md
```



## Notes
Defensive_XFG: We model defense not as a predictor of makes, but as a contextual modifier of expected shot quality. The learned defense feature represents the marginal risk induced by giving a shooter additional space, enabling defender-specific guarding decisions rather than outcome prediction.

Calcualting Player Effects on Shots
- α = 400 means “don’t believe player shooting skill in a zone until you’ve seen ~400 shots there,” which is exactly the conservative behavior you want for defense analysis.

Deficiencies

Your current “impact” is shot-based, so it won’t capture:
- playmaking gravity (passing, advantage creation)
- off-ball screening value
- rim pressure that creates fouls (missed fouled shots aren’t in shot charts)


Your xFG / xPPS models were trained at shot time, but you’re now using them pre-shot. That’s fine as long as you interpret IST as a latent threat, not a literal probability of a shot.
In other words:


Variables used: 
Index(['GAME_ID', 'SHOT_EVENT_ID', 'tracking_event_id', 'release_frame_idx',
       'event_list_idx', 'PERIOD', 'game_clock', 'PLAYER_ID', 'TEAM_ID',
       'x_ft', 'y_ft', 'xFG_offense', 'xPPS_offense', 'SHOT_MADE_FLAG',
       'close_def_dist_release', 'closest_def_dist', 'close_def_id',
       'num_defenders_tracked', 'w0_close_def_dist_mean',
       'w0_close_def_dist_min', 'w0_shooter_speed_mean',
       'w0_shooter_speed_max', 'w0_def_speed_mean', 'w0_closing_speed_mean',
       'w1_close_def_dist_mean', 'w1_close_def_dist_min',
       'w1_shooter_speed_mean', 'w1_shooter_speed_max', 'w1_def_speed_mean',
       'w1_closing_speed_mean', 'shooter_speed', 'game_clock_tracking',
       'shot_clock_tracking', 'w0_shooter_accel_mean', 'w1_shooter_accel_mean',
       'Real_IST', 'Real_Q', 'Real_O', 'Real_S', 'Empirical_IST'],
      dtype='object')

Variables filtered down to:
