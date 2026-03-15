import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from src.data_io.maps import save_maps_npz

# Import our custom modules
from src.features.shots.geometry import normalize_coordinates, filter_halfcourt, add_spatial_features
from src.features.shots.shot_maps import build_player_maps 

# remove warnings for future use
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_player_priors(train_df, full_df, alpha=300, min_atts=30):
    """Applies empirical Bayes smoothing to calculate player offensive impact."""
    grp_cols = ["PLAYER_ID", "SHOT_ZONE_BASIC", "is_three"]
    league_rate = train_df["SHOT_MADE_FLAG"].mean()

    grp = (
        train_df.groupby(grp_cols)["SHOT_MADE_FLAG"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "makes", "count": "atts"})
        .reset_index()
    )

    # Bayesian smoothing
    grp["player_prior"] = (grp["makes"] + alpha * league_rate) / (grp["atts"] + alpha)
    grp.loc[grp["atts"] < min_atts, "player_prior"] = league_rate

    # Merge back into full dataset
    full_df = full_df.drop(columns=["player_prior"], errors="ignore")
    full_df = full_df.merge(grp[grp_cols + ["player_prior"]], on=grp_cols, how="left")
    full_df["player_prior"] = full_df["player_prior"].fillna(league_rate)
    
    return full_df

def run_xfg_pipeline(input_path, output_parquet, output_map_npz):
    """Full ETL and modeling pipeline for Expected Field Goals."""
    print("[*] Loading raw shots...")
    shots = pd.read_parquet(input_path)
    
    # 1. Feature Engineering
    print("[*] Engineering geometric features...")
    shots = normalize_coordinates(shots)
    shots = filter_halfcourt(shots)
    shots = add_spatial_features(shots)
    
    shots['SHOT_ZONE_BASIC'] = shots['SHOT_ZONE_BASIC'].astype(str)
    shots['GAME_ID'] = shots['GAME_ID'].astype(int)
    shots["GAME_DATE"] = pd.to_datetime(shots["GAME_DATE"].astype(str), format="%Y%m%d", errors="coerce")
    
    # 2. Train/Val/Test Splits
    print("[*] Splitting data by date...")
    train_idx = shots.index[shots["GAME_DATE"] < "2016-01-01"]
    val_idx   = shots.index[(shots["GAME_DATE"] >= "2016-01-01") & (shots["GAME_DATE"] < "2016-02-15")]
    
    feature_cols_num = ["shot_dist", "angle", "is_three", "is_corner"]
    feature_cols_cat = ["SHOT_ZONE_BASIC"]
    
    X_train = shots.loc[train_idx, feature_cols_num + feature_cols_cat]
    y_train = shots.loc[train_idx, "SHOT_MADE_FLAG"].astype(int)
    X_val   = shots.loc[val_idx, feature_cols_num + feature_cols_cat]
    y_val   = shots.loc[val_idx, "SHOT_MADE_FLAG"].astype(int)
    
    # 3. Base Model (Location Only)
    print("[*] Training Base Isotonic Logistic Regression...")
    prep = ColumnTransformer([
        ("num", StandardScaler(), feature_cols_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
    ])
    
    base_model = Pipeline([
    ("prep", prep),
    ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=200))
    ])

    base_model.fit(X_train, y_train)
    
    calibrator = CalibratedClassifierCV(estimator=base_model, method="isotonic")
    calibrator.fit(X_val, y_val)

    print("\n--- MODEL EVALUATION ---")
    p_val  = calibrator.predict_proba(X_val)[:, 1]
    
    val_log_loss = log_loss(y_val, p_val)
    val_brier = brier_score_loss(y_val, p_val)
    
    print(f"Validation Log Loss: {val_log_loss:.4f}")
    print(f"Validation Brier Score: {val_brier:.4f}")
    print("------------------------\n")
    
    X_all = shots[feature_cols_num + feature_cols_cat].copy()
    shots["xFG_base"]  = calibrator.predict_proba(X_all)[:,1]
    
    # 4. Offense Model (Location + Player Identity)
    print("[*] Calculating Empirical Bayes Player Priors...")
    shots = calculate_player_priors(shots.loc[train_idx].copy(), shots)
    
    print("[*] Training Player-Adjusted Offense Model...")
    offense_features = ["xFG_base", "player_prior", "is_three"]
    
    X2_train = shots.loc[train_idx, offense_features]
    X2_val   = shots.loc[val_idx, offense_features]
    
    offense_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=500))
    ])
    offense_model.fit(X2_train, y_train)
    
    offense_cal = CalibratedClassifierCV(offense_model, method="isotonic", cv="prefit")
    offense_cal.fit(X2_val, y_val)
    
    shots["xFG_offense"] = offense_cal.predict_proba(shots[offense_features])[:, 1]
    shots["xPPS_offense"] = shots["xFG_offense"] * np.where(shots["is_three"], 3, 2)
    
    # 5. Generate Maps
    print("[*] Building spatial player maps...")
    maps, meta = build_player_maps(
        shots,
        min_attempts=75,
        value_col="xPPS_offense",
        grid_kwargs=dict(x_min=-25, x_max=25, y_min=-5, y_max=42, bin_size=1.0),
        smooth_sigma=1.25
    )
    # Save the npz array map
    save_maps_npz(output_map_npz, maps)
    
    # 6. Format and Save Parquet
    shots["game_clock"] = (shots["MINUTES_REMAINING"] * 60 + shots["SECONDS_REMAINING"])
    shots['SHOT_EVENT_ID'] = shots['GAME_EVENT_ID']
    
    relevant_columns = [
        'GAME_ID', 'SHOT_EVENT_ID', 'PERIOD', 'game_clock', 'PLAYER_ID', 
        'TEAM_ID', 'x_ft', 'y_ft', 'xFG_offense', 'xPPS_offense', 'SHOT_MADE_FLAG'
    ]
    
    shots_filtered = shots[relevant_columns].copy()
    shots_filtered.to_parquet(output_parquet, index=False)
    print(f"[✓] Pipeline complete! Saved {shots_filtered.shape[0]} shots to {output_parquet}")

if __name__ == "__main__":
    run_xfg_pipeline(
        input_path="data/raw/shots/league_shots_2015-16.parquet",
        output_parquet="data/demo/processed/all_season_shots.parquet",
        output_map_npz="data/demo/processed/maps_1ft_xpps.npz"
    )