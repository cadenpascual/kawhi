import pandas as pd
import orjson  # <--- Much faster JSON parsing
import tempfile
import os
import glob
import py7zr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


# --- CONFIGURATION & ARGUMENTS ---
parser = argparse.ArgumentParser(description="Process NBA Tracking Data")
parser.add_argument("--demo", action="store_true", help="Run on the 1-game demo dataset")
args, _ = parser.parse_known_args()

# --- IMPORT YOUR MODULES ---
from src.data_io.maps import load_maps_npz
from src.processing.indexing import build_tracking_time_index
from src.features.traj.compute import build_defensive_configurations
from src.processing.sportvu_to_events import parse_sportvu_kinematics

# --- CONFIGURATION ---
if args.demo:
    print("[*] DEMO MODE ACTIVATED: Routing to data/demo/ folders...")
    RAW_DATA_DIR = "data/demo/raw/"
    PROCESSED_DIR = "data/demo/processed"
    ALL_SHOTS_PARQUET = "data/demo/raw/demo_shots.parquet"
    MAPS_PATH = "data/demo/processed/maps_1ft_xpps.npz"
else:
    print("[*] FULL SEASON MODE ACTIVATED: Routing to data/raw/ folders...")
    RAW_DATA_DIR = "data/raw/7z/"
    PROCESSED_DIR = "data/processed/traj_features"
    ALL_SHOTS_PARQUET = "data/processed/shots/all_season_shots.parquet"
    MAPS_PATH = "data/processed/shot_maps/maps_1ft_xpps.npz"

IS_DEMO = args.demo  # <--- ADD THIS LINE
MAX_WORKERS = max(1, os.cpu_count() - 1)  # Automatically use all but 1 of your CPU cores

# --- LOAD GLOBALS ONCE ---
# By loading these globally, child processes inherit them automatically 
# without heavy inter-process pickling overhead.
print(f"[*] Loading spatial maps from: {MAPS_PATH}")
maps, pid2row = load_maps_npz(MAPS_PATH)

print(f"[*] Loading shots from: {ALL_SHOTS_PARQUET}")
ALL_SHOTS_DF = pd.read_parquet(ALL_SHOTS_PARQUET)
ALL_SHOTS_DF["GAME_ID"] = ALL_SHOTS_DF["GAME_ID"].astype(int)

import tempfile # Make sure this is imported at the top

def process_single_game_archive(file_path):
    """Worker function optimized for memory and speed."""
    file_path = Path(file_path)
    file_stem = file_path.stem 
    
    try:
        raw_json_data = None
        
        # --- A. LOAD DATA (Hybrid Disk/Memory Approach) ---
        if file_path.suffix == ".7z":
            # Safely extract to a temp folder, but read with the lightning-fast orjson
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    with py7zr.SevenZipFile(file_path, mode='r') as z:
                        z.extractall(path=tmp_dir)
                    
                    extracted_files = glob.glob(os.path.join(tmp_dir, "*.json"))
                    if not extracted_files: 
                        return f"Skipped: No JSON found inside {file_path.name}"
                    
                    # Read as raw bytes for orjson
                    with open(extracted_files[0], "rb") as f:
                        raw_json_data = orjson.loads(f.read())
                except Exception as e:
                    return f"7z Extraction Error on {file_path.name}: {e}"

        elif file_path.suffix == ".json":
            with open(file_path, "rb") as f:
                raw_json_data = orjson.loads(f.read())
        
        if raw_json_data is None: return f"Skipped: No data loaded for {file_path.name}"

        # --- B. CONVERT DATA ---
        tracking_events = parse_sportvu_kinematics(raw_json_data)
        if not tracking_events: return f"Skipped: No events found in {file_path.name}"
        
        # Free up memory immediately to prevent RAM spikes
        del raw_json_data 
        
        game_id = int(tracking_events[0]["gameid"])
        
        # --- C. CHECK OUTPUT EXISTANCE ---
        if IS_DEMO:
            output_filename = "demo_traj.parquet"
        else:
            output_filename = f"traj_{file_stem}_{game_id}.parquet"
        
        out_file = Path(PROCESSED_DIR) / output_filename
        if out_file.exists():
            return f"Skipped: {output_filename} already exists."

        # --- D. FILTER SHOTS (Using Global DF) ---
        shots_game = ALL_SHOTS_DF[ALL_SHOTS_DF["GAME_ID"] == game_id].copy()
        if shots_game.empty:
            return f"Skipped: No shots found in DB for {game_id}"

        # --- E. COMPUTE FEATURES ---
        event_index = build_tracking_time_index(tracking_events)
        
        def_feats_df = build_defensive_configurations(
            shots_g=shots_game,
            tracking_events=tracking_events,
            event_index=event_index,
            maps_npz=maps,
            pid2row=pid2row,
            show_diagnostics= not IS_DEMO
        )
        
        # --- F. CLEAN AND SAVE ---
        if not def_feats_df.empty:
            initial_count = len(def_feats_df)
            
            # 1. Drop standard NaNs
            def_feats_df = def_feats_df.dropna()
            
            # 2. Deep check for NaNs hidden inside the trajectory arrays/lists
            import numpy as np
            def has_hidden_nans(row):
                for val in row.values:
                    if isinstance(val, (list, np.ndarray)):
                        if pd.isna(val).any():
                            return True
                    elif pd.isna(val):
                        return True
                return False
            
            # Apply the mask to keep only clean rows
            clean_mask = ~def_feats_df.apply(has_hidden_nans, axis=1)
            def_feats_df = def_feats_df[clean_mask].copy()
            
            final_count = len(def_feats_df)
            dropped = initial_count - final_count
            
            if final_count > 0:
                def_feats_df.to_parquet(out_file, index=False)
                drop_msg = f"(Dropped {dropped} NaN plays)" if dropped > 0 else "(100% clean)"
                return f"Success: {output_filename} ({final_count} shots) {drop_msg}"
            else:
                return f"Warning: {output_filename} processed but all plays contained NaNs."
        else:
            return f"Warning: {output_filename} processed but 0 features."

    except Exception as e:
        return f"CRITICAL ERROR on {file_path.name}: {str(e)}"

# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================

def run_season_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.7z")) + \
            glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))
            
    print(f"Found {len(files)} files to process in {RAW_DATA_DIR}")
    print(f"Running on {MAX_WORKERS} parallel workers...")

    # Use Multiprocessing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Note: We no longer pass 'all_shots' here!
        futures = {executor.submit(process_single_game_archive, f): f for f in files}
        
        total = len(files)
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"[{i+1}/{total}] {result}")

    print("\nPipeline Complete.")

if __name__ == "__main__":
    run_season_pipeline()
    if IS_DEMO:
        print(f"STEP 2 COMPLETE: Check QUICKSTART_DEMO.ipynb to verify trajectories")