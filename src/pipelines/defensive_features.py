import pandas as pd
import json
import os
import glob
import py7zr
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- IMPORT YOUR MODULES ---
# Adjust these imports to match your folder structure if needed
from src.processing.indexing import build_tracking_time_index
from src.features.defense.computing_features import compute_defense_features_for_shots_refactored
from src.processing.sportvu_to_events import convert_events

# --- CONFIGURATION ---
RAW_DATA_DIR = "../data/raw/7z/"  # Folder containing your .7z files
PROCESSED_DIR = "../data/processed/def_features/"
ALL_SHOTS_PARQUET = "../data/processed/shots/all_season_shots.parquet"
MAX_WORKERS = 3  # Adjust based on your CPU cores

def process_single_game_archive(file_path, all_shots_df):
    """
    Worker function that uses the SOURCE FILENAME for the output name.
    """
    file_path = Path(file_path)
    
    # 1. Extract the "Stem" (filename without .7z or .json extension)
    # Example: "01.01.2016.DAL.at.MIA"
    file_stem = file_path.stem 
    
    try:
        # --- A. LOAD DATA (Handle 7z or JSON) ---
        raw_json_data = None
        
        if file_path.suffix == ".7z":
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    with py7zr.SevenZipFile(file_path, mode='r') as z:
                        z.extractall(path=tmp_dir)
                    extracted_files = glob.glob(os.path.join(tmp_dir, "*.json"))
                    if not extracted_files: return f"Skipped: No JSON found inside {file_path.name}"
                    with open(extracted_files[0], "r") as f:
                        raw_json_data = json.load(f)
                except Exception as e:
                    return f"7z Extraction Error on {file_path.name}: {e}"

        elif file_path.suffix == ".json":
            with open(file_path, "r") as f:
                raw_json_data = json.load(f)
        
        if raw_json_data is None: return f"Skipped: No data loaded for {file_path.name}"

        # --- B. CONVERT RAW TO STANDARD FORMAT ---
        tracking_events = convert_events(raw_json_data)
        if not tracking_events: return f"Skipped: No events found in {file_path.name}"
        
        game_id = tracking_events[0]["gameid"]
        
        # --- C. DEFINE OUTPUT FILENAME ---
        # We combine the original filename with the Game ID for safety.
        # Output: defense_01.01.2016.DAL.at.MIA_21500492.parquet
        output_filename = f"defense_{file_stem}_{game_id}.parquet"
        out_file = Path(PROCESSED_DIR) / output_filename
        
        if out_file.exists():
            return f"Skipped: {output_filename} already exists."

        # --- D. FILTER SHOTS ---
        shots_game = all_shots_df[all_shots_df["GAME_ID"] == game_id].copy()
        if shots_game.empty:
            return f"Skipped: No shots found in DB for {game_id}"

        # --- E. COMPUTE FEATURES ---
        event_index = build_tracking_time_index(tracking_events)
        
        def_feats_df = compute_defense_features_for_shots_refactored(
            shots_g=shots_game,
            tracking_events=tracking_events,
            event_index=event_index,
            verbose_summary=False,
            include_accel=True, 
            include_game_shot_clock=True 
        )
        
        # --- F. SAVE ---
        if not def_feats_df.empty:
            def_feats_df.to_parquet(out_file, index=False)
            return f"Success: {output_filename} ({len(def_feats_df)} shots)"
        else:
            return f"Warning: {output_filename} processed but 0 features."

    except Exception as e:
        return f"CRITICAL ERROR on {file_path.name}: {str(e)}"

# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================

def run_season_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print(f"Loading shots from: {ALL_SHOTS_PARQUET}")
    try:
        all_shots = pd.read_parquet(ALL_SHOTS_PARQUET)
        # Ensure GAME_ID is int for matching
        all_shots["GAME_ID"] = all_shots["GAME_ID"].astype(int)
    except Exception as e:
        print(f"Critial Error: Could not load shots file. {e}")
        return

    # Look for both 7z and json files
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.7z")) + \
            glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))
            
    print(f"Found {len(files)} files to process in {RAW_DATA_DIR}")

    # Use Multiprocessing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit jobs
        futures = {
            executor.submit(process_single_game_archive, f, all_shots): f 
            for f in files
        }
        
        # Monitor progress
        total = len(files)
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"[{i+1}/{total}] {result}")

    print("\nPipeline Complete.")

if __name__ == "__main__":
    run_season_pipeline()