import os
# ==========================================
# 1. JAX MEMORY SAFETY SETTINGS (MUST BE FIRST)
# ==========================================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import math
import time
import pandas as pd
import jax
import jax.numpy as jnp
import gc
from pathlib import Path
from tqdm.notebook import tqdm
from nba_api.stats.endpoints import boxscoresummaryv2

# Import your external functions
from src.gradient_flows.utils import extract_trajectories_from_row

NBA_TEAMS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740, 
    'NO':  1610612740, 'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 
    'GSW': 1610612744, 'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 
    'MIA': 1610612748, 'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 
    'NJN': 1610612751, 'NYK': 1610612752, 'ORL': 1610612753, 'IND': 1610612754, 
    'PHI': 1610612755, 'PHX': 1610612756, 'PHO': 1610612756, 'POR': 1610612757, 
    'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760, 'TOR': 1610612761, 
    'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764, 'DET': 1610612765, 
    'CHA': 1610612766, 'CHO': 1610612766
}

def extract_teams_from_filename(filename):
    """
    Parses 'traj_01.01.2016.CHA.at.TOR_21500492.parquet' to get Team IDs.
    """
    try:
        middle_chunk = filename.split('_')[1]
        parts = middle_chunk.split('.')
        away_abbr = parts[-3].upper() 
        home_abbr = parts[-1].upper() 
        home_id = NBA_TEAMS.get(home_abbr)
        away_id = NBA_TEAMS.get(away_abbr)
        return home_id, away_id
    except Exception as e:
        print(f"Could not parse teams from filename {filename}: {e}")
        return None, None

# ==========================================
# 1. NBA API TEAM CACHE
# ==========================================
game_teams_cache = {}

def get_game_teams(game_id):
    """Fetches the Home and Visitor Team IDs for a given Game ID, cached to avoid API limits."""
    game_id_str = str(game_id).zfill(10)
    if game_id_str not in game_teams_cache:
        try:
            summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id_str).get_data_frames()[0]
            home_id = summary['HOME_TEAM_ID'].iloc[0]
            visitor_id = summary['VISITOR_TEAM_ID'].iloc[0]
            game_teams_cache[game_id_str] = (home_id, visitor_id)
            time.sleep(0.5) 
        except Exception as e:
            print(f"Failed to fetch teams for Game {game_id_str}: {e}")
            return None, None
    return game_teams_cache[game_id_str]

# ==========================================
# 2. METRIC EXTRACTION
# ==========================================
def get_play_ist(row, params):
    """Calculates IST metrics. A win is defined as Real IST < Sim IST."""
    (sim_def_no_ist_traj, sim_def_ist_traj, real_def_traj, 
     weights_sim_no_ist, weights_sim_ist, weights_real, 
     off_traj, ball_traj, basket_pos, off_ids, shot_frame) = extract_trajectories_from_row(row, params, pad_for_bulk=True)
    
    shot_idx = int(shot_frame)
    pos_idx = max(0, shot_idx - 1)
    
    # --- 1. COORDINATE & BASKET NORMALIZATION ---
    # We use 'flipped_coordinates' to determine the correct basket
    is_flipped = row.get('flipped_coordinates', 0)
    shooter_pos = off_traj[pos_idx, 0, :] # [x, y]
    
    # If coordinates aren't flipped and shooter is past half-court, use the far basket
    if is_flipped == 0 and shooter_pos[0] > 47:
        target_basket = jnp.array([90.0, 25.0])
    else:
        target_basket = jnp.array([4.0, 25.0])
        
    # --- 2. Corrected Distance Calculation ---
    dist_x = shooter_pos[0] - target_basket[0]
    dist_y = shooter_pos[1] - target_basket[1]
    shot_dist = float(jnp.sqrt(dist_x**2 + dist_y**2))
    
    # Standard 3-Point threshold
    is_three = 1 if shot_dist >= 23.0 else 0 
    
    # --- 3. IST Summation & Win Logic ---
    # Collapse 5 defenders into 1 Team Total
    if weights_real.ndim == 2:
        weights_real = jnp.sum(weights_real, axis=1)
        weights_sim_ist = jnp.sum(weights_sim_ist, axis=1)

    mask = jnp.arange(len(weights_real)) < shot_idx
    
    total_real = float(jnp.sum(jnp.where(mask, weights_real, 0.0)))
    total_sim = float(jnp.sum(jnp.where(mask, weights_sim_ist, 0.0)))
    
    # WIN CONDITION: Real IST > Sim IST (Defense outperformed the model)
    won_mask = mask & (weights_real > weights_sim_ist)
    frames_won = int(jnp.sum(won_mask))
    
    return total_real, total_sim, frames_won, shot_idx, float(weights_real[pos_idx]), float(weights_sim_ist[pos_idx]), shot_dist, is_three
# ==========================================
# 3. SINGLE GAME PROCESSING (WITH TEAM IDs)
# ==========================================
import traceback

def process_single_game_detailed(parquet_path, params):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return []

    if len(df) == 0: return []

    filename = parquet_path.name
    home_team_id, away_team_id = extract_teams_from_filename(filename)
    raw_game_id = df['GAME_ID'].iloc[0] if 'GAME_ID' in df.columns else filename

    game_plays = []
    dropped_plays = 0 

    for index, row in df.iterrows():
        try:
            # 1. Run the IST and Distance math
            results = get_play_ist(row, params)
            real_ist, sim_ist, f_won, f_total, pos_real, pos_sim, dist, is_3 = results
            
            # 2. Basic Validation & Casting
            real_ist, sim_ist = float(real_ist), float(sim_ist)
            if math.isnan(real_ist) or (real_ist == 0.0 and sim_ist == 0.0):
                dropped_plays += 1
                continue
                
            # 3. Defensive Team Logic
            off_team_id = row.get('TEAM_ID') 
            def_team_id = None
            if home_team_id and away_team_id and off_team_id:
                def_team_id = away_team_id if off_team_id == home_team_id else home_team_id

            # 4. Build the master dictionary
            play_data = {
                "Game_File": filename,
                "Play_Index": index,
                "GAME_ID": raw_game_id,
                "SHOT_EVENT_ID": row.get('SHOT_EVENT_ID'),
                "PERIOD": int(row.get('PERIOD', 0)),        # NEW: Period
                "GAME_CLOCK": row.get('game_clock'),        # NEW: Game Clock
                "Offensive_Team_ID": off_team_id,
                "Defensive_Team_ID": def_team_id,
                "Shooter_PID": int(row.get('off1_pid')) if not pd.isna(row.get('off1_pid')) else None,
                "Shot_Dist": round(float(dist), 2),
                "Is_Three": int(is_3),
                "Real_IST": round(real_ist, 2),
                "Sim_IST": round(sim_ist, 2),
                "Pressure_Prevented": round(real_ist - sim_ist, 2),
                "POS_Real_IST": round(float(pos_real), 2),
                "POS_Sim_IST": round(float(pos_sim), 2),
                "Frames_Won": int(f_won),
                "Total_Frames": int(f_total),
                "Frame_Win_Rate": round(float(f_won) / float(f_total), 4) if f_total > 0 else 0
            }
            
            # 5. Fix: Casting all Player IDs to Integers to avoid .0 float issue
            for i in range(1, 6):
                off_pid = row.get(f'off{i}_pid')
                def_pid = row.get(f'def{i}_pid')
                
                # Use int() only if the value isn't NaN/None
                if not pd.isna(off_pid):
                    play_data[f'Offender_{i}_PID'] = int(off_pid)
                if not pd.isna(def_pid):
                    play_data[f'Defender_{i}_PID'] = int(def_pid)

            if 'SHOT_MADE_FLAG' in df.columns: 
                play_data['Shot_Made'] = row.get('SHOT_MADE_FLAG')

            game_plays.append(play_data)
            
        except Exception:
            dropped_plays += 1
            continue 
            
    print(f"  -> {filename}: Extracted {len(game_plays)} (Dropped {dropped_plays})")
    return game_plays

# ==========================================
# 4. BULK PIPELINE RUNNER (MEMORY SAFE)
# ==========================================
def analyze_bulk_plays_safe(directory_path, params, output_csv="all_plays_ist_master.csv"):
    folder = Path(directory_path)
    parquet_files = list(folder.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {directory_path}")
        return None
        
    print(f"Found {len(parquet_files)} games. Building Play-Level Master Dataset...")
    all_plays_master_list = []
    
    for i, file_path in enumerate(tqdm(parquet_files, desc="Games Processed")):
        plays = process_single_game_detailed(file_path, params)
        
        if plays:
            all_plays_master_list.extend(plays) 
        
        # Intermittent Checkpoint Save (changed to every 5 games for safety)
        if (i + 1) % 5 == 0 and all_plays_master_list:
            pd.DataFrame(all_plays_master_list).to_csv(output_csv, index=False)
            
        # FORCE MEMORY CLEARING
        jax.clear_caches()
        gc.collect() 
            
    if all_plays_master_list:
        final_df = pd.DataFrame(all_plays_master_list)
        final_df.to_csv(output_csv, index=False)
        print("\n==========================================")
        print("      MASTER DATASET CREATION COMPLETE    ")
        print("==========================================")
        print(f"Total Valid Plays Extracted: {len(final_df)}")
        print(f"File Saved To:               {output_csv}")
        return final_df
    else:
        print("ERROR: No plays were successfully processed.")
        return None