import os
# ==========================================
# 1. JAX MEMORY SAFETY SETTINGS (MUST BE FIRST)
# ==========================================
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import optuna
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your external functions
from src.gradient_flows.utils import extract_trajectories_from_row
from src.gradient_flows.potentials import params as default_params

def get_optimized_params(study_name, storage_name, target_trial_num=None):
    """Loads either a user-specified trial or the optimal kinematic baseline."""
    merged_params = default_params.copy()
    
    # Ultra-Stable Solver Dynamics
    solver_defaults = {
        'learning_rate': 0.05,       
        'jko_lambda': 0.5,
        'sinkhorn_epsilon': 0.5,     
        'velocity_cap': 0.8,
        'court_dims': [[0.0, 94.0], [0.0, 50.0]],
        'max_gradient_norm': 1.0,
        'acceleration_penalty_weight': 2.0,
        'velocity_penalty_weight': 0.5,
        'ist_q_exp': 2.16,
        'ist_o_exp': 1.03
    }
    merged_params.update(solver_defaults)

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        selected_trial = None
        
        # If the user requested a specific trial, hunt it down
        if target_trial_num is not None:
            for t in study.trials:
                if t.number == target_trial_num:
                    selected_trial = t
                    break
            
            if selected_trial is not None:
                print(f"[*] User Override: Loaded Specific Tuning Parameters (Trial #{selected_trial.number}).")
            else:
                print(f"[!] Warning: Trial #{target_trial_num} not found. Falling back to the best trial.")
        
        # If no specific trial was requested (or if the requested one wasn't found), grab the best one
        if selected_trial is None:
            selected_trial = min(study.best_trials, key=lambda t: t.values[0])
            print(f"[*] Automatically Loaded Best Tuning Parameters (Trial #{selected_trial.number}).")
            
        merged_params.update(selected_trial.params)
        
    except Exception as e:
        print(f"[!] Warning: Could not load Optuna study. Using default parameters. ({e})")
        
    return merged_params

def process_single_play(row, params):
    """
    Runs the JKO simulation for a single play, un-flips the coordinates,
    and saves each simulated defender's X and Y trajectory as separate columns.
    """
    try:
        (sim_def_no_ist_traj, sim_def_ist_traj, real_def_traj, 
         weights_sim_no_ist, weights_sim_ist, weights_real, 
         off_traj, ball_traj, basket_pos, off_ids, shot_frame) = extract_trajectories_from_row(row, params, pad_for_bulk=False)

        # 1. Convert to numpy for vectorized coordinate un-flipping
        sim_def_np = np.array(sim_def_ist_traj) 
        base_def_np = np.array(sim_def_no_ist_traj)
        
        # Un-Flip and Un-Translate the coordinates
        # Index 1 is X (subtract 25.0), Index 0 is Y (subtract 5.25)
        sim_def_np[:, :, 1] -= 25.0
        sim_def_np[:, :, 0] -= 5.25
        
        base_def_np[:, :, 1] -= 25.0
        base_def_np[:, :, 0] -= 5.25
        
        new_data = {}
        
        # Store the frame-by-frame IST threat arrays
        for i in range(5):
            player_num = i + 1
            
            # 1. Trajectories
            new_data[f'sim_def{player_num}_x_traj'] = sim_def_np[:, i, 1].tolist()
            new_data[f'sim_def{player_num}_y_traj'] = sim_def_np[:, i, 0].tolist()
            
            new_data[f'base_def{player_num}_x_traj'] = base_def_np[:, i, 1].tolist()
            new_data[f'base_def{player_num}_y_traj'] = base_def_np[:, i, 0].tolist()
            
            # 2. IST Threat Values (Saved per player!)
            new_data[f'ist_real_{player_num}'] = weights_real[:, i].tolist()
            new_data[f'ist_sim_{player_num}'] = weights_sim_ist[:, i].tolist()
            new_data[f'ist_base_{player_num}'] = weights_sim_no_ist[:, i].tolist()
            
        # --- METRICS MATH ---
        shot_idx = int(shot_frame)
        mask = np.arange(len(weights_real)) < shot_idx

        # Collapse the 5 defenders into a single Team IST metric per frame
        if weights_real.ndim == 2:
            weights_real_team = np.sum(weights_real, axis=1)
            weights_sim_team = np.sum(weights_sim_ist, axis=1)
            weights_base_team = np.sum(weights_sim_no_ist, axis=1)
        else:
            weights_real_team = weights_real
            weights_sim_team = weights_sim_ist
            weights_base_team = weights_sim_no_ist

        # 1. Total IST Math
        total_real = float(np.sum(np.where(mask, weights_real_team, 0.0)))
        total_sim = float(np.sum(np.where(mask, weights_sim_team, 0.0)))
        total_base = float(np.sum(np.where(mask, weights_base_team, 0.0)))
        
        # 2. Frame Win Rate Math (Sim vs Real)
        won_mask = mask & (weights_real_team > weights_sim_team)
        frames_won = int(np.sum(won_mask))
        total_frames = int(np.sum(mask))
        frame_win_rate = round(float(frames_won) / float(total_frames), 4) if total_frames > 0 else 0.0
        
        # --- SAVE TO PARQUET ---
        new_data['Real_IST_Total'] = round(total_real, 3)
        new_data['Sim_IST_Total'] = round(total_sim, 3)
        new_data['Base_IST_Total'] = round(total_base, 3)
        
        new_data['IST_Threat_Prevented'] = round(total_real - total_sim, 3)
        new_data['Base_Threat_Prevented'] = round(total_real - total_base, 3)
        
        new_data['Frames_Won'] = frames_won
        new_data['Total_Frames'] = total_frames
        new_data['Frame_Win_Rate'] = frame_win_rate
        
        return pd.Series(new_data)

    except Exception as e:
        print(f"Skipping play due to error: {e}")
        return pd.Series(dtype=float)
    
    
def worker_func(args):
    idx, r, p = args
    return idx, process_single_play(r, p)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Final Simulated Trajectories & Metrics")
    parser.add_argument("--demo", action="store_true", help="Run on the 1-game demo dataset")
    parser.add_argument("--trial", type=int, default=None, help="Specific Optuna trial number to load (e.g., --trial 14)")
    args, _ = parser.parse_known_args()

    # --- 1. DYNAMIC PATHING ---
    if args.demo:
        print("[*] DEMO MODE ACTIVATED")
        INPUT_FILE = "data/demo/processed/demo_traj.parquet"
        OUTPUT_FILE = "data/demo/final/demo_simulated_traj.parquet"
        DB_PATH = "sqlite:///data/demo/processed/demo-ist-tuning.db"
        STUDY_NAME = "demo-ist-tuning"
    else:
        print("[*] FULL SEASON MODE ACTIVATED")
        INPUT_FILE = "data/processed/traj_features/all_season_traj.parquet"
        OUTPUT_FILE = "data/processed/traj_features/all_season_simulated_traj.parquet"
        DB_PATH = "sqlite:///data/processed/optimization/stage2-ist-tuning.db" 
        STUDY_NAME = "stage2-ist-tuning"

    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: {INPUT_FILE} not found.")
        exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # --- 2. LOAD DATA & PARAMS ---
    df = pd.read_parquet(INPUT_FILE)
    
    # Pass the user's trial choice into the loader
    params = get_optimized_params(STUDY_NAME, DB_PATH, target_trial_num=args.trial)
    
    # Strip known poison pills
    if args.demo:
        df = df[~df['SHOT_EVENT_ID'].isin([284, 321, 503])].copy()

    print(f"[*] Processing {len(df)} total possessions. Running JKO physics engine...")

    # --- 3. RUN SIMULATIONS & APPEND COLUMNS ---
    print(f"[*] Simulating Plays across {os.cpu_count() - 1} CPU cores...")
    
    # Package the rows so they can be sent to different CPU cores
    tasks = [(index, row, params) for index, row in df.iterrows()]
    results_dict = {}


    # Spin up the parallel workers
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = {executor.submit(worker_func, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Simulating"):
            idx, result_series = future.result()
            results_dict[idx] = result_series

    # Reconstruct the dataframe from the parallel results
    simulated_columns_df = pd.DataFrame.from_dict(results_dict, orient='index')
    
    # --- Strip out old simulation columns before merging ---
    cols_to_drop = simulated_columns_df.columns.intersection(df.columns)
    if len(cols_to_drop) > 0:
        print(f"[*] Cleaning up {len(cols_to_drop)} old simulation columns...")
        df = df.drop(columns=cols_to_drop)
        
    final_df = pd.concat([df, simulated_columns_df], axis=1)
    
    initial_len = len(final_df)
    final_df = final_df.dropna(subset=['Real_IST_Total'])
    dropped = initial_len - len(final_df)
    
    if dropped > 0:
        print(f"[*] Dropped {dropped} plays due to kinematic anomalies.")

    # --- 4. SAVE FINAL MASTER PARQUET ---
    final_df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n" + "="*50)
    print("      SIMULATION GENERATION COMPLETE")
    print("="*50)
    print(f"Total Valid Plays Saved: {len(final_df)}")
    print(f"File Saved To:           {OUTPUT_FILE}")
    print(f"STEP 4 COMPLETE: Check QUICKSTART_DEMO.ipynb to view results")