import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import jax
import jax.numpy as jnp
import optuna
import json
from functools import partial
from jax import jit, vmap, vmap
from IPython.display import clear_output, display
import time
from .animation import animate_comparison_play

from .solver import run_simulation

# --- Visualization Helper --- #

def smooth_trajectory(traj, alpha=0.3):
    """
    Applies an Exponential Moving Average (EMA) to smooth out micro-jitters.
    Lower alpha = smoother, but very slightly delayed.
    """
    # Convert JAX array to standard NumPy array for easy processing
    traj = np.array(traj) 
    smoothed = np.zeros_like(traj)
    smoothed[0] = traj[0] 
    
    for t in range(1, len(traj)):
        smoothed[t] = alpha * traj[t] + (1 - alpha) * smoothed[t-1]
        
    return smoothed


# --- Data Extraction --- #

def extract_trajectories_from_row(row, solver_params, pad_for_bulk=True):
    """Extracts raw data, pads to fixed length for JAX, and runs the JKO simulations."""
    ball_traj = jnp.stack([jnp.array(row['ball_y_traj']) + 5.25, 
                           jnp.array(row['ball_x_traj']) + 25.0], axis=1)
    
    off_list, q_list, def_list, off_ids = [], [], [], []
    for i in range(1, 6):
        off_list.append(jnp.stack([jnp.array(row[f'off{i}_y_traj']) + 5.25, 
                                   jnp.array(row[f'off{i}_x_traj']) + 25.0], axis=1))
        q_list.append(jnp.array(row[f'off{i}_q_traj']))
        def_list.append(jnp.stack([jnp.array(row[f'def{i}_y_traj']) + 5.25, 
                                   jnp.array(row[f'def{i}_x_traj']) + 25.0], axis=1))
        off_ids.append(jnp.array(row[f'off{i}_pid']))
        
    off_traj = jnp.stack(off_list, axis=1) 
    q_traj = jnp.stack(q_list, axis=1)     
    real_def_traj = jnp.stack(def_list, axis=1)

    shot_frame = int(row['local_release_idx'])

    # -- STRICT PADDING FOR JAX COMPILATION CACHING --
    if pad_for_bulk:
        MAX_FRAMES = 75
        cutoff_idx = shot_frame  
        
        ball_traj = ball_traj[:cutoff_idx]
        off_traj = off_traj[:cutoff_idx]
        q_traj = q_traj[:cutoff_idx]
        real_def_traj = real_def_traj[:cutoff_idx]
        
        current_len = ball_traj.shape[0]
        
        if current_len < MAX_FRAMES:
            pad_len = MAX_FRAMES - current_len
            ball_traj = jnp.pad(ball_traj, ((0, pad_len), (0, 0)), mode='edge')
            off_traj = jnp.pad(off_traj, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            q_traj = jnp.pad(q_traj, ((0, pad_len), (0, 0)), mode='edge')
            real_def_traj = jnp.pad(real_def_traj, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            actual_shot_frame = current_len 
        else:
            ball_traj = ball_traj[:MAX_FRAMES]
            off_traj = off_traj[:MAX_FRAMES]
            q_traj = q_traj[:MAX_FRAMES]
            real_def_traj = real_def_traj[:MAX_FRAMES]
            actual_shot_frame = MAX_FRAMES
            
    else:
        actual_shot_frame = shot_frame 

    # -- Weights Calculation --
    ball_dist = jnp.linalg.norm(off_traj - ball_traj[:, None, :], axis=2)
    b_traj = 0.4 + 0.6 * (1.0 / (1.0 + jnp.exp(0.3 * (ball_dist - 18.0))))
    sim_weights = jnp.maximum((q_traj ** solver_params.get('ist_q_exp', 1.0)) * b_traj, 0.35)  
    basket_pos = jnp.array([5.25, 25.0]) if jnp.mean(real_def_traj[0, :, 0]) < 47.0 else jnp.array([88.75, 25.0])
        
    # -- Run Simulations --
    params_no_ist = solver_params.copy()
    params_no_ist['ist_weight'] = 0.0
    raw_sim_traj = run_simulation(real_def_traj[0], ball_traj, off_traj, q_traj, basket_pos, params_no_ist, 20)
    sim_def_no_ist_traj = smooth_trajectory(raw_sim_traj[:len(off_traj)])

    raw_sim_ist_traj = run_simulation(real_def_traj[0], ball_traj, off_traj, q_traj, basket_pos, solver_params, 20)
    sim_def_ist_traj = smooth_trajectory(raw_sim_ist_traj[:len(off_traj)]) 
   
    # -- IST Distances --
    off_exp = off_traj[:, :, None, :]
    sim_no_ist_dist = jnp.min(jnp.linalg.norm(off_exp - sim_def_no_ist_traj[:, None, :, :], axis=-1), axis=2)
    sim_ist_dist = jnp.min(jnp.linalg.norm(off_exp - sim_def_ist_traj[:, None, :, :], axis=-1), axis=2)
    real_dist = jnp.min(jnp.linalg.norm(off_exp - real_def_traj[:, None, :, :], axis=-1), axis=2)
    
    o_exp = solver_params.get('ist_o_exp', 1.0)
    weights_sim_no_ist = sim_weights * (jnp.clip(sim_no_ist_dist / 6.0, 0.5, 1.5) ** o_exp)
    weights_sim_ist = sim_weights * (jnp.clip(sim_ist_dist / 6.0, 0.5, 1.5) ** o_exp)
    weights_real = sim_weights * (jnp.clip(real_dist / 6.0, 0.5, 1.5) ** o_exp)

    # -- Zero-Out Post-Shot Padding --
    if pad_for_bulk:
        weights_sim_no_ist = weights_sim_no_ist.at[actual_shot_frame:].set(0.0)
        weights_sim_ist = weights_sim_ist.at[actual_shot_frame:].set(0.0)
        weights_real = weights_real.at[actual_shot_frame:].set(0.0)
    else:
        weights_sim_no_ist = weights_sim_no_ist.at[shot_frame+10:].set(0.0)
        weights_sim_ist = weights_sim_ist.at[shot_frame+10:].set(0.0)
        weights_real = weights_real.at[shot_frame+10:].set(0.0)

    return (sim_def_no_ist_traj, sim_def_ist_traj, real_def_traj, 
            weights_sim_no_ist, weights_sim_ist, weights_real, 
            off_traj, ball_traj, basket_pos, off_ids, actual_shot_frame)

def prepare_play_data(row):
    """Extracts and slices data specifically up to the local_release_idx."""
    
    # 1. Get the pre-calculated shot index
    end_idx = int(row['local_release_idx']) + 1 
    
    # 2. Ball
    ball_x = jnp.array(row['ball_x_traj'])[:end_idx] + 25.0
    ball_y = jnp.array(row['ball_y_traj'])[:end_idx] + 5.25
    ball_traj = jnp.stack([ball_y, ball_x], axis=1)

    # 3. Offense & Q-Values
    off_list, q_list = [], []
    for i in range(1, 6):
        x = jnp.array(row[f'off{i}_x_traj'])[:end_idx] + 25.0
        y = jnp.array(row[f'off{i}_y_traj'])[:end_idx] + 5.25
        off_list.append(jnp.stack([y, x], axis=1))
        
        q_vals = jnp.array(row[f'off{i}_q_traj'])[:end_idx]
        q_list.append(q_vals)
        
    off_traj = jnp.stack(off_list, axis=1)
    q_traj = jnp.stack(q_list, axis=1)

    # 4. Real Defense
    def_list = []
    for i in range(1, 6):
        x = jnp.array(row[f'def{i}_x_traj'])[:end_idx] + 25.0
        y = jnp.array(row[f'def{i}_y_traj'])[:end_idx] + 5.25
        def_list.append(jnp.stack([y, x], axis=1))
        
    real_def_traj = jnp.stack(def_list, axis=1)
    init_defenders = real_def_traj[0]

    # 5. Basket Position
    basket_pos = jnp.array([5.25, 25.0]) if row['flipped_coordinates'] else jnp.array([88.75, 25.0])

    # 6. Offender Weights 
    ball_expanded = ball_traj[:, None, :] 
    dist_to_ball = jnp.linalg.norm(off_traj - ball_expanded, axis=2)
    b_traj = 0.4 + 0.6 * (1.0 / (1.0 + jnp.exp(0.3 * (dist_to_ball - 18.0))))
    offender_weights_traj = q_traj * b_traj

    data_slice = (init_defenders, ball_traj, off_traj, real_def_traj, basket_pos)
    return data_slice, offender_weights_traj



# --- Summary Functions --- #

def get_play_summary(row, params):
    """
    Calculates aggregate pressure statistics for a play.
    Compares the Real Defense against the JKO Simulated Defense.
    """
    # Extract the play type label we just created (default to 'Unlabeled' if missing)
    play_type = row.get('play_action', 'Unlabeled')
    
    # 1. Run the simulation
    res = extract_trajectories_from_row(row, params)
    sim_traj, real_traj, ball_traj, ist_sim, ist_real, off_traj, basket = res
    
    # 2. Identify the shot frame (the point of ball release)
    shot_frame = int(row['local_release_idx'])
    
    # 3. Mask the trajectories
    ist_sim_active = ist_sim.at[shot_frame:].set(0.0)
    ist_real_active = ist_real.at[shot_frame:].set(0.0)
    
    # 4. Calculate Aggregate Statistics
    total_real_pressure = float(jnp.sum(ist_real_active))
    total_sim_pressure = float(jnp.sum(ist_sim_active))
    
    # Pressure Saved (Positive means the Simulation was 'Better' / tighter)
    pressure_saved = total_real_pressure - total_sim_pressure
    pct_improvement = (pressure_saved / total_real_pressure) * 100 if total_real_pressure > 0 else 0
    
    return {
        "Play Index": row.name,
        "Play Type": play_type,  # <-- NEW: Added to output
        "Total Real IST": round(total_real_pressure, 2),
        "Total Sim IST": round(total_sim_pressure, 2),
        "Pressure Prevented": round(pressure_saved, 2),
        "Efficiency Gain (%)": f"{pct_improvement:.1f}%",
        "Active Frames": shot_frame
    }

def get_global_report(summary_df):
    # 1. Clean data (convert string percentages to floats)
    if summary_df['Efficiency Gain (%)'].dtype == object:
        eff_floats = summary_df['Efficiency Gain (%)'].str.replace('%', '').astype(float)
        summary_df['Efficiency Gain Float'] = eff_floats
    else:
        summary_df['Efficiency Gain Float'] = summary_df['Efficiency Gain (%)']
        
    total_real_ist = summary_df['Total Real IST'].sum()
    total_sim_ist = summary_df['Total Sim IST'].sum()
    total_pressure_prevented = summary_df['Pressure Prevented'].sum()
    
    # 2. Overall Defensive Efficiency 
    global_efficiency = (total_pressure_prevented / total_real_ist) * 100 if total_real_ist > 0 else 0
    
    # 3. Play-Level Stats
    avg_gain = summary_df['Efficiency Gain Float'].mean()
    median_gain = summary_df['Efficiency Gain Float'].median()
    std_gain = summary_df['Efficiency Gain Float'].std()
    
    # 4. Outlier Detection (Top 3 Defensive Lapses)
    top_3_lapses = summary_df.nlargest(3, 'Pressure Prevented')[
        ['Play Index', 'Play Type', 'Pressure Prevented', 'Efficiency Gain (%)']
    ]

    # --- NEW: 5. PLAY TYPE AGGREGATION ---
    # Group by Play Type and calculate volume and efficiency
    type_stats = summary_df.groupby('Play Type').agg(
        Count=('Play Index', 'count'),
        Total_Real_IST=('Total Real IST', 'sum'),
        Total_Pressure_Saved=('Pressure Prevented', 'sum')
    ).reset_index()
    
    # Calculate True Efficiency Gain per play type
    type_stats['Efficiency Gain'] = (type_stats['Total_Pressure_Saved'] / type_stats['Total_Real_IST'] * 100)
    type_stats['Efficiency Gain'] = type_stats['Efficiency Gain'].fillna(0).apply(lambda x: f"{x:.1f}%")
    
    # Clean up columns for printing
    type_stats = type_stats[['Play Type', 'Count', 'Efficiency Gain']].sort_values(by='Count', ascending=False)

    # --- PRINTING THE REPORT ---
    print("==========================================")
    print("      GLOBAL DEFENSIVE ANALYSIS REPORT    ")
    print("==========================================")
    print(f"Total Plays Analyzed:       {len(summary_df)}")
    print(f"Total Real Pressure:        {total_real_ist:.2f}")
    print(f"Total Sim Pressure:         {total_sim_ist:.2f}")
    print(f"Total Pressure Prevented:   {total_pressure_prevented:.2f}")
    print(f"---")
    print(f"GLOBAL EFFICIENCY GAIN:     {global_efficiency:.2f}%")
    print(f"---")
    print(f"Avg Gain per Play:          {avg_gain:.2f}%")
    print(f"Consistency (Std Dev):      {std_gain:.2f}%")
    
    print("\n==========================================")
    print("         BREAKDOWN BY PLAY TYPE           ")
    print("==========================================")
    print(type_stats.to_string(index=False))
    
    print("\n==========================================")
    print("  TOP 3 DEFENSIVE LAPSES (Real vs Sim)    ")
    print("==========================================")
    print(top_3_lapses.to_string(index=False))


# --- Add Play Types --- #
def add_play_action(df, best_params):
    # Create an empty column if it doesn't exist
    if 'play_action' not in df.columns:
        df['play_action'] = None

    # Filter to only unlabeled rows, then get unique GAME_ID and SHOT_EVENT_ID pairs
    unlabeled_df = df[df['play_action'].isnull()]
    unlabeled_plays = unlabeled_df[['GAME_ID', 'SHOT_EVENT_ID']].drop_duplicates().values

    total_unlabeled = len(unlabeled_plays)

    # Iterate by index through the unique pairs
    for idx in range(total_unlabeled):
        game_id, shot_event_id = unlabeled_plays[idx]
        
        play_mask = (df['GAME_ID'] == game_id) & (df['SHOT_EVENT_ID'] == shot_event_id)
        play = df[play_mask].iloc[0] # Grab the single row
        
        # 1. Extract trajectories
        sim_def_traj, real_def_traj, ball_traj, ist_sim, ist_real, off_traj, basket_pos = extract_trajectories_from_row(play, best_params)
        
        # Optional: Smooth the simulated trajectory if you are still doing that
        # smoothed_sim_def_traj = smooth_trajectory(sim_def_traj, alpha=0.3)
        
        # 2. Build the animation
        fig = animate_comparison_play(off_traj, real_def_traj, sim_def_traj, ball_traj, ist_real, ist_sim) 
        
        # FORCE INLINE RENDERING USING IFRAME    
        # Pause for just a fraction of a second to ensure the frontend loads the iframe 
        # before the input prompt pauses the kernel
        time.sleep(0.5) 
        
        # 3. Prompt yourself for the label right in the Jupyter cell
        prompt_text = (f"Watch the play above. Enter label for play {idx + 1} of {total_unlabeled} "
                    f"(Game: {game_id}, Event: {shot_event_id}) | Type 'quit' to stop: ")
        
        label = input(prompt_text)
        
        if label.lower() == 'quit':
            print(f"Stopping labeling session at index {idx}.")
            break
            
        # 4. Save the label directly into the dataframe
        df.loc[play_mask, 'play_action'] = label

        # Clear the output to prepare for the next play
        clear_output(wait=True)

    print("Labeling session complete or exited.")