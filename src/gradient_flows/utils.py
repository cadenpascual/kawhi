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
from .viz_tools import run_simulation

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

def extract_trajectories_from_row(row, solver_params):
    # 1. Extract Ball Data -> Shape: (num_frames, 2)
    ball_traj = jnp.stack([jnp.array(row['ball_y_traj']) + 5.25, 
                           jnp.array(row['ball_x_traj']) + 25.0], axis=1)
    
    # 2. Extract Player Data
    off_list, q_list, def_list, off_ids = [], [], [], []
    for i in range(1, 6):
        # Offense: (num_frames, 2) per player
        off_list.append(jnp.stack([jnp.array(row[f'off{i}_y_traj']) + 5.25, 
                                   jnp.array(row[f'off{i}_x_traj']) + 25.0], axis=1))
        
        # Q-values: (num_frames,) per player
        q_list.append(jnp.array(row[f'off{i}_q_traj']))
        
        # Defense: (num_frames, 2) per player
        def_list.append(jnp.stack([jnp.array(row[f'def{i}_y_traj']) + 5.25, 
                                   jnp.array(row[f'def{i}_x_traj']) + 25.0], axis=1))
        
        off_ids.append(jnp.array(row[f'off{i}_pid']))
        
    # 3. Stack into 3D Arrays -> Shape: (num_frames, 5, 2)
    off_traj = jnp.stack(off_list, axis=1) 
    q_traj = jnp.stack(q_list, axis=1)     
    real_def_traj = jnp.stack(def_list, axis=1)

    # 4. Compute Weights
    # ball_traj[:, None, :] broadcasts to (num_frames, 1, 2) for distance calculation
    ball_dist = jnp.linalg.norm(off_traj - ball_traj[:, None, :], axis=2)
    b_traj = 0.4 + 0.6 * (1.0 / (1.0 + jnp.exp(0.3 * (ball_dist - 18.0))))
    
    sim_weights = jnp.maximum((q_traj ** solver_params.get('ist_q_exp', 1.0)) * b_traj, 0.35)  
    
    # 5. Hoop Position
    basket_pos = jnp.array([5.25, 25.0]) if jnp.mean(real_def_traj[0, :, 0]) < 47.0 else jnp.array([88.75, 25.0])
        
    # 6. Run Simulation
    params_no_ist = solver_params.copy()
    params_no_ist['ist_weight'] = 0.0
    raw_sim_traj = run_simulation(real_def_traj[0], ball_traj, off_traj, q_traj, basket_pos, params_no_ist, 20)
    sim_def_no_ist_traj = smooth_trajectory(raw_sim_traj[:len(off_traj)])

    raw_sim_ist_traj = run_simulation(real_def_traj[0], ball_traj, off_traj, q_traj, basket_pos, solver_params, 20)
    sim_def_ist_traj = smooth_trajectory(raw_sim_ist_traj[:len(off_traj)]) # Match lengths
   

    # 7. Display IST Calculation 
    off_exp = off_traj[:, :, None, :]
    sim_no_ist_dist = jnp.min(jnp.linalg.norm(off_exp - sim_def_no_ist_traj[:, None, :, :], axis=-1), axis=2)
    sim_ist_dist = jnp.min(jnp.linalg.norm(off_exp - sim_def_ist_traj[:, None, :, :], axis=-1), axis=2)
    real_dist = jnp.min(jnp.linalg.norm(off_exp - real_def_traj[:, None, :, :], axis=-1), axis=2)
    
    o_exp = solver_params.get('ist_o_exp', 1.0)
    weights_sim_no_ist = sim_weights * (jnp.clip(sim_no_ist_dist / 6.0, 0.5, 1.5) ** o_exp)
    weights_sim_ist = sim_weights * (jnp.clip(sim_ist_dist / 6.0, 0.5, 1.5) ** o_exp)
    weights_real = sim_weights * (jnp.clip(real_dist / 6.0, 0.5, 1.5) ** o_exp)

    # 8. Cutoff IST Calculation at Shot Frame
    shot_frame = row['local_release_idx']
    weights_sim_no_ist = weights_sim_no_ist.at[shot_frame+10:].set(0.0)
    weights_sim_ist = weights_sim_ist.at[shot_frame+10:].set(0.0)
    weights_real = weights_real.at[shot_frame+10:].set(0.0)

    return sim_def_no_ist_traj, sim_def_ist_traj, real_def_traj, weights_sim_no_ist, weights_sim_ist, weights_real, off_traj, ball_traj, basket_pos, off_ids, shot_frame

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
    Upgraded: Uses all 11 arguments from extract_trajectories_from_row.
    Calculates the 'Systemic Success' by comparing frame-by-frame performance.
    """
    play_type = row.get('play_action', 'Unlabeled')
    
    # 1. Extract the full set of results
    (sim_def_no_ist_traj, 
     sim_def_ist_traj, 
     real_def_traj, 
     weights_sim_no_ist, 
     weights_sim_ist, 
     weights_real, 
     off_traj, 
     ball_traj, 
     basket_pos, 
     off_ids, 
     shot_frame) = extract_trajectories_from_row(row, params)
    
    # 2. Slice to the active play (before shot release)
    # We use weights_real and weights_sim_ist for the primary comparison
    ist_sim_active = weights_sim_ist[:shot_frame]
    ist_real_active = weights_real[:shot_frame]
    
    # 3. Calculate Team Totals per frame (Summing across the 5 offenders)
    team_sim_per_frame = np.sum(ist_sim_active, axis=1)
    team_real_per_frame = np.sum(ist_real_active, axis=1)
    
    # --- THE SYSTEMIC SUCCESS CALCULATION ---
    # How often was the Optimized Sim actually lower threat than the Real Defense?
    frames_won = np.sum(team_sim_per_frame < team_real_per_frame)
    win_rate = (frames_won / shot_frame) * 100 if shot_frame > 0 else 0
    
    # 4. Aggregate Totals for the Play
    total_real = float(np.sum(team_real_per_frame))
    total_sim = float(np.sum(team_sim_per_frame))
    pressure_saved = total_real - total_sim
    pct_improvement = (pressure_saved / total_real) * 100 if total_real > 0 else 0
    
    # 5. Baseline Comparison (Sim with IST vs Sim without IST)
    # This proves the "IST Logic" is specifically what is helping
    total_no_ist_sim = float(np.sum(np.sum(weights_sim_no_ist[:shot_frame], axis=1)))
    ist_added_value = total_no_ist_sim - total_sim
    
    return {
        "Play Index": row.name,
        "Play Type": play_type,
        "Total Real IST": round(total_real, 2),
        "Total Sim IST": round(total_sim, 2),
        "Pressure Prevented": round(pressure_saved, 2),
        "Efficiency Gain (%)": pct_improvement,
        "Frame Win Rate (%)": round(win_rate, 1),
        "IST Value Add": round(ist_added_value, 2), # Value of the IST model specifically
        "Active Frames": shot_frame
    }

def get_global_report(summary_df):
    """
    Prints a professional 'Front Office' style report.
    """
    # 1. Metric Calculations
    total_real = summary_df['Total Real IST'].sum()
    total_prevented = summary_df['Pressure Prevented'].sum()
    global_gain = (total_prevented / total_real) * 100
    
    avg_win_rate = summary_df['Frame Win Rate (%)'].mean()
    median_gain = summary_df['Efficiency Gain (%)'].median()
    std_dev = summary_df['Efficiency Gain (%)'].std()

    # 2. Printing the Report
    print("="*42)
    print("      CLIPPERS SPATIAL AUDIT: GLOBAL REPORT")
    print("="*42)
    print(f"Total Plays Analyzed:       {len(summary_df)}")
    print(f"Total IST Units Prevented:  {total_prevented:,.2f}")
    print(f"---")
    print(f"GLOBAL EFFICIENCY GAIN:     {global_gain:.2f}%")
    print(f"SYSTEMIC SUCCESS RATE:      {avg_win_rate:.1f}% of frames improved")
    print(f"---")
    print(f"Median Gain per Play:       {median_gain:.2f}%")
    print(f"Model Consistency (σ):      {std_dev:.2f}%")
    
    # 3. Play Type Breakdown
    print("\n" + "="*42)
    print("      EFFICIENCY BY PLAY CATEGORY")
    print("="*42)
    type_stats = summary_df.groupby('Play Type').agg(
        Count=('Play Index', 'count'),
        Avg_Gain=('Efficiency Gain (%)', 'mean')
    ).sort_values('Avg_Gain', ascending=False)
    print(type_stats.to_string())

    # 4. Identifying the 'Gravity Lapses'
    print("\n" + "="*42)
    print("      TOP 3 DETECTED GRAVITY LAPSES")
    print("="*42)
    top_3 = summary_df.nlargest(3, 'Pressure Prevented')
    print(top_3[['Play Index', 'Play Type', 'Efficiency Gain (%)']].to_string(index=False))

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