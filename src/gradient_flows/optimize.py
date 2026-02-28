import jax
import jax.numpy as jnp
import optuna
import json
import math
from functools import partial
from jax import jit, vmap, vmap


# Import from our existing modules
from .potentials import params as default_params
from .solver import run_simulation
from .utils import extract_quality_trajectories, prepare_play_data
from src.data_io.maps import load_maps_npz

# --- Data Loading ---

def load_data_slice(file_path, num_frames=50):
    """
    Loads a slice of NBA tracking data from a JSON file.
    
    For simplicity, this function takes the first event and extracts a fixed
    number of frames from it. It also crudely assigns teams to offense/defense.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    event = data[0] # Use the first event
    frames = event['frames'][:num_frames]
    
    # Identify the two teams
    team_ids = sorted(list(set(p['teamid'] for p in frames[0]['players'])))
    if len(team_ids) != 2:
        raise ValueError("Data slice does not contain exactly two teams.")
    
    # Heuristic: Find player closest to the ball in the first frame to determine offense
    first_frame = frames[0]
    ball_pos = jnp.array([first_frame['ball']['x'], first_frame['ball']['y']])
    min_dist = float('inf')
    off_team_id = -1

    for p in first_frame['players']:
        player_pos = jnp.array([p['x'], p['y']])
        dist = jnp.linalg.norm(ball_pos - player_pos)
        if dist < min_dist:
            min_dist = dist
            off_team_id = p['teamid']
            
    def_team_id = team_ids[0] if team_ids[1] == off_team_id else team_ids[1]
    
    # --- Extract Player IDs ---
    # Sort them by playerid to guarantee alignment with coordinate extraction
    off_pids = sorted([p['playerid'] for p in first_frame['players'] if p['teamid'] == off_team_id])

    # Extract trajectories
    ball_traj = []
    offenders_traj = []
    defenders_traj = []

    for frame in frames:
        ball_pos = jnp.array([frame['ball']['x'], frame['ball']['y']])
        
        off_players = sorted([p for p in frame['players'] if p['teamid'] == off_team_id], key=lambda p: p['playerid'])
        def_players = sorted([p for p in frame['players'] if p['teamid'] == def_team_id], key=lambda p: p['playerid'])
        
        if len(off_players) != 5 or len(def_players) != 5:
            continue # Skip frames without 5 players per side

        off_pos = jnp.array([[p['x'], p['y']] for p in off_players])
        def_pos = jnp.array([[p['x'], p['y']] for p in def_players])

        ball_traj.append(ball_pos)
        offenders_traj.append(off_pos)
        defenders_traj.append(def_pos)

    if not defenders_traj:
        raise ValueError("Could not extract a valid slice with 5v5 gameplay.")

    # We only need the initial defensive positions
    init_defenders = defenders_traj[0]
    
    return (
        jnp.array(init_defenders),
        jnp.array(ball_traj),
        jnp.array(offenders_traj),
        off_pids
    )

# --- Loss Metrics ---
@jit
def calculate_losses(def_traj, off_traj, offender_weights_traj, basket_pos):
    """
    Calculates pressure based on Optimal Transport / IST weights.
    High threat players must be guarded closely to lower the loss.
    """
    def pressure_at_t(def_pos, off_pos, weights):
        # 1. Distance matrix (5 offenders x 5 defenders)
        diffs = off_pos[:, jnp.newaxis, :] - def_pos[jnp.newaxis, :, :]
        dists = jnp.linalg.norm(diffs, axis=2)
        
        # 2. Find the closest defender for each offender
        min_dists = jnp.min(dists, axis=1)
        
        # 3. WEIGHT the loss by IST (offender_weights_traj)
        # If a 90% threat shooter is open, the loss is huge.
        # If a 5% threat non-shooter is open, the loss is small.
        weighted_pressure = min_dists * weights
        
        return jnp.mean(weighted_pressure)

    # Vmap over the timeline
    pressure_loss = jnp.mean(vmap(pressure_at_t)(def_traj, off_traj, offender_weights_traj))
    
    # Smoothness loss keeps movements realistic
    smoothness_loss = jnp.mean(jnp.linalg.norm(jnp.diff(def_traj, axis=0), axis=2))
    
    return pressure_loss, smoothness_loss

# --- Optuna Objective Function ---

def objective(trial, batch_data):
    """Minimizes average pressure and smoothness loss across the batch."""
    params = default_params.copy()

    params.update({
    # --- HARD PHYSICS LIMITS (The new anti-slingshot variables) ---
    'max_acceleration': trial.suggest_float("max_acceleration", 30.0, 80.0), # Limits instant direction changes
    'velocity_cap': trial.suggest_float("velocity_cap", 15.0, 22.0),         # Human top speed
    
    # --- SOLVER TUNING ---
    'learning_rate': trial.suggest_float("learning_rate", 0.5, 3.0),       # Smaller steps so SGD doesn't jump over the target
    'jko_num_steps': trial.suggest_int("jko_num_steps", 20, 50),             # More steps to reach the target smoothly
    
    # --- TACTICS & ASSIGNMENTS ---
    'cushion_dist': trial.suggest_float("cushion_dist", 1.5, 3.5),           # How tight the defense plays
    'attraction_weight': trial.suggest_float("attraction_weight", 8.0, 20.0),# How aggressive they close out
    'field_weight': trial.suggest_float("field_weight", -50.0, -20.0),
    'basket_gravity_weight': trial.suggest_float("basket_gravity_weight", 5.0, 20.0),
    'global_ball_weight': trial.suggest_float("global_ball_weight", 0.01, 0.15),
    
    # --- SPREAD & SPACING ---
    'sigma_long': trial.suggest_float("sigma_long", 4.0, 10.0),
    'sigma_wide': trial.suggest_float("sigma_wide", 2.0, 6.0),
    'lane_blocking_weight': trial.suggest_float("lane_blocking_weight", 20.0, 45.0),
    'occupancy_weight': trial.suggest_float("occupancy_weight", 20.0, 45.0),
    'cohesion_weight': trial.suggest_float("cohesion_weight", 0.1, 1.5),
    'formation_radius': trial.suggest_float("formation_radius", 12.0, 20.0),
    
    'court_dims': [[0., 94.], [0., 50.]],
    'acceleration_penalty_weight': 0.0,
    'jko_lambda': 0.5,             
    'sinkhorn_epsilon': 0.05,       
    'max_gradient_norm': 100, 
    'sprint_penalty_weight': 2.0, # (Adding this too just in case it was missing)
})

    
    total_pressure_loss = 0.0
    total_smoothness_loss = 0.0
    valid_plays = 0
    
    for _, row in batch_data.iterrows(): # Assuming batch_data is a DataFrame
        play_data, offender_weights_traj = prepare_play_data(row)
        init_defenders, ball_traj, off_traj, real_def_traj, basket_pos = play_data

        # SANITY CHECK: If any input data is NaN, skip this play
        if jnp.any(jnp.isnan(init_defenders)) or jnp.any(jnp.isnan(off_traj)):
            print(f"Skipping Play: Input data contains NaNs! Row index: {row.name}")
            continue

        if jnp.any(jnp.isnan(offender_weights_traj)):
            # This happens if q_traj extraction failed for a specific coordinate
            offender_weights_traj = jnp.nan_to_num(offender_weights_traj, nan=0.1)
        
        # Skip plays that are too short after slicing (e.g., less than 5 frames)
        if len(ball_traj) < 5:
            # Add a print statement here to see if plays are too short
            # print(f"Play skipped! Length: {len(ball_traj)}") 
            continue
            
        try:
            simulated_def_traj = run_simulation(
                init_defenders, ball_traj, off_traj, offender_weights_traj, 
                basket_pos, params, jko_num_steps=20
            )
            
            # DIAGNOSTIC: Check if the trajectory itself is the problem
            if jnp.any(jnp.isnan(simulated_def_traj)):
                print(f"Pruned: Trajectory exploded (NaNs found). Field Weight: {params['field_weight']:.2f}, LR: {params.get('learning_rate', 'N/A')}")
                raise optuna.exceptions.TrialPruned()

            pressure_loss, smoothness_loss = calculate_losses(
                simulated_def_traj, off_traj, offender_weights_traj, basket_pos 
            )
            
            # ... rest of your code ...
            
            if jnp.isnan(pressure_loss) or jnp.isnan(smoothness_loss):
                print("Pruned: Losses were NaN")
                raise optuna.exceptions.TrialPruned()
                
            total_pressure_loss += pressure_loss
            total_smoothness_loss += smoothness_loss
            valid_plays += 1
            
        except Exception as e:
            # THIS IS THE CRITICAL CHANGE: Print the actual error!
            if not isinstance(e, optuna.exceptions.TrialPruned):
                print(f"Pruned due to Crash: {e}")
            raise optuna.exceptions.TrialPruned()
            
    if valid_plays == 0:
        print("Pruned: valid_plays was 0 (All plays were skipped or failed)")
        raise optuna.exceptions.TrialPruned()
    return (total_pressure_loss / valid_plays), (total_smoothness_loss / valid_plays)

# --- Main Execution ---

if __name__ == "__main__":
    STUDY_NAME = "nba-defensive-optimization"
    STORAGE_NAME = "sqlite:///{}.db".format(STUDY_NAME)
    DATA_FILE = "datasets/0021500622_labeled.json"
    
    print("Loading data slice...")
    # Load data once outside the objective function for efficiency
    try:
        data_slice = load_data_slice(DATA_FILE, num_frames=50)
        init_defenders, ball_traj, off_traj, off_pids = data_slice
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'datasets/0021500622_labeled.json' is available.")
        exit(1)
        

    print("Data loaded. Computing threat weights...")
        # --- Calculate Offender Weights --- 
    maps, meta = load_maps_npz("datasets/maps_1ft_xpps.npz")
    pid2row = {int(p): i for i, p in enumerate(maps["player_ids"])}

    # Extract Quality (Q)
    q_traj = extract_quality_trajectories(off_traj, off_pids, maps, pid2row)

    # Extract Ball Factor (B)
    ball_expanded = ball_traj[:, None, :] 
    dist_to_ball = jnp.linalg.norm(off_traj - ball_expanded, axis=2)
    k = 0.3
    d0 = 18.0
    b_floor = 0.4
    raw_logistic = 1.0 / (1.0 + jnp.exp(k * (dist_to_ball - d0)))
    b_traj = b_floor + (1.0 - b_floor) * raw_logistic

    # Final Pre-Computed Weights
    offender_weights_traj = q_traj * b_traj

    print("Weights computed. Starting optimization...")

    # 1. Create or load the Optuna study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        directions=["minimize", "minimize"], # For multi-objective
        load_if_exists=True,
    )

    # 2. Run the optimization
    # Pass data using a lambda to avoid reloading it in every trial
    study.optimize(lambda trial: objective(trial, data_slice), n_trials=100)
    # 3. Print results
    print("\nOptimization Finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nPareto Front (Best Trials):")
    for trial in study.best_trials:
        print(f"  Trial {trial.number}:")
        print(f"    Values: {trial.values}")
        print(f"    Params: {trial.params}")

    # You can also visualize the results if you have plotly installed
    try:
        import optuna.visualization as vis
        fig = vis.plot_pareto_front(study)
        fig.show()
    except ImportError:
        print("\nInstall plotly and kaleido to visualize the Pareto front:")
        print("pip install plotly kaleido")

