import jax
import jax.numpy as jnp
import optuna
import json
from functools import partial
from jax import jit, vmap, vmap

# Import from our existing modules
from .potentials import params as default_params
from .solver import run_simulation
from .utils import prepare_play_data

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
        jnp.array(offenders_traj)
    )

# --- Loss Metrics ---

@jit
def calculate_losses(def_traj, off_traj, ball_traj, basket_pos, q_traj, params):
    """
    Calculates the true IST of the simulated defense to use as the loss metric!
    We want Optuna to find the weights that MINIMIZE the total IST.
    """
    def ist_at_t(off_pos, def_pos, ball_pos, q_vals):
        # 1. B-Traj (Distance to ball) 
        ball_dist = jnp.linalg.norm(off_pos - ball_pos, axis=1)
        b_val = 0.4 + 0.6 * (1.0 / (1.0 + jnp.exp(0.3 * (ball_dist - 18.0))))
        
        # 2. Sim Weights 
        sim_weight = jnp.maximum((q_vals ** params['ist_q_exp']) * b_val, 0.35)
        
        # 3. Openness (O) using hard minimum for true evaluation
        dists_to_defs = jnp.linalg.norm(def_pos - off_pos[:, None, :], axis=2)
        closest_dist = jnp.min(dists_to_defs, axis=1)
        openness = jnp.clip(closest_dist / 6.0, 0.5, 1.5)
        
        return jnp.sum(sim_weight * (openness ** params['ist_o_exp']))

    # Vmap across time and mean it
    mean_ist_loss = jnp.mean(vmap(ist_at_t)(off_traj, def_traj, ball_traj, q_traj))
    
    # Keep smoothness so it doesn't just vibrate wildly to get low IST
    smoothness_loss = jnp.mean(jnp.linalg.norm(jnp.diff(def_traj, axis=0), axis=2))
    
    return mean_ist_loss, smoothness_loss

# --- Optuna Objective Function ---

def objective(trial, data):
    """The Optuna objective function to be minimized."""

    # 1. HARDCODE your best parameters
    params = {
        'blending_radius': jnp.array(15.0),
        'blending_k': jnp.array(0.5),
        'attraction_weight': 7.570765933969438,
        'basket_gravity_weight': 1.5510598901937178,
        'basket_gravity_sigma': jnp.array(12.0),
        'field_weight': -16.627957913865643,
        'global_ball_weight': 0.10151136411006659,
        'offender_threat_scale': jnp.array(2.0),
        'offender_threat_dist': jnp.array(15.0),
        'k_softmin': 7.211277690769142,
        'occupancy_weight': 4.060469919776791,
        'cohesion_weight': 1.192885663129121,
        'formation_radius': 19.943871757515588,
        'sigma_long': 4.2939688756721175,
        'sigma_wide': 2.7709911763599617,
        'cushion_dist': 2.1430822408443366,
        'learning_rate': 0.1,
        'sprint_penalty_weight': 85.28395263390695,
        'jko_lambda': 0.5,
        'sinkhorn_epsilon': 0.01,
        'velocity_cap': 0.8,
        'soft_velocity_cap': 0.6,
        'court_dims': [[0.0, 94.0], [0.0, 50.0]],
        'max_gradient_norm': 1.0,
        'acceleration_penalty_weight': 2.0,
        'velocity_penalty_weight': 1.0,
    }

    # 2. SUGGEST the IST variables
    params.update({
        'ist_q_exp': trial.suggest_float("ist_q_exp", 2.0, 4.0), 
        'ist_o_exp': trial.suggest_float("ist_o_exp", 1.0, 3.0), 
        'ist_weight': trial.suggest_float("ist_weight", 10, 40.0), 
        'ist_k_smooth': trial.suggest_float("ist_k_smooth", 5, 25.0),
    })

    
    total_ist_loss = 0.0
    total_smoothness_loss = 0.0
    valid_plays = 0

    basket_pos = jnp.array([5.25, 25.0]) # Standard basket position
    jko_steps = 20

    for _, row in data.iterrows(): # Assuming batch_data is a DataFrame
        play_data, q_traj = prepare_play_data(row)
        init_defenders, ball_traj, offenders_traj, real_def_traj, basket_pos = play_data

        simulated_def_traj = run_simulation(
            init_defenders,
            ball_traj,
            offenders_traj,
            q_traj,
            basket_pos,
            params,
            jko_num_steps=jko_steps
        )
    
        # 3. Calculate True IST Loss
        ist_loss, smoothness_loss = calculate_losses(
            simulated_def_traj, offenders_traj, ball_traj, basket_pos, q_traj, params
        )

        # --- THE FIX: Catch NaNs immediately per-play ---
        if jnp.isnan(ist_loss) or jnp.isnan(smoothness_loss):
            raise optuna.exceptions.TrialPruned()

        total_ist_loss += ist_loss
        total_smoothness_loss += smoothness_loss
        valid_plays += 1
    
    # Prune trial if the simulation results in NaN values (numerical instability)
    if jnp.isnan(ist_loss) or jnp.isnan(smoothness_loss):
        raise optuna.exceptions.TrialPruned()
    
    return (total_ist_loss / valid_plays), (total_smoothness_loss / valid_plays)


# --- Main Execution ---

if __name__ == "__main__":
    STUDY_NAME = "nba-defensive-optimization"
    STORAGE_NAME = "sqlite:///{}.db".format(STUDY_NAME)
    DATA_FILE = "datasets/0021500622_labeled.json"
    
    print("Loading data slice...")
    # Load data once outside the objective function for efficiency
    try:
        data_slice = load_data_slice(DATA_FILE, num_frames=50)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'datasets/0021500622_labeled.json' is available.")
        exit(1)
        
    print("Data loaded. Starting optimization...")

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
