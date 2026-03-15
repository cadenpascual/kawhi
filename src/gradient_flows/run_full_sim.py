import json
import jax
import jax.numpy as jnp
import numpy as np
import optuna
import os
from tqdm import tqdm

from src.gradient_flows.potentials import total_energy, params as default_params
from src.gradient_flows.solver import run_simulation

def get_best_params(study_name, storage_name):
    """Loads the best parameters from an Optuna study if available."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        print(f"Loaded best parameters from Trial #{best_trial.number}")
        return best_trial.params
    except Exception as e:
        print(f"Could not load best parameters: {e}. Using defaults.")
        return {}

def process_event(event, params):
    """Runs simulation for a single event and replaces defender data."""
    frames = event.get('frames', [])
    if not frames:
        return event

    # 1. Identify teams and active players (5v5)
    # We look at the first frame to determine who is playing
    first_frame = frames[0]
    players = first_frame.get('players', [])
    if len(players) < 10:
        return event # Skip if not enough players

    team_ids = list(set(p['teamid'] for p in players))
    if len(team_ids) != 2:
        return event

    # Heuristic: Player closest to ball is on offense
    ball_pos = np.array([first_frame['ball']['x'], first_frame['ball']['y']])
    min_dist = float('inf')
    off_team_id = -1
    for p in players:
        dist = np.linalg.norm(ball_pos - np.array([p['x'], p['y']]))
        if dist < min_dist:
            min_dist = dist
            off_team_id = p['teamid']
    
    def_team_id = team_ids[0] if team_ids[1] == off_team_id else team_ids[1]

    # Filter for exactly 5 players per team (closest to ball if more exist)
    def get_top_5(frame_players, team_id, b_pos):
        team_players = [p for p in frame_players if p['teamid'] == team_id]
        team_players.sort(key=lambda p: np.linalg.norm(np.array([p['x'], p['y']]) - b_pos))
        return [p['playerid'] for p in team_players[:5]]

    off_player_ids = get_top_5(players, off_team_id, ball_pos)
    def_player_ids = get_top_5(players, def_team_id, ball_pos)

    # 2. Extract Trajectories
    ball_traj = []
    off_traj = []
    def_init = []
    
    valid_frames_indices = []
    for i, frame in enumerate(frames):
        ps = {p['playerid']: p for p in frame['players']}
        # Check if all 10 players are present in this frame
        if all(pid in ps for pid in off_player_ids) and all(pid in ps for pid in def_player_ids):
            ball_traj.append([frame['ball']['x'], frame['ball']['y']])
            off_traj.append([[ps[pid]['x'], ps[pid]['y']] for pid in off_player_ids])
            if not def_init:
                def_init = [[ps[pid]['x'], ps[pid]['y']] for pid in def_player_ids]
            valid_frames_indices.append(i)

    if len(ball_traj) < 2:
        return event

    # 3. Run Simulation
    # Combine params
    sim_params = {
        **default_params,
        **params,
        'jko_lambda': 0.5,
        'sinkhorn_epsilon': 0.01,
        'velocity_cap': 0.8,
        'court_dims': [[0., 94.], [0., 50.]],
        'max_gradient_norm': 1.0,
        'acceleration_penalty_weight': 2.0,
        'velocity_penalty_weight': 0.5,
    }
    if 'learning_rate' not in sim_params:
        sim_params['learning_rate'] = 0.1

    sim_def_traj = run_simulation(
        jnp.array(def_init),
        jnp.array(ball_traj),
        jnp.array(off_traj),
        basket_pos=jnp.array([5.25, 25.0]),
        params=sim_params,
        jko_num_steps=20
    )

    # 4. Map back to JSON
    sim_def_traj_np = np.array(sim_def_traj)
    for t_idx, frame_idx in enumerate(valid_frames_indices):
        frame = event['frames'][frame_idx]
        ps_dict = {p['playerid']: p for p in frame['players']}
        for p_idx, pid in enumerate(def_player_ids):
            ps_dict[pid]['x'] = float(sim_def_traj_np[t_idx, p_idx, 0])
            ps_dict[pid]['y'] = float(sim_def_traj_np[t_idx, p_idx, 1])
            ps_dict[pid]['simulated'] = True # Mark as simulated

    return event

def main():
    INPUT_FILE = "datasets/0021500622_labeled.json"
    OUTPUT_FILE = "results/0021500622_simulated.json"
    STUDY_NAME = "nba-defensive-optimization"
    STORAGE_NAME = "sqlite:///nba-defensive-optimization.db"

    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    learned_params = get_best_params(STUDY_NAME, STORAGE_NAME)

    print("Starting full simulation...")
    simulated_data = []
    for event in tqdm(data):
        sim_event = process_event(event, learned_params)
        simulated_data.append(sim_event)

    print(f"Saving simulated data to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(simulated_data, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
