import argparse
import pandas as pd
import jax
import jax.numpy as jnp
import optuna
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Import from our existing modules
from .potentials import params as default_params
from .solver import run_simulation
from .utils import prepare_play_data

def get_baseline_params(study_name, storage_name):
    """Loads the optimal kinematic baseline and enforces strict solver stability."""
    merged_params = default_params.copy()
    
    # --- SAFETY NET 1: Ultra-Stable Solver Dynamics ---
    solver_defaults = {
        'learning_rate': 0.05,       
        'jko_lambda': 0.5,
        'sinkhorn_epsilon': 0.5,     # Aggressively raised to ensure Wasserstein NEVER collapses
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
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        print(f"[*] Loaded Stage 1 Kinematic Baseline (Trial #{best_trial.number}).")
        merged_params.update(best_trial.params)
    except Exception as e:
        print(f"[!] Warning: Could not load previous study. Using defaults. ({e})")
        
    return merged_params

@jax.jit
def calculate_losses(def_traj, off_traj, ball_traj, basket_pos, q_traj, params):
    """Calculates the True IST of the simulated defense to use as the loss metric."""
    def ist_at_t(off_pos, def_pos, ball_pos, q_vals):
        ball_dist = jnp.linalg.norm(off_pos - ball_pos, axis=1)
        b_val = 0.4 + 0.6 * (1.0 / (1.0 + jnp.exp(0.3 * (ball_dist - 18.0))))
        sim_weight = jnp.maximum((q_vals ** params.get('ist_q_exp', 2.16)) * b_val, 0.35)
        
        # Safe distance calculation
        dists_to_defs = jnp.sqrt(jnp.sum(jnp.square(def_pos - off_pos[:, None, :]), axis=2) + 1e-6)
        closest_dist = jnp.min(dists_to_defs, axis=1)
        openness = jnp.clip(closest_dist / 6.0, 0.5, 1.5)
        
        return jnp.sum(sim_weight * (openness ** params.get('ist_o_exp', 1.03)))

    mean_ist_loss = jnp.mean(jax.vmap(ist_at_t)(off_traj, def_traj, ball_traj, q_traj))
    smoothness_loss = jnp.mean(jnp.sqrt(jnp.sum(jnp.square(jnp.diff(def_traj, axis=0)), axis=2) + 1e-6))
    return mean_ist_loss, smoothness_loss

def evaluate_performance(data, params, jko_steps=15):
    """Core simulation loop used for both Training and Testing."""
    total_ist_loss = 0.0
    total_smoothness_loss = 0.0
    valid_plays = 0

    for idx, row in data.iterrows(): 
        play_data, q_traj = prepare_play_data(row)
        
        q_traj = jnp.maximum(q_traj, 0.001)
        init_defenders, ball_traj, offenders_traj, real_def_traj, basket_pos = play_data

        simulated_def_traj = run_simulation(
            init_defenders, ball_traj, offenders_traj, q_traj, basket_pos, params, jko_num_steps=jko_steps
        )
    
        ist_loss, smoothness_loss = calculate_losses(
            simulated_def_traj, offenders_traj, ball_traj, basket_pos, q_traj, params
        )

        if jnp.isnan(ist_loss) or jnp.isnan(smoothness_loss):
            print(f"\n[!!!] POISON PILL DETECTED: Play {row['SHOT_EVENT_ID']} caused a NaN collapse!")
            print(f"      IST Loss: {ist_loss} | Smoothness Loss: {smoothness_loss}")
            return float('inf'), float('inf') # Flag numerical instability gracefully

        total_ist_loss += ist_loss
        total_smoothness_loss += smoothness_loss
        valid_plays += 1
    
    if valid_plays == 0: 
        return float('inf'), float('inf')
        
    return (total_ist_loss / valid_plays), (total_smoothness_loss / valid_plays)


def objective(trial, train_data, base_params):
    """The Optuna objective function to be minimized."""
    params = base_params.copy()
    
    # weights to test
    params['ist_weight'] = trial.suggest_float("ist_weight", 1.0, 20.0) 
    params['ist_k_smooth'] = trial.suggest_float("ist_k_smooth", 2.0, 20.0)
    
    ist_loss, smooth_loss = evaluate_performance(train_data, params)
    
    if ist_loss == float('inf'):
        raise optuna.exceptions.TrialPruned()
        
    return ist_loss, smooth_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 JKO Optimization (Threat Tuning)")
    parser.add_argument("--demo", action="store_true", help="Run a fast demo on a controlled section of possessions")
    parser.add_argument("--trials", type=int, default=None, help="Number of Optuna tuning trials to run")
    args, _ = parser.parse_known_args()

    # --- 1. STAGE 1 BASELINE PATH ---
    STAGE_1_DB = "sqlite:///data/processed/optimization/nba-defensive-optimization.db"
    STAGE_1_STUDY = "nba-defensive-optimization"

    # --- 2. DYNAMIC PATHING BASED ON MODE ---
    if args.demo:
        print("[*] DEMO MODE: Loading demo_traj.parquet...")
        INPUT_FILE = "data/demo/processed/demo_traj.parquet"
        
        os.makedirs("data/demo/processed", exist_ok=True)
        DEMO_DB = "sqlite:///data/demo/processed/demo-ist-tuning.db"
        DEMO_STUDY = "demo-ist-tuning"
        N_TRIALS = args.trials if args.trials is not None else 20

    else:
        print("[*] FULL MODE: Loading full season trajectories...")
        INPUT_FILE = "data/processed/traj_features/all_season_traj.parquet"
        
        os.makedirs("data/processed/optimization", exist_ok=True)
        DEMO_DB = "sqlite:///data/processed/optimization/stage2-ist-tuning.db" 
        DEMO_STUDY = "stage2-ist-tuning"
        N_TRIALS = args.trials if args.trials is not None else 100

    if not os.path.exists(INPUT_FILE):
        print(f"[!] Critical Error: {INPUT_FILE} not found.")
        exit(1)
        
    df = pd.read_parquet(INPUT_FILE)
    
    if args.demo:
        bad_plays = [284, 321, 503]
        df = df[~df['SHOT_EVENT_ID'].isin(bad_plays)].copy()
    
    print(f"[*] Loaded {len(df)} total plays. Deep-cleaning any corrupted tracking frames...")
    def has_no_nans(row):
        for val in row.values:
            if isinstance(val, (list, np.ndarray)):
                if pd.isna(val).any(): return False
            elif pd.isna(val): return False
        return True
        
    clean_mask = df.apply(has_no_nans, axis=1)
    df = df[clean_mask].copy()
 
    # --- THE CONTROLLED SUBSET ---
    # 1. Grab the unique play IDs so we don't split frames from the same play
    unique_plays = df['SHOT_EVENT_ID'].unique()
    
    # 2. Split the IDs (Train on 70% of the plays, Test on 30%)
    train_play_ids, test_play_ids = train_test_split(unique_plays, test_size=0.3, random_state=42)

    # 3. Filter the master dataframe using those cleanly separated IDs
    train_data = df[df['SHOT_EVENT_ID'].isin(train_play_ids)].copy()
    test_data = df[df['SHOT_EVENT_ID'].isin(test_play_ids)].copy()
    
    print(f"[*] Split Data: {len(train_play_ids)} Training Plays | {len(test_play_ids)} Testing Plays\n")

    print("-" * 55)
    print("  STAGE 2: STRATEGIC THREAT TUNING SANDBOX")
    print("-" * 55)
    print(" Optimizing two key parameters:")
    print(" 1. ist_weight:   How aggressively defenders panic and swarm the ball.")
    print(" 2. ist_k_smooth: How strictly defenders play 1-on-1 vs. providing help defense.\n")

    base_params = get_baseline_params(STAGE_1_STUDY, STAGE_1_DB)

    print("\n[*] Launching Optuna Trials (TRAINING)...")
    study = optuna.create_study(
        study_name=DEMO_STUDY,
        storage=DEMO_DB,
        directions=["minimize", "minimize"], 
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, train_data, base_params), n_trials=N_TRIALS)

    print("\n" + "="*50)
    print("      STAGE 2 TUNING COMPLETE")
    print("="*50)
    
    if len(study.best_trials) == 0:
        print("[!] ERROR: All trials pruned. Physics engine failed to converge on the subset.")
        exit(1)
        
    best_trial = min(study.best_trials, key=lambda t: t.values[0])
    print(f"\n[*] Best Training Parameters Found:")
    print(f"    ist_weight={best_trial.params['ist_weight']:>5.2f}, ist_k_smooth={best_trial.params['ist_k_smooth']:>5.2f}")
    print(f"    Train IST Loss: {best_trial.values[0]:.3f} | Train Smoothness: {best_trial.values[1]:.3f}")

    print("\n[*] Running Out-Of-Sample Evaluation on Unseen Test Plays...")
    final_params = base_params.copy()
    final_params.update(best_trial.params)
    
    test_ist_loss, test_smoothness_loss = evaluate_performance(test_data, final_params)
    
    print(f"    Test IST Loss:  {test_ist_loss:.3f}")
    print(f"    Test Smoothness: {test_smoothness_loss:.3f}")

    print("\n[*] Opening Pareto Front Visualization...")
    try:
        import optuna.visualization as vis
        fig = vis.plot_pareto_front(study, target_names=["IST Threat (Lower is safer)", "Jerkiness (Lower is smoother)"])
        fig.show()
    except ImportError:
        pass
