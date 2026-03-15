import optuna
import pandas as pd
import numpy as np
from ipywidgets import interact, IntSlider, Dropdown
import ipywidgets as widgets
from IPython.display import display, clear_output

from src.gradient_flows.utils import extract_trajectories_from_row
from src.gradient_flows.potentials import params as default_params

def get_pareto_trials(db_path, study_name):
    """
    Fetches the best trade-off trials (Pareto front) from the database.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=db_path)
        trials = study.best_trials
        
        data = []
        for t in trials:
            row = {
                "trial_no": t.number, 
                "threat_reduction": round(t.values[0], 4), 
                "smoothness": round(t.values[1], 4)
            }
            row.update(t.params)
            data.append(row)
            
        return pd.DataFrame(data).sort_values("threat_reduction")
    except Exception as e:
        print(f"Error loading trials: {e}")
        return pd.DataFrame().reset_index()

def run_trial_simulation(trial_no, play_index, df, db_path, study_name, animate_fn):
    """
    Core logic to simulate a specific play using a specific trial's parameters.
    """
    # 1. Load trial parameters
    study = optuna.load_study(study_name=study_name, storage=db_path)
    trial = [t for t in study.trials if t.number == trial_no][0]
    
    params = default_params.copy()
    params.update(trial.params)
    
    # 2. Extract play data
    row = df.iloc[play_index]
    
    # 3. Run physics simulation
    res = extract_trajectories_from_row(row, params, pad_for_bulk=False)
    
    # Unpack the 11-variable result
    # EXACT UNPACKING for your 11-return function:
    (sim_def_no_ist_traj,   # index 0
     sim_def_ist_traj,      # index 1
     real_def_traj,         # index 2
     weights_sim_no_ist,    # index 3
     weights_sim_ist,       # index 4 (Team/Player Threat)
     weights_real,          # index 5 (Team/Player Threat)
     off_traj,              # index 6
     ball_traj,             # index 7
     basket_pos,            # index 8
     off_ids,               # index 9
     shot_frame             # index 10
    ) = res
    
    # 4. Generate the visualization
    fig = animate_fn(
        off_traj, 
        real_def_traj, 
        sim_def_ist_traj, 
        ball_traj, 
        weights_real, 
        weights_sim_ist
    )
    fig.show()

def create_trial_browser(df, db_path, study_name, animate_fn):
    """
    Generates the interactive UI for the notebook.
    """
    best_trials = get_pareto_trials(db_path, study_name)
    
    if best_trials.empty:
        print("No trials found in the database.")
        return

    print("--- Best Pareto Trials ---")
    display(best_trials)

    interact(
        lambda trial_no, play_idx: run_trial_simulation(
            trial_no, play_idx, df, db_path, study_name, animate_fn
        ), 
        trial_no=Dropdown(options=best_trials['trial_no'].tolist(), description='Optuna Trial:'),
        play_idx=IntSlider(min=0, max=len(df)-1, step=1, value=0, description='Play Index:')
    )