import optuna
import pandas as pd
import numpy as np
from ipywidgets import interact, IntSlider, Dropdown
import ipywidgets as widgets
from IPython.display import display, clear_output
import optuna
import optuna.visualization as vis
import os


from src.gradient_flows.utils import extract_trajectories_from_row
from src.gradient_flows.potentials import params as default_params

def generate_optimization_viz(db_path, study_name, target_names=None):
    if target_names is None:
        target_names = ["IST Threat", "Smoothness"]

    try:
        study = optuna.load_study(study_name=study_name, storage=db_path)
        
        # 1. Use the base Optuna viz
        fig = vis.plot_pareto_front(
            study, 
            target_names=target_names,
            include_dominated_trials=True
        )

        # 2. FORCE LIGHT MODE & VISIBILITY
        fig.update_layout(
            template="plotly_white",        # This kills the dark background immediately
            title={'text': f"OPTIMIZATION TRIALS: {study_name.upper()}", 'x': 0.5},
            paper_bgcolor='white',          # Clean white outer border
            plot_bgcolor='#F8F9FA',         # Very light grey for the grid area (depth)
            
            # Make the text pop (Naval Blue)
            font=dict(family="Arial Black", size=14, color='#1D428A'),
            
            # Grid line visibility
            xaxis=dict(showgrid=True, gridcolor='#DDDDDD', zerolinecolor='black'),
            yaxis=dict(showgrid=True, gridcolor='#DDDDDD', zerolinecolor='black'),
            
            # Size for demo
            width=900,
            height=600
        )
        
        # 3. Make the Pareto points larger and easier to see
        # This targets the scatter traces
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        
        return fig

    except Exception as e:
        print(f"Error generating Pareto Front: {e}")
        return None

# --- Usage Example ---
DB_URL = "sqlite:///../data/demo/processed/demo-ist-tuning.db"
TRAJ_FILE = "../data/demo/processed/demo_traj.parquet"

pareto_fig = generate_optimization_viz(DB_URL, "demo-ist-tuning", TRAJ_FILE)

if pareto_fig:
    # Save as HTML so it's interactive for the demo
    pareto_fig.write_html("pareto_optimization_results.html")
    pareto_fig.show()



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

def audit_final_results(df, triple_animate_fn):
    """
    Dashboard for browsing the final triple-comparison results.
    Directly feeds 3D arrays from the DataFrame to the Plotly engine.
    """
    
    def run_audit(play_idx):
        row = df.iloc[play_idx]
        num_f = len(row['ball_x_traj'])
        
        # Initialize the empty arrays
        off_traj = np.zeros((num_f, 5, 2))
        real_def_traj = np.zeros((num_f, 5, 2))
        base_def_traj = np.zeros((num_f, 5, 2))
        sim_def_traj = np.zeros((num_f, 5, 2))
        
        ist_real = np.zeros((num_f, 5))
        ist_base = np.zeros((num_f, 5))
        ist_sim = np.zeros((num_f, 5))
        
        # Reconstruct ALL data from their safe 1D columns
        # Reconstruct ALL data, swapping X and Y so they run left-to-right!
        for i in range(1, 6):
            # 1. Offense
            off_traj[:, i-1, 0] = np.array(row[f'off{i}_y_traj']) + 5.25
            off_traj[:, i-1, 1] = np.array(row[f'off{i}_x_traj']) + 25.0
            
            # 2. Real Defense
            real_def_traj[:, i-1, 0] = np.array(row[f'def{i}_y_traj']) + 5.25
            real_def_traj[:, i-1, 1] = np.array(row[f'def{i}_x_traj']) + 25.0
            
            # 3. Baseline Simulation
            base_def_traj[:, i-1, 0] = np.array(row[f'base_def{i}_y_traj']) + 5.25
            base_def_traj[:, i-1, 1] = np.array(row[f'base_def{i}_x_traj']) + 25.0
            
            # 4. Optimized Simulation
            sim_def_traj[:, i-1, 0] = np.array(row[f'sim_def{i}_y_traj']) + 5.25
            sim_def_traj[:, i-1, 1] = np.array(row[f'sim_def{i}_x_traj']) + 25.0
            
            # 5. IST Values (Keep identical)
            ist_real[:, i-1] = row[f'ist_real_{i}']
            ist_base[:, i-1] = row[f'ist_base_{i}']
            ist_sim[:, i-1] = row[f'ist_sim_{i}']

        # 6. Extract Ball exactly the same way
        ball_traj = np.column_stack([
            np.array(row['ball_y_traj']) + 5.25, 
            np.array(row['ball_x_traj']) + 25.0
        ])
        
        # Launch Animation
        fig = triple_animate_fn(
            off_traj=off_traj,
            real_def_traj=real_def_traj,
            base_def_traj=base_def_traj,   
            sim_def_traj=sim_def_traj,     
            ball_traj=ball_traj,
            ist_real=ist_real,     
            ist_base=ist_base,     
            ist_sim=ist_sim,       
            half_court='left' # Now you can safely lock this to 'left' forever!
        )
        
        fig.show()

    interact(
        run_audit, 
        play_idx=IntSlider(min=0, max=len(df)-1, value=0, description='Select Play:')
    )