import json
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import pandas as pd

import optuna

# Visualization
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Arc

# Import from our project modules
from .potentials import total_energy, params as default_params
from .solver import run_simulation, params as solver_params
from src.data_io.maps import load_maps_npz
from src.gradient_flows.utils import extract_quality_trajectories


# --- Data Handling ---
def load_viz_data(file_path, num_frames=150):
    """
    Loads a slice of NBA tracking data from the specified JSON file format,
    robustly handling player selection.
    
    It identifies the 5 players on each team who are closest to the ball
    in the first frame of the slice, assuming they are the active players.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Access the moments from the first event
    event = data['events'][0]
    moments = event['moments'][:num_frames] if num_frames else event['moments']

    # --- Parse the raw moment data into a structured format ---
    parsed_frames = []
    for i, mom in enumerate(moments):
        ball_data = mom[5][0]
        players_data = mom[5][1:]
        frame_obj = {
            'frame_id': i,
            'ball': {'x': ball_data[2], 'y': ball_data[3]},
            'players': [{'teamid': p[0], 'playerid': p[1], 'x': p[2], 'y': p[3]} for p in players_data]
        }
        parsed_frames.append(frame_obj)

    # Use pandas to simplify player selection
    df = pd.json_normalize(parsed_frames, record_path='players', meta=['frame_id', ['ball', 'x'], ['ball', 'y']])
    
    # Ensure all coordinate columns are numeric before any processing
    for col in ['x', 'y', 'ball.x', 'ball.y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['x', 'y', 'ball.x', 'ball.y'], inplace=True)

    # Identify the two teams
    team_ids = df['teamid'].unique()
    team_ids = [tid for tid in team_ids if tid != -1] # Filter out the ball's team id
    if len(team_ids) != 2:
        raise ValueError(f"Data slice does not contain exactly two teams. Found: {team_ids}")

    # Heuristic: Find 5 active players per team based on proximity to the ball in the first frame
    first_frame = df[df['frame_id'] == df['frame_id'].min()].copy()
    dist_sq = (first_frame['x'] - first_frame['ball.x']).pow(2) + (first_frame['y'] - first_frame['ball.y']).pow(2)
    first_frame['dist_to_ball'] = dist_sq.pow(0.5)

    active_players = first_frame.groupby('teamid').apply(lambda g: g.nsmallest(5, 'dist_to_ball')).reset_index(drop=True)
    active_player_ids = active_players['playerid'].unique()

    # Filter the full trajectory to only include active players
    df_active = df[df['playerid'].isin(active_player_ids)].copy()

    # Determine which team is on offense (closest to the ball)
    off_team_id = active_players.loc[active_players['dist_to_ball'].idxmin()]['teamid']
    def_team_id = team_ids[0] if team_ids[1] == off_team_id else team_ids[1]

    # Extract the actual player IDs for the 5 offensive players
    off_pids = active_players[active_players['teamid'] == off_team_id]['playerid'].tolist()
    
    # Pivot to get trajectories, ensuring consistent player order
    df_active = df_active.sort_values(['frame_id', 'teamid', 'playerid'])

    # Ensure all frames have 5 players for each team
    frame_counts = df_active.groupby(['frame_id', 'teamid']).size().unstack(fill_value=0)
    valid_frames = frame_counts[(frame_counts[def_team_id] == 5) & (frame_counts[off_team_id] == 5)].index

    if len(valid_frames) == 0:
        raise ValueError("Could not find any valid 5v5 frames in the provided slice.")

    df_final = df_active[df_active['frame_id'].isin(valid_frames)]

    def_traj = df_final[df_final['teamid'] == def_team_id][['x', 'y']].values.reshape(-1, 5, 2)
    off_traj = df_final[df_final['teamid'] == off_team_id][['x', 'y']].values.reshape(-1, 5, 2)

    ball_traj_df = df_final[['frame_id', 'ball.x', 'ball.y']].drop_duplicates().sort_values('frame_id')
    ball_traj = ball_traj_df[['ball.x', 'ball.y']].values

    return jnp.array(def_traj), jnp.array(off_traj), jnp.array(ball_traj), off_pids


# --- Court Drawing Helpers ---

def draw_court_matplotlib(ax=None):
    """Draws a basketball court on a Matplotlib axis."""
    if ax is None:
        ax = plt.gca()

    # Court lines
    ax.add_patch(Rectangle((0, 0), 94, 50, facecolor='none', edgecolor='black', lw=2))
    ax.plot([47, 47], [0, 50], color='black', lw=2) # Half court line
    
    # Hoops
    ax.add_patch(Circle((5.25, 25), radius=.75, facecolor='none', edgecolor='black', lw=2))
    ax.add_patch(Circle((94-5.25, 25), radius=.75, facecolor='none', edgecolor='black', lw=2))
    
    # 3-point lines
    ax.add_patch(Arc((5.25, 25), 47.5, 47.5, theta1=270, theta2=90, color='black', lw=2))
    ax.add_patch(Arc((94-5.25, 25), 47.5, 47.5, theta1=90, theta2=270, color='black', lw=2))
    
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal', adjustable='box')
    return ax

def get_court_shapes_plotly():
    """Returns a list of shapes for drawing a basketball court in Plotly."""
    shapes = []
    # Court outline and center line
    shapes.append(dict(type="rect", x0=0, y0=0, x1=94, y1=50, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=47, y0=0, x1=47, y1=50, line=dict(color="black", width=2)))
    
    # Hoops
    shapes.append(dict(type="circle", x0=4.5, y0=24.25, x1=6, y1=25.75, line=dict(color="black", width=2)))
    shapes.append(dict(type="circle", x0=94-6, y0=24.25, x1=94-4.5, y1=25.75, line=dict(color="black", width=2)))

    # 3-point arcs using SVG paths
    r = 23.75  # 3pt radius
    shapes.append(dict(type="path", path=f"M {r+5.25},47.5 A {r},{r} 0 0,0 {r+5.25},2.5", line=dict(color="black", width=2)))
    shapes.append(dict(type="path", path=f"M {94-(r+5.25)},47.5 A {r},{r} 0 0,1 {94-(r+5.25)},2.5", line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=5.25, y0=2.5, x1=5.25+r, y1=2.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=5.25, y0=47.5, x1=5.25+r, y1=47.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=94-5.25, y0=2.5, x1=94-(r+5.25), y1=2.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=94-5.25, y0=47.5, x1=94-(r+5.25), y1=47.5, line=dict(color="black", width=2)))
    return shapes

# --- Visualization Functions ---

def plot_speed_analysis(sim_traj, real_traj=None, filename="speed_analysis.png", fps=25.0):
    """
    Calculates and plots the speed (feet/sec) of each defender over time.
    Compares Simulated 'Ghosts' vs Real Humans if real_traj is provided.
    
    Args:
        sim_traj: (T, 5, 2) JAX array of simulated positions.
        real_traj: (T, 5, 2) JAX/Numpy array of real positions (optional).
        filename: Output filename for the plot.
        fps: Frames per second of the tracking data (usually 25 for SportVU).
    """
    def calculate_speeds(trajectory):
        # Velocity = dx/dt. Speed = norm(Velocity).
        # We calculate the difference between consecutive frames.
        delta = trajectory[1:] - trajectory[:-1]
        dist_per_frame = jnp.linalg.norm(delta, axis=-1)
        # Convert to feet/second
        speeds = dist_per_frame * fps
        return speeds

    sim_speeds = calculate_speeds(sim_traj)
    
    # Setup plot
    rows = 2 if real_traj is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, 6 * rows), sharex=True, sharey=True)
    if rows == 1: axes = [axes]
    
    time_axis = np.arange(len(sim_speeds)) / fps

    # Plot Simulated Speeds
    ax_sim = axes[0]
    for i in range(5):
        ax_sim.plot(time_axis, sim_speeds[:, i], label=f'Sim Defender {i}', alpha=0.7)
    
    # Add reference lines
    ax_sim.axhline(y=15.0, color='orange', linestyle='--', label='Soft Cap (15 ft/s)')
    ax_sim.axhline(y=20.0, color='red', linestyle='--', label='Phys. Limit (20 ft/s)')
    
    ax_sim.set_title("Simulated Defenders Speed")
    ax_sim.set_ylabel("Speed (ft/s)")
    ax_sim.grid(True, alpha=0.3)
    ax_sim.legend(loc='upper right')

    # Plot Real Speeds (if provided)
    if real_traj is not None:
        real_speeds = calculate_speeds(real_traj)
        ax_real = axes[1]
        for i in range(5):
            ax_real.plot(time_axis, real_speeds[:, i], label=f'Real Defender {i}', alpha=0.7)
            
        ax_real.axhline(y=15.0, color='orange', linestyle='--', label='Soft Cap')
        ax_real.axhline(y=20.0, color='red', linestyle='--', label='Phys. Limit')
        
        ax_real.set_title("Real Human Defenders Speed")
        ax_real.set_ylabel("Speed (ft/s)")
        ax_real.set_xlabel("Time (seconds)")
        ax_real.grid(True, alpha=0.3)
        ax_real.legend(loc='upper right')
    else:
        ax_sim.set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved speed analysis plot to {filename}")


def plot_potential_surface(defenders, offenders, ball, basket, params, filename="potential_heatmap.png"):
    """
    Computes and plots the total_energy surface for a single defender.
    """
    # Create a grid for the court
    xx, yy = jnp.meshgrid(jnp.linspace(0, 94, 100), jnp.linspace(0, 50, 50))
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    # Assume we're plotting the potential for the first defender
    fixed_defenders = defenders[1:]
    
    # Function to compute energy at one point for one defender
    def get_energy_at_point(point):
        # Create the full set of 5 defenders for the energy function
        current_defenders = jnp.vstack([point, fixed_defenders])
        # We need the energy contribution for the first defender only
        energies = total_energy(current_defenders, offenders, ball, basket, params)
        return energies[0]

    # Compute energy across the entire grid
    energies = vmap(get_energy_at_point)(grid_points)
    zz = energies.reshape(xx.shape)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_court_matplotlib(ax)
    
    contour = ax.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.7)
    fig.colorbar(contour, ax=ax, orientation='vertical', label='Potential Energy')
    
    # Overlay players
    ax.scatter(defenders[:, 0], defenders[:, 1], c='blue', s=100, label='Defenders')
    ax.scatter(offenders[:, 0], offenders[:, 1], c='red', s=100, label='Offenders')
    ax.scatter(ball[0], ball[1], c='orange', s=120, marker='*', label='Ball')
    
    ax.set_title("Defensive Potential Surface (for Defender 0)")
    ax.legend()
    plt.savefig(filename)
    print(f"Saved potential surface plot to {filename}")


def save_simulation_gif(sim_traj, real_traj, ball_traj, offenders_traj, filename='sim.gif'):
    """
    Generates and saves a GIF comparing real vs. simulated defense side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Defensive Simulation vs. Reality', fontsize=16)

    # Plot 1: Real Defense
    draw_court_matplotlib(ax1)
    ax1.set_title('Real Defense')
    real_def_sc = ax1.scatter([], [], c='blue', s=120, label='Defenders')
    real_off_sc = ax1.scatter([], [], c='red', s=120, label='Offenders')
    real_ball_sc = ax1.scatter([], [], c='orange', marker='o', s=80, label='Ball')
    ax1.legend(loc='upper right')

    # Plot 2: Simulated Defense
    draw_court_matplotlib(ax2)
    ax2.set_title('Simulated (JKO)')
    sim_def_sc = ax2.scatter([], [], c='blue', alpha=0.7, s=120, label='Simulated Defenders')
    sim_off_sc = ax2.scatter([], [], c='red', s=120, label='Offenders')
    sim_ball_sc = ax2.scatter([], [], c='orange', marker='o', s=80, label='Ball')
    ax2.legend(loc='upper right')

    def update(frame):
        # Update Real Plot
        real_def_sc.set_offsets(real_traj[frame])
        real_off_sc.set_offsets(offenders_traj[frame])
        real_ball_sc.set_offsets(ball_traj[frame])
        
        # Update Sim Plot
        sim_def_sc.set_offsets(sim_traj[frame])
        sim_off_sc.set_offsets(offenders_traj[frame])
        sim_ball_sc.set_offsets(ball_traj[frame])
        
        return real_def_sc, real_off_sc, real_ball_sc, sim_def_sc, sim_off_sc, sim_ball_sc

    anim = FuncAnimation(fig, update, frames=len(sim_traj), interval=40, blit=True)
    anim.save(filename, writer='pillow')
    print(f"Saved simulation GIF to {filename}")


def create_interactive_plot(sim_traj, real_traj, ball_traj, offenders_traj):
    """Creates an interactive Plotly figure comparing real vs. simulated defense."""
    from plotly.subplots import make_subplots
    pio.renderers.default = "browser"

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Real Defense', 'Simulated (JKO)'))

    # Initial Traces
    fig.add_trace(go.Scatter(x=real_traj[0, :, 0], y=real_traj[0, :, 1], mode='markers', marker=dict(color='blue', size=15), name='Real Defenders'), row=1, col=1)
    fig.add_trace(go.Scatter(x=offenders_traj[0, :, 0], y=offenders_traj[0, :, 1], mode='markers', marker=dict(color='red', size=15), name='Offenders'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[ball_traj[0, 0]], y=[ball_traj[0, 1]], mode='markers', marker=dict(color='orange', size=12), name='Ball'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=sim_traj[0, :, 0], y=sim_traj[0, :, 1], mode='markers', marker=dict(color='blue', size=15, opacity=0.6), name='Sim Defenders'), row=1, col=2)
    fig.add_trace(go.Scatter(x=offenders_traj[0, :, 0], y=offenders_traj[0, :, 1], mode='markers', marker=dict(color='red', size=15), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[ball_traj[0, 0]], y=[ball_traj[0, 1]], mode='markers', marker=dict(color='orange', size=12), showlegend=False), row=1, col=2)

    # Create frames
    frames = [go.Frame(data=[
        go.Scatter(x=real_traj[k, :, 0], y=real_traj[k, :, 1]),
        go.Scatter(x=offenders_traj[k, :, 0], y=offenders_traj[k, :, 1]),
        go.Scatter(x=[ball_traj[k, 0]], y=[ball_traj[k, 1]]),
        go.Scatter(x=sim_traj[k, :, 0], y=sim_traj[k, :, 1]),
        go.Scatter(x=offenders_traj[k, :, 0], y=offenders_traj[k, :, 1]),
        go.Scatter(x=[ball_traj[k, 0]], y=[ball_traj[k, 1]]),
    ], name=str(k)) for k in range(len(sim_traj))]
    
    fig.frames = frames

    # Animation controls
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}])]
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))]) for f in fig.frames],
            transition=dict(duration=0),
            x=0,
            xanchor="left",
            len=1,
        )]
    )

    # General layout
    court_shapes = get_court_shapes_plotly()
    fig.update_layout(
        title_text="Defensive Simulation vs. Reality",
        shapes=court_shapes
    )
    fig.update_xaxes(range=[0, 94], autorange=False)
    fig.update_yaxes(range=[0, 50], autorange=False, scaleanchor="x1", scaleratio=1)
    
    return fig
if __name__ == '__main__':
    DATA_FILE = "datasets/0021500622_labeled.json"
    STUDY_NAME = "nba-defensive-optimization"
    STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

    # --- Define a complete set of default and fixed parameters ---
    solver_params = {
        **default_params,
        'jko_lambda': 0.5,
        'sinkhorn_epsilon': 2.0,
        'learning_rate': 0.1, # Default learning rate
        'velocity_cap': 0.8,
        'court_dims': [[0., 94.], [0., 50.]],
        'max_gradient_norm': 1.0,
        'acceleration_penalty_weight': 2.0,
        'velocity_penalty_weight': 1.0,
    }

    # --- Load learned hyperparameters and update the params ---
    print(f"\nLoading best parameters from Optuna study '{STUDY_NAME}'...")
    try:
        import optuna
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
        # Select the trial with the best "pressure" score (the first objective)
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        print(f"Using parameters from Trial #{best_trial.number} with values: {best_trial.values}")
        # Update the solver_params with the learned values
        solver_params.update(best_trial.params)
    except (FileNotFoundError, ValueError, IndexError, ImportError):
        print("Could not load study or find best trial. Using default parameters.")
        
    # --- Load Data ---
    print(f"Loading data from {DATA_FILE}...")
    real_def_traj, off_traj, ball_traj, off_pids = load_viz_data(DATA_FILE, num_frames=None)

    # --- Calculate Offender Weights --- 
    print("Loading maps and calculating offender threat weights...")
    maps, meta = load_maps_npz("../datasets/maps_1ft_xpps.npz")
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

    # Combine into final weights
    offender_weights_traj = q_traj * b_traj

    # --- Run Simulation ---
    init_defenders = real_def_traj[0]
    print(f"\nRunning simulation for {len(ball_traj)} timesteps...")
    
    sim_def_traj = run_simulation(
        init_defenders,
        ball_traj,
        off_traj,
        offender_weights_traj,
        basket_pos=jnp.array([5.25, 25.0]),
        params=solver_params,
        jko_num_steps=20
    )

    # --- Generate Visualizations ---
    print("\n1. Generating GIF...")
    save_simulation_gif(
        sim_def_traj,
        real_def_traj,
        ball_traj,
        off_traj,
        filename="results/nba_simulation_viz.gif"
    )
    
    print("\n2. Generating Speed Analysis...")
    plot_speed_analysis(
        sim_def_traj,
        real_def_traj,
        filename="results/speed_analysis.png",
        fps=25.0
    )

    print("\n3. Generating Potential Heatmap...")
    last_frame_idx = len(sim_def_traj) - 1
    plot_potential_surface(
        defenders=sim_def_traj[last_frame_idx],
        offenders=off_traj[last_frame_idx],
        ball=ball_traj[last_frame_idx],
        basket=jnp.array([5.25, 25.0]),
        params=solver_params,
        filename="results/potential_heatmap.png"
    )

    print("\n4. Creating Interactive Plot...")
    interactive_fig = create_interactive_plot(
        sim_def_traj,
        real_def_traj,
        ball_traj,
        off_traj,
    )
    interactive_fig.write_html("interactive_simulation.html")
    print("Saved interactive plot to interactive_simulation.html. Open this file in a browser.")