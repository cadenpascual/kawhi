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

# Import from your project modules (Ensure these are in the same directory)
from potentials import total_energy, params as default_params
from solver import run_simulation, params as solver_params


# --- 1. Math & Feature Helpers ---

def calculate_ist_tensor(off_traj, def_traj, q_traj):
    """
    Computes IST for all players across all frames using JAX broadcasting.
    off_traj shape: (frames, 5, 2)
    def_traj shape: (frames, 5, 2)
    q_traj shape:   (frames, 5)
    """
    # 1. Calculate distances from every offender to every defender
    # off shape: (frames, 5, 1, 2) | def shape: (frames, 1, 5, 2)
    dx = off_traj[:, :, None, 0] - def_traj[:, None, :, 0]
    dy = off_traj[:, :, None, 1] - def_traj[:, None, :, 1]
    
    distances = jnp.sqrt(dx**2 + dy**2)
    
    # 2. Find Openness (distance to nearest defender)
    openness = jnp.min(distances, axis=2) # Shape: (frames, 5)
    
    # 3. Calculate IST: (Q^2.16) * (O^1.03)
    ist = (q_traj ** 2.16) * (openness ** 1.03)
    return ist


# --- 2. Data Handling ---

def load_viz_data(file_path, num_frames=150):
    """
    Loads a slice of NBA tracking data from the specified JSON file format.
    Identifies active 5v5 players and formats trajectories for the simulator.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    event = data['events'][0]
    moments = event['moments'][:num_frames] if num_frames else event['moments']

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

    df = pd.json_normalize(parsed_frames, record_path='players', meta=['frame_id', ['ball', 'x'], ['ball', 'y']])
    
    for col in ['x', 'y', 'ball.x', 'ball.y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['x', 'y', 'ball.x', 'ball.y'], inplace=True)

    team_ids = df['teamid'].unique()
    team_ids = [tid for tid in team_ids if tid != -1]
    if len(team_ids) != 2:
        raise ValueError(f"Data slice does not contain exactly two teams. Found: {team_ids}")

    first_frame = df[df['frame_id'] == df['frame_id'].min()].copy()
    dist_sq = (first_frame['x'] - first_frame['ball.x']).pow(2) + (first_frame['y'] - first_frame['ball.y']).pow(2)
    first_frame['dist_to_ball'] = dist_sq.pow(0.5)

    active_players = first_frame.groupby('teamid').apply(lambda g: g.nsmallest(5, 'dist_to_ball')).reset_index(drop=True)
    active_player_ids = active_players['playerid'].unique()

    df_active = df[df['playerid'].isin(active_player_ids)].copy()

    off_team_id = active_players.loc[active_players['dist_to_ball'].idxmin()]['teamid']
    def_team_id = team_ids[0] if team_ids[1] == off_team_id else team_ids[1]

    df_active = df_active.sort_values(['frame_id', 'teamid', 'playerid'])

    frame_counts = df_active.groupby(['frame_id', 'teamid']).size().unstack(fill_value=0)
    valid_frames = frame_counts[(frame_counts[def_team_id] == 5) & (frame_counts[off_team_id] == 5)].index

    if len(valid_frames) == 0:
        raise ValueError("Could not find any valid 5v5 frames in the provided slice.")

    df_final = df_active[df_active['frame_id'].isin(valid_frames)]

    def_traj = df_final[df_final['teamid'] == def_team_id][['x', 'y']].values.reshape(-1, 5, 2)
    off_traj = df_final[df_final['teamid'] == off_team_id][['x', 'y']].values.reshape(-1, 5, 2)

    ball_traj_df = df_final[['frame_id', 'ball.x', 'ball.y']].drop_duplicates().sort_values('frame_id')
    ball_traj = ball_traj_df[['ball.x', 'ball.y']].values

    # --- Q Trajectory Placeholder ---
    # If you load your actual spatial Q maps from your parquet/npz files, 
    # replace this line with your real (frames, 5) array of Spatial Quality values!
    q_traj = jnp.ones((len(valid_frames), 5))

    return jnp.array(def_traj), jnp.array(off_traj), jnp.array(ball_traj), jnp.array(q_traj)


# --- 3. Court Drawing Helpers ---

def draw_court_matplotlib(ax=None):
    if ax is None: ax = plt.gca()
    ax.add_patch(Rectangle((0, 0), 94, 50, facecolor='none', edgecolor='black', lw=2))
    ax.plot([47, 47], [0, 50], color='black', lw=2)
    ax.add_patch(Circle((5.25, 25), radius=.75, facecolor='none', edgecolor='black', lw=2))
    ax.add_patch(Circle((94-5.25, 25), radius=.75, facecolor='none', edgecolor='black', lw=2))
    ax.add_patch(Arc((5.25, 25), 47.5, 47.5, theta1=270, theta2=90, color='black', lw=2))
    ax.add_patch(Arc((94-5.25, 25), 47.5, 47.5, theta1=90, theta2=270, color='black', lw=2))
    ax.set_xlim(0, 94); ax.set_ylim(0, 50)
    ax.set_aspect('equal', adjustable='box')
    return ax

def get_court_shapes_plotly():
    shapes = []
    shapes.append(dict(type="rect", x0=0, y0=0, x1=94, y1=50, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=47, y0=0, x1=47, y1=50, line=dict(color="black", width=2)))
    shapes.append(dict(type="circle", x0=4.5, y0=24.25, x1=6, y1=25.75, line=dict(color="black", width=2)))
    shapes.append(dict(type="circle", x0=94-6, y0=24.25, x1=94-4.5, y1=25.75, line=dict(color="black", width=2)))
    r = 23.75
    shapes.append(dict(type="path", path=f"M {r+5.25},47.5 A {r},{r} 0 0,0 {r+5.25},2.5", line=dict(color="black", width=2)))
    shapes.append(dict(type="path", path=f"M {94-(r+5.25)},47.5 A {r},{r} 0 0,1 {94-(r+5.25)},2.5", line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=5.25, y0=2.5, x1=5.25+r, y1=2.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=5.25, y0=47.5, x1=5.25+r, y1=47.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=94-5.25, y0=2.5, x1=94-(r+5.25), y1=2.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=94-5.25, y0=47.5, x1=94-(r+5.25), y1=47.5, line=dict(color="black", width=2)))
    return shapes


# --- 4. Visualization Functions ---

def plot_speed_analysis(sim_traj, real_traj=None, filename="speed_analysis.png", fps=25.0):
    def calculate_speeds(trajectory):
        delta = trajectory[1:] - trajectory[:-1]
        dist_per_frame = jnp.linalg.norm(delta, axis=-1)
        return dist_per_frame * fps

    sim_speeds = calculate_speeds(sim_traj)
    rows = 2 if real_traj is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, 6 * rows), sharex=True, sharey=True)
    if rows == 1: axes = [axes]
    time_axis = np.arange(len(sim_speeds)) / fps

    ax_sim = axes[0]
    for i in range(5):
        ax_sim.plot(time_axis, sim_speeds[:, i], label=f'Sim Defender {i}', alpha=0.7)
    ax_sim.axhline(y=15.0, color='orange', linestyle='--', label='Soft Cap (15 ft/s)')
    ax_sim.axhline(y=20.0, color='red', linestyle='--', label='Phys. Limit (20 ft/s)')
    ax_sim.set_title("Simulated Defenders Speed")
    ax_sim.set_ylabel("Speed (ft/s)")
    ax_sim.grid(True, alpha=0.3)
    ax_sim.legend(loc='upper right')

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
    xx, yy = jnp.meshgrid(jnp.linspace(0, 94, 100), jnp.linspace(0, 50, 50))
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    fixed_defenders = defenders[1:]
    
    def get_energy_at_point(point):
        current_defenders = jnp.vstack([point, fixed_defenders])
        energies = total_energy(current_defenders, offenders, ball, basket, params)
        return energies[0]

    energies = vmap(get_energy_at_point)(grid_points)
    zz = energies.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(12, 7))
    draw_court_matplotlib(ax)
    contour = ax.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.7)
    fig.colorbar(contour, ax=ax, orientation='vertical', label='Potential Energy')
    ax.scatter(defenders[:, 0], defenders[:, 1], c='blue', s=100, label='Defenders')
    ax.scatter(offenders[:, 0], offenders[:, 1], c='red', s=100, label='Offenders')
    ax.scatter(ball[0], ball[1], c='orange', s=120, marker='*', label='Ball')
    ax.set_title("Defensive Potential Surface (for Defender 0)")
    ax.legend()
    plt.savefig(filename)
    print(f"Saved potential surface plot to {filename}")


def save_simulation_gif(sim_traj, real_traj, ball_traj, offenders_traj, q_traj, filename='sim.gif'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Defensive Simulation vs. Reality (with IST Threat)', fontsize=16)

    # Precompute IST matrices
    real_ist = calculate_ist_tensor(offenders_traj, real_traj, q_traj)
    sim_ist = calculate_ist_tensor(offenders_traj, sim_traj, q_traj)

    # Plot 1: Real Defense
    draw_court_matplotlib(ax1)
    ax1.set_title('Real Defense')
    real_def_sc = ax1.scatter([], [], c='blue', s=120, label='Defenders')
    real_off_sc = ax1.scatter([], [], c='red', s=120, label='Offenders')
    real_ball_sc = ax1.scatter([], [], c='orange', marker='o', s=80, label='Ball')
    ax1.legend(loc='upper right')
    real_texts = [ax1.text(0, 0, '', color='darkred', fontsize=10, fontweight='bold', ha='center', va='bottom') for _ in range(5)]

    # Plot 2: Simulated Defense
    draw_court_matplotlib(ax2)
    ax2.set_title('Simulated (JKO)')
    sim_def_sc = ax2.scatter([], [], c='blue', alpha=0.7, s=120, label='Simulated Defenders')
    sim_off_sc = ax2.scatter([], [], c='red', s=120, label='Offenders')
    sim_ball_sc = ax2.scatter([], [], c='orange', marker='o', s=80, label='Ball')
    ax2.legend(loc='upper right')
    sim_texts = [ax2.text(0, 0, '', color='darkred', fontsize=10, fontweight='bold', ha='center', va='bottom') for _ in range(5)]

    def update(frame):
        # Update Real Plot
        real_def_sc.set_offsets(real_traj[frame])
        real_off_sc.set_offsets(offenders_traj[frame])
        real_ball_sc.set_offsets(ball_traj[frame])
        for i in range(5):
            real_texts[i].set_position((offenders_traj[frame, i, 0], offenders_traj[frame, i, 1] + 1.5))
            real_texts[i].set_text(f'{real_ist[frame, i]:.1f}')
        
        # Update Sim Plot
        sim_def_sc.set_offsets(sim_traj[frame])
        sim_off_sc.set_offsets(offenders_traj[frame])
        sim_ball_sc.set_offsets(ball_traj[frame])
        for i in range(5):
            sim_texts[i].set_position((offenders_traj[frame, i, 0], offenders_traj[frame, i, 1] + 1.5))
            sim_texts[i].set_text(f'{sim_ist[frame, i]:.1f}')
        
        return [real_def_sc, real_off_sc, real_ball_sc, sim_def_sc, sim_off_sc, sim_ball_sc] + real_texts + sim_texts

    anim = FuncAnimation(fig, update, frames=len(sim_traj), interval=40, blit=True)
    anim.save(filename, writer='pillow')
    print(f"Saved simulation GIF to {filename}")


def create_interactive_plot(sim_traj, real_traj, ball_traj, offenders_traj, q_traj):
    from plotly.subplots import make_subplots
    pio.renderers.default = "browser"

    # Precompute IST
    real_ist = calculate_ist_tensor(offenders_traj, real_traj, q_traj)
    sim_ist = calculate_ist_tensor(offenders_traj, sim_traj, q_traj)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Real Defense', 'Simulated (JKO)'))

    def get_text_labels(ist_array):
        return [f"<b>{val:.1f}</b>" for val in ist_array]

    # Initial Traces
    fig.add_trace(go.Scatter(x=real_traj[0, :, 0], y=real_traj[0, :, 1], mode='markers', marker=dict(color='blue', size=15), name='Real Defenders'), row=1, col=1)
    fig.add_trace(go.Scatter(x=offenders_traj[0, :, 0], y=offenders_traj[0, :, 1], mode='markers+text', text=get_text_labels(real_ist[0]), textposition="top center", marker=dict(color='red', size=15), name='Offenders'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[ball_traj[0, 0]], y=[ball_traj[0, 1]], mode='markers', marker=dict(color='orange', size=12), name='Ball'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=sim_traj[0, :, 0], y=sim_traj[0, :, 1], mode='markers', marker=dict(color='blue', size=15, opacity=0.6), name='Sim Defenders'), row=1, col=2)
    fig.add_trace(go.Scatter(x=offenders_traj[0, :, 0], y=offenders_traj[0, :, 1], mode='markers+text', text=get_text_labels(sim_ist[0]), textposition="top center", marker=dict(color='red', size=15), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[ball_traj[0, 0]], y=[ball_traj[0, 1]], mode='markers', marker=dict(color='orange', size=12), showlegend=False), row=1, col=2)

    # Create frames
    frames = [go.Frame(data=[
        go.Scatter(x=real_traj[k, :, 0], y=real_traj[k, :, 1]),
        go.Scatter(x=offenders_traj[k, :, 0], y=offenders_traj[k, :, 1], text=get_text_labels(real_ist[k])),
        go.Scatter(x=[ball_traj[k, 0]], y=[ball_traj[k, 1]]),
        
        go.Scatter(x=sim_traj[k, :, 0], y=sim_traj[k, :, 1]),
        go.Scatter(x=offenders_traj[k, :, 0], y=offenders_traj[k, :, 1], text=get_text_labels(sim_ist[k])),
        go.Scatter(x=[ball_traj[k, 0]], y=[ball_traj[k, 1]]),
    ], name=str(k)) for k in range(len(sim_traj))]
    
    fig.frames = frames

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}])]
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))]) for f in fig.frames],
            transition=dict(duration=0), x=0, xanchor="left", len=1,
        )]
    )

    court_shapes = get_court_shapes_plotly()
    fig.update_layout(title_text="Defensive Simulation vs. Reality (with IST Threat)", shapes=court_shapes)
    fig.update_xaxes(range=[0, 94], autorange=False)
    fig.update_yaxes(range=[0, 50], autorange=False, scaleanchor="x1", scaleratio=1)
    
    return fig


# --- 5. Main Execution ---

if __name__ == '__main__':
    # Make sure this points to your real tracking JSON!
    DATA_FILE = "0021500492.json"
    STUDY_NAME = "nba-defensive-optimization"
    STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

    solver_params = {
        **default_params,
        'jko_lambda': 0.5,
        'sinkhorn_epsilon': 0.01,
        'learning_rate': 0.1, 
        'velocity_cap': 0.8,
        'court_dims': [[0., 94.], [0., 50.]],
        'max_gradient_norm': 1.0,
        'acceleration_penalty_weight': 2.0,
        'velocity_penalty_weight': 1.0,
    }

    print(f"\nLoading best parameters from Optuna study '{STUDY_NAME}'...")
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        print(f"Using parameters from Trial #{best_trial.number} with values: {best_trial.values}")
        solver_params.update(best_trial.params)
    except (FileNotFoundError, ValueError, IndexError, ImportError):
        print("Could not load study or find best trial. Using default parameters.")
        
    print(f"Loading data from {DATA_FILE}...")
    
    # -------------------------------------------------------------
    # load_viz_data NOW RETURNS q_traj!
    # -------------------------------------------------------------
    real_def_traj, off_traj, ball_traj, q_traj = load_viz_data(DATA_FILE, num_frames=None)

    init_defenders = real_def_traj[0]
    print(f"\nRunning simulation for {len(ball_traj)} timesteps...")
    
    sim_def_traj = run_simulation(
        init_defenders,
        ball_traj,
        off_traj,
        basket_pos=jnp.array([5.25, 25.0]),
        params=solver_params,
        jko_num_steps=20
    )

    print("\n1. Generating GIF...")
    save_simulation_gif(
        sim_def_traj,
        real_def_traj,
        ball_traj,
        off_traj,
        q_traj,       # <--- Passed dynamically here
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
        q_traj        # <--- Passed dynamically here
    )
    interactive_fig.write_html("interactive_simulation.html")
    print("Saved interactive plot to interactive_simulation.html. Open this file in a browser.")