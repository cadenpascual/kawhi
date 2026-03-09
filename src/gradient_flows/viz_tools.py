import json
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import pandas as pd
import os
from PIL import Image
import io

# Visualization
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Import from our project modules
from .potentials import total_energy, _total_energy_per_defender, params as default_params
from .solver import run_simulation, params as solver_params


# --- Data Handling ---
def load_viz_data(file_path, num_frames=150):
    """
    Loads a slice of NBA tracking data from the specified JSON file format.
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

    return jnp.array(def_traj), jnp.array(off_traj), jnp.array(ball_traj)


# --- Court Drawing Helpers ---

def draw_plotly_court(fig, xref='x', yref='y', half_court=None):
    """Adds high-quality NBA court lines to a Plotly figure."""
    
    def ellipse_arc(x_center, y_center, a, b, start_angle, end_angle, N=100):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]},{y[0]}'
        for k in range(1, len(t)):
            path += f' L {x[k]},{y[k]}'
        return path

    line_col = "#777777"
    three_r = 23.75
    
    # Base shapes (full court perimeter)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=94, y1=50, line=dict(color=line_col, width=2), layer='below'),
    ]

    # Midcourt Line
    shapes.append(dict(type="line", x0=47, y0=0, x1=47, y1=50, line=dict(color=line_col, width=2), layer='below'))

    # Midcourt Circle
    if half_court is None:
        shapes.append(dict(type="circle", x0=41, y0=19, x1=53, y1=31, line=dict(color=line_col, width=2), layer='below'))
    elif half_court == 'left':
        # Left half of center circle (from 90 to 270 degrees)
        shapes.append(dict(type="path", path=ellipse_arc(47, 25, 6, 6, np.pi/2, 3*np.pi/2), line=dict(color=line_col, width=2), layer='below'))
    elif half_court == 'right':
        # Right half of center circle (from -90 to 90 degrees)
        shapes.append(dict(type="path", path=ellipse_arc(47, 25, 6, 6, -np.pi/2, np.pi/2), line=dict(color=line_col, width=2), layer='below'))

    # Left Side Features
    if half_court is None or half_court == 'left':
        shapes += [
            dict(type="rect", x0=0, y0=17, x1=19, y1=33, line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=0, y0=19, x1=19, y1=31, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=0, y0=3, x1=14, y1=3, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=0, y0=47, x1=14, y1=47, line=dict(color=line_col, width=1), layer='below'),
            dict(type="path", path=ellipse_arc(5.25, 25, three_r, three_r, -1.18, 1.18), line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=4, y0=22, x1=4.2, y1=28, line=dict(color="#ec7607", width=2), fillcolor='#ec7607'),
            dict(type="circle", x0=4.5, y0=24.25, x1=6, y1=25.75, line=dict(color="#ec7607", width=2)),
        ]

    # Right Side Features
    if half_court is None or half_court == 'right':
        shapes += [
            dict(type="rect", x0=75, y0=17, x1=94, y1=33, line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=75, y0=19, x1=94, y1=31, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=80, y0=3, x1=94, y1=3, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=80, y0=47, x1=94, y1=47, line=dict(color=line_col, width=1), layer='below'),
            dict(type="path", path=ellipse_arc(94-5.25, 25, three_r, three_r, np.pi - 1.18, np.pi + 1.18), line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=94-4.2, y0=22, x1=94-4, y1=28, line=dict(color="#ec7607", width=2), fillcolor='#ec7607'),
            dict(type="circle", x0=94-6, y0=24.25, x1=94-4.5, y1=25.75, line=dict(color="#ec7607", width=2)),
        ]
    
    for s in shapes:
        s['xref'] = xref
        s['yref'] = yref
        fig.add_shape(s)

# --- Visualization Functions ---

def create_interactive_plot(sim_traj, real_traj, ball_traj, offenders_traj, filename, half_court='left'):
    """Creates an interactive Plotly figure comparing real vs. simulated defense."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Real Defense', 'Simulated (JKO)'))

    def add_traces(f, row, col, traj, name_prefix, showlegend=True):
        fig.add_trace(go.Scatter(x=traj[f, :, 0], y=traj[f, :, 1], mode='markers', 
                                 marker=dict(color='blue', size=12, line=dict(width=1, color='white')), name=f'{name_prefix} Defenders', showlegend=showlegend), row=row, col=col)
        fig.add_trace(go.Scatter(x=offenders_traj[f, :, 0], y=offenders_traj[f, :, 1], mode='markers', 
                                 marker=dict(color='red', size=12, line=dict(width=1, color='white')), name='Offenders', showlegend=showlegend and row==1 and col==1), row=row, col=col)
        fig.add_trace(go.Scatter(x=[ball_traj[f, 0]], y=[ball_traj[f, 1]], mode='markers', 
                                 marker=dict(color='orange', size=10, symbol='star'), name='Ball', showlegend=showlegend and row==1 and col==1), row=row, col=col)

    add_traces(0, 1, 1, real_traj, 'Real')
    add_traces(0, 1, 2, sim_traj, 'Sim', showlegend=False)

    frames = []
    for k in range(len(sim_traj)):
        frames.append(go.Frame(data=[
            go.Scatter(x=real_traj[k, :, 0], y=real_traj[k, :, 1]),
            go.Scatter(x=offenders_traj[k, :, 0], y=offenders_traj[k, :, 1]),
            go.Scatter(x=[ball_traj[k, 0]], y=[ball_traj[k, 1]]),
            go.Scatter(x=sim_traj[k, :, 0], y=sim_traj[k, :, 1]),
            go.Scatter(x=offenders_traj[k, :, 0], y=offenders_traj[k, :, 1]),
            go.Scatter(x=[ball_traj[k, 0]], y=[ball_traj[k, 1]])
        ], name=str(k)))
    
    fig.frames = frames
    
    draw_plotly_court(fig, xref='x', yref='y', half_court=half_court)
    draw_plotly_court(fig, xref='x2', yref='y2', half_court=half_court)

    x_range = [0, 47] if half_court == 'left' else ([47, 94] if half_court == 'right' else [0, 94])

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                dict(label="Stop", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )],
        sliders=[dict(
            steps=[dict(method='animate', label=str(k), args=[[str(k)], dict(mode='immediate', frame=dict(duration=40, redraw=True), transition=dict(duration=0))]) for k in range(len(sim_traj))],
            transition=dict(duration=0),
            x=0, y=0, currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True, xanchor="right")
        )],
        height=600,
        margin=dict(l=20, r=20, t=60, b=80),
        plot_bgcolor='white'
    )
    # Apply aspect ratio to BOTH subplots
    fig.update_xaxes(range=x_range, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 50], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
    # Target second axis explicitly for safety in some plotly versions
    fig.update_layout(yaxis2=dict(scaleanchor="x2", scaleratio=1))
    
    fig.write_html(filename, include_plotlyjs='cdn', full_html=False)

def create_gradient_flow_plotly(sim_traj, ball_traj, offenders_traj, basket_pos, params, filename, half_court='left'):
    """Interactive Plotly version of the gradient flow animation."""
    x_min, x_max = (0, 47) if half_court == 'left' else ((47, 94) if half_court == 'right' else (0, 94))
    nx, ny = 40, 20
    xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, nx), jnp.linspace(0, 50, ny))
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    def get_z(t):
        def energy_at_point(p):
            full_defs = jnp.vstack([p, sim_traj[t, 1:]])
            return _total_energy_per_defender(p, full_defs, offenders_traj[t], ball_traj[t], basket_pos, params)
        return vmap(energy_at_point)(grid_points).reshape(ny, nx)

    initial_z = get_z(0)
    abs_max = float(jnp.max(jnp.abs(initial_z))) * 1.5

    fig = go.Figure(data=[
        go.Heatmap(x=np.linspace(x_min, x_max, nx), y=np.linspace(0, 50, ny), z=initial_z, 
                   colorscale='RdBu', reversescale=True, zmid=0, zmin=-abs_max, zmax=abs_max,
                   opacity=0.7, showscale=True, name='Potential Energy'),
        go.Scatter(x=offenders_traj[0, :, 0], y=offenders_traj[0, :, 1], mode='markers', marker=dict(color='red', size=12, line=dict(width=1, color='white')), name='Offenders'),
        go.Scatter(x=[ball_traj[0, 0]], y=[ball_traj[0, 1]], mode='markers', marker=dict(color='orange', size=14, symbol='star', line=dict(width=1, color='black')), name='Ball'),
        go.Scatter(x=sim_traj[0, :, 0], y=sim_traj[0, :, 1], mode='markers', marker=dict(color='blue', size=12, line=dict(width=2, color='white')), name='Defenders')
    ])

    frames = []
    step = 2
    for k in range(0, len(sim_traj), step):
        zk = get_z(k)
        frames.append(go.Frame(data=[
            go.Heatmap(z=zk),
            go.Scatter(x=offenders_traj[k, :, 0], y=offenders_traj[k, :, 1]),
            go.Scatter(x=[ball_traj[k, 0]], y=[ball_traj[k, 1]]),
            go.Scatter(x=sim_traj[k, :, 0], y=sim_traj[k, :, 1])
        ], name=str(k)))

    fig.frames = frames
    draw_plotly_court(fig, xref='x', yref='y', half_court=half_court)

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                dict(label="Stop", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )],
        sliders=[dict(
            steps=[dict(method='animate', label=str(k), args=[[str(k)], dict(mode='immediate', frame=dict(duration=40, redraw=True), transition=dict(duration=0))]) for k in range(0, len(sim_traj), step)],
            transition=dict(duration=0),
            x=0, y=0, currentvalue=dict(font=dict(size=12), prefix="Frame: ", visible=True, xanchor="right")
        )],
        title="JKO Gradient Flow Surface (Interactive)",
        height=600,
        margin=dict(l=20, r=20, t=60, b=80),
        plot_bgcolor='white'
    )
    fig.update_xaxes(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 50], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
    
    fig.write_html(filename, include_plotlyjs='cdn', full_html=False)

def plot_potential_surface(defenders, offenders, ball, basket, params, filename, half_court='left'):
    """Computes and plots the total_energy surface for a single defender using Plotly."""
    x_min, x_max = (0, 47) if half_court == 'left' else ((47, 94) if half_court == 'right' else (0, 94))
    nx, ny = 100, 50
    xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, nx), jnp.linspace(0, 50, ny))
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    fixed_defenders = defenders[1:]
    
    def get_energy_at_point(point):
        current_defenders = jnp.vstack([point, fixed_defenders])
        energies = total_energy(current_defenders, offenders, ball, basket, params)
        return energies[0]

    energies = vmap(get_energy_at_point)(grid_points)
    zz = energies.reshape(ny, nx)
    abs_max = float(jnp.max(jnp.abs(zz)))

    fig = go.Figure(data=[
        go.Heatmap(x=np.linspace(x_min, x_max, nx), y=np.linspace(0, 50, ny), z=zz, 
                   colorscale='RdBu', reversescale=True, zmid=0, zmin=-abs_max, zmax=abs_max,
                   opacity=0.7, showscale=True, name='Potential Energy'),
        go.Scatter(x=defenders[:, 0], y=defenders[:, 1], mode='markers', marker=dict(color='blue', size=12, line=dict(width=2, color='white')), name='Defenders'),
        go.Scatter(x=offenders[:, 0], y=offenders[:, 1], mode='markers', marker=dict(color='red', size=12, line=dict(width=1, color='white')), name='Offenders'),
        go.Scatter(x=[ball[0]], y=[ball[1]], mode='markers', marker=dict(color='orange', size=14, symbol='star', line=dict(width=1, color='black')), name='Ball')
    ])

    draw_plotly_court(fig, xref='x', yref='y', half_court=half_court)

    fig.update_layout(
        title="Defensive Potential Surface",
        height=600,
        margin=dict(l=20, r=20, t=60, b=80),
        plot_bgcolor='white'
    )
    fig.update_xaxes(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 50], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
    
    if filename.endswith(".html"):
        fig.write_html(filename, include_plotlyjs='cdn', full_html=False)
    else:
        fig.write_image(filename)

def save_simulation_gif(sim_traj, real_traj, ball_traj, offenders_traj, filename, half_court='left'):
    """Saves side-by-side simulation GIF using Plotly for high-quality frames."""
    print(f"Generating {filename}...")
    frames = []
    x_min, x_max = (0, 47) if half_court == 'left' else ((47, 94) if half_court == 'right' else (0, 94))
    
    # Subsample for speed (every 2nd frame)
    for f in range(0, len(sim_traj), 2):
        if f % 20 == 0: print(f"  Frame {f}/{len(sim_traj)}...")
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Real', 'Simulated'))
        
        # Real Side
        fig.add_trace(go.Scatter(x=real_traj[f, :, 0], y=real_traj[f, :, 1], mode='markers', marker=dict(color='blue', size=10, line=dict(width=1, color='white'))), row=1, col=1)
        fig.add_trace(go.Scatter(x=offenders_traj[f, :, 0], y=offenders_traj[f, :, 1], mode='markers', marker=dict(color='red', size=10, line=dict(width=1, color='white'))), row=1, col=1)
        fig.add_trace(go.Scatter(x=[ball_traj[f, 0]], y=[ball_traj[f, 1]], mode='markers', marker=dict(color='orange', size=12, symbol='star')), row=1, col=1)
        
        # Simulated Side
        fig.add_trace(go.Scatter(x=sim_traj[f, :, 0], y=sim_traj[f, :, 1], mode='markers', marker=dict(color='blue', size=10, line=dict(width=1, color='white'))), row=1, col=2)
        fig.add_trace(go.Scatter(x=offenders_traj[f, :, 0], y=offenders_traj[f, :, 1], mode='markers', marker=dict(color='red', size=10, line=dict(width=1, color='white'))), row=1, col=2)
        fig.add_trace(go.Scatter(x=[ball_traj[f, 0]], y=[ball_traj[f, 1]], mode='markers', marker=dict(color='orange', size=12, symbol='star')), row=1, col=2)
        
        draw_plotly_court(fig, xref='x', yref='y', half_court=half_court)
        draw_plotly_court(fig, xref='x2', yref='y2', half_court=half_court)
        
        fig.update_layout(width=1000, height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='white')
        fig.update_xaxes(range=[x_min, x_max], visible=False)
        fig.update_yaxes(range=[0, 50], visible=False, scaleanchor="x", scaleratio=1)
        fig.update_layout(yaxis2=dict(scaleanchor="x2", scaleratio=1))
        
        img_bytes = fig.to_image(format="png")
        frames.append(Image.open(io.BytesIO(img_bytes)))
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f"Saved {filename}")

def save_gradient_flow_gif(sim_traj, ball_traj, offenders_traj, basket_pos, params, filename, half_court='left'):
    """Saves gradient flow GIF using Plotly for high-quality frames."""
    print(f"Generating {filename}...")
    frames = []
    x_min, x_max = (0, 47) if half_court == 'left' else ((47, 94) if half_court == 'right' else (0, 94))
    nx, ny = 40, 20
    xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, nx), jnp.linspace(0, 50, ny))
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    def get_energy(t):
        def energy_at_point(p):
            return _total_energy_per_defender(p, jnp.vstack([p, sim_traj[t, 1:]]), offenders_traj[t], ball_traj[t], basket_pos, params)
        return vmap(energy_at_point)(grid_points).reshape(ny, nx)
    
    abs_max = float(jnp.max(jnp.abs(get_energy(0)))) * 1.2

    for f in range(0, len(sim_traj), 4):
        if f % 20 == 0: print(f"  Frame {f}/{len(sim_traj)}...")
        fig = go.Figure()
        zz = get_energy(f)
        fig.add_trace(go.Heatmap(x=np.linspace(x_min, x_max, nx), y=np.linspace(0, 50, ny), z=zz, 
                                 colorscale='RdBu', reversescale=True, zmid=0, zmin=-abs_max, zmax=abs_max, opacity=0.7, showscale=False))
        fig.add_trace(go.Scatter(x=offenders_traj[f, :, 0], y=offenders_traj[f, :, 1], mode='markers', marker=dict(color='red', size=10, line=dict(width=1, color='white'))))
        fig.add_trace(go.Scatter(x=[ball_traj[f, 0]], y=[ball_traj[f, 1]], mode='markers', marker=dict(color='orange', size=12, symbol='star')))
        fig.add_trace(go.Scatter(x=sim_traj[f, :, 0], y=sim_traj[f, :, 1], mode='markers', marker=dict(color='blue', size=10, line=dict(width=1, color='white'))))
        
        draw_plotly_court(fig, half_court=half_court)
        fig.update_layout(width=600, height=400, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white')
        fig.update_xaxes(range=[x_min, x_max], visible=False); fig.update_yaxes(range=[0, 50], visible=False, scaleanchor="x", scaleratio=1)
        
        img_bytes = fig.to_image(format="png")
        frames.append(Image.open(io.BytesIO(img_bytes)))
    
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=160, loop=0)
    print(f"Saved {filename}")

def plot_speed_analysis(sim_traj, real_traj, filename):
    def get_speeds(traj):
        return jnp.linalg.norm(traj[1:] - traj[:-1], axis=-1) * 25.0
    s_speeds = get_speeds(sim_traj); r_speeds = get_speeds(real_traj)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for i in range(5):
        axes[0].plot(s_speeds[:, i], alpha=0.7); axes[1].plot(r_speeds[:, i], alpha=0.7)
    axes[0].set_title("Simulated Speeds (ft/s)"); axes[1].set_title("Real Speeds (ft/s)")
    axes[0].axhline(y=15, color='r', linestyle='--'); axes[1].axhline(y=15, color='r', linestyle='--')
    plt.savefig(filename); plt.close()

if __name__ == '__main__':
    os.makedirs('assets', exist_ok=True)
    os.makedirs('_includes/visualizations', exist_ok=True)
    
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
        'velocity_penalty_weight': 0.5,
    }

    try:
        import optuna
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
        best_trial = min(study.best_trials, key=lambda t: t.values[0])
        solver_params.update(best_trial.params)
    except:
        pass

    print("Loading data...")
    real_def_traj, off_traj, ball_traj = load_viz_data(DATA_FILE)
    
    print("Running simulation...")
    sim_def_traj = run_simulation(real_def_traj[0], ball_traj, off_traj, jnp.array([5.25, 25.0]), solver_params, jko_num_steps=15)

    print("Saving assets/nba_simulation_viz.gif (25 FPS, side-by-side)...")
    save_simulation_gif(sim_def_traj, real_def_traj, ball_traj, off_traj, 'assets/nba_simulation_viz.gif', half_court='left')
    
    print("Saving assets/gradient_flow.gif (25 FPS)...")
    save_gradient_flow_gif(sim_def_traj, ball_traj, off_traj, jnp.array([5.25, 25.0]), solver_params, 'assets/gradient_flow.gif', half_court='left')

    print("Saving _includes/visualizations/defense_rotation_plot.html...")
    create_interactive_plot(sim_def_traj, real_def_traj, ball_traj, off_traj, '_includes/visualizations/defense_rotation_plot.html', half_court='left')

    print("Saving _includes/visualizations/gradient_flow_interactive.html...")
    create_gradient_flow_plotly(sim_def_traj, ball_traj, off_traj, jnp.array([5.25, 25.0]), solver_params, '_includes/visualizations/gradient_flow_interactive.html', half_court='left')

    print("Saving assets/speed_analysis.png...")
    plot_speed_analysis(sim_def_traj, real_def_traj, 'assets/speed_analysis.png')

    print("Saving assets/potential_surface.png...")
    plot_potential_surface(sim_def_traj[0], off_traj[0], ball_traj[0], jnp.array([5.25, 25.0]), solver_params, 'assets/potential_surface.png', half_court='left')
    
    print("Done.")
