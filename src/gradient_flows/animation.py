import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import vmap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch
import seaborn as sns
import base64
from PIL import Image, ImageOps, ImageDraw
import io
import os

from .potentials import total_energy, _calculate_ist_penalty
from src.gradient_flows.court import draw_court_matplotlib, draw_plotly_court

import numpy as np

import numpy as np

def extract_plot_trajectories(row):
    """
    Takes a single row from the tracking dataframe and stacks the individual 
    player columns into 3D numpy arrays for Plotly animations.
    Applies the spatial offsets (+5.25, +25.0) and swaps X/Y for court alignment.
    """
    # 1. Extract Ball -> Shape: (num_frames, 2)
    # Note the order: Y first (with +5.25), then X (with +25.0)
    ball_traj = np.column_stack((
        np.array(row['ball_y_traj']) + 5.25, 
        np.array(row['ball_x_traj']) + 25.0
    ))
    
    # 2. Extract Offense -> Shape: (num_frames, 5, 2)
    off_list = []
    for i in range(1, 6):
        off_player = np.column_stack((
            np.array(row[f'off{i}_y_traj']) + 5.25, 
            np.array(row[f'off{i}_x_traj']) + 25.0
        ))
        off_list.append(off_player)
    off_traj = np.stack(off_list, axis=1)
    
    # 3. Extract Real Defense -> Shape: (num_frames, 5, 2)
    def_list = []
    for i in range(1, 6):
        def_player = np.column_stack((
            np.array(row[f'def{i}_y_traj']) + 5.25, 
            np.array(row[f'def{i}_x_traj']) + 25.0
        ))
        def_list.append(def_player)
    def_traj = np.stack(def_list, axis=1)
    
    return off_traj, def_traj, ball_traj


# --- Animation Functions --- #
def animate_standard_play(off_traj, def_traj, ball_traj):
    """
    Animates a single standard play showing offense, defense, and the ball.
    """
    num_frames = off_traj.shape[0]

    fig = go.Figure()

    # --- Initial Frame Setup ---
    # Trace 0: Defense
    fig.add_trace(go.Scatter(
        x=def_traj[0,:,0], y=def_traj[0,:,1], mode='markers', 
        marker=dict(color='blue', size=14), name='Defense'
    ))
    
    # Trace 1: Offense
    fig.add_trace(go.Scatter(
        x=off_traj[0,:,0], y=off_traj[0,:,1], mode='markers', 
        marker=dict(color='red', size=14), name='Offense'
    ))
    
    # Trace 2: Ball
    fig.add_trace(go.Scatter(
        x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', 
        marker=dict(color='orange', size=10, line=dict(width=2, color='black')), name='Ball'
    ))

    # --- Build Animation Frames ---
    frames = []
    for f in range(num_frames):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=def_traj[f,:,0], y=def_traj[f,:,1]),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]),
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]])
            ],
            name=str(f),
            traces=[0, 1, 2] # Explicitly tell Plotly which traces to update
        ))
    fig.frames = frames

    # --- Draw the Court ---
    # Assuming draw_plotly_court is defined elsewhere in your utils
    shapes = draw_plotly_court(xref="x", yref="y")
    
    # --- Layout & Controls ---
    fig.update_layout(
        title="Standard Play Animation",
        shapes=shapes,
        
        # --- HIDE AXIS NUMBERS AND LINES ---
        xaxis=dict(range=[0, 94], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 50], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, visible=False),
        
        template="plotly_white",
        
        # --- MAXIMIZE COURT SIZE ---
        width=900,   
        height=450,   
        margin=dict(l=0, r=0, t=50, b=80), # 0 margin on left/right to stretch the court
        
        # --- Playback Controls ---
        updatemenus=[dict(
            type="buttons", 
            showactive=False,
            x=0.0,
            y=-0.05,  # Tucked slightly closer to the court now that axes are gone
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": False}}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0,
            x=0.1,
            y=-0.05,  # Tucked slightly closer
            xanchor="left",
            yanchor="top",
            len=0.9,
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=False))]) 
                   for k in range(num_frames)]
        )]
    )
    
    return fig.show()

def animate_comparison_play(off_traj, real_def_traj, sim_def_traj, ball_traj, ist_real, ist_sim):
    """
    Takes pre-calculated IST arrays and generates a side-by-side interactive comparison.
    """
    num_frames = off_traj.shape[0]

    # Create 1x2 Subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Real Defense", "Simulated (JKO) Defense"),
                        shared_xaxes=False, shared_yaxes=False)

    def get_labels(vals):
        return [f"<b>{v:.2f}</b>" for v in vals]

    # --- Initial Frame Setup ---
    # Col 1: Real Defense (Traces 0, 1, 2)
    fig.add_trace(go.Scatter(x=real_def_traj[0,:,0], y=real_def_traj[0,:,1], mode='markers', 
                             marker=dict(color='blue', size=14), name='Real Def'), row=1, col=1)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode='markers+text', 
                             text=get_labels(ist_real[0]), textposition="top center", 
                             marker=dict(color='red', size=14), name='Offense'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', 
                             marker=dict(color='orange', size=10, line=dict(width=2, color='black')), name='Ball'), row=1, col=1)

    # Col 2: Sim Defense (Traces 3, 4, 5)
    fig.add_trace(go.Scatter(x=sim_def_traj[0,:,0], y=sim_def_traj[0,:,1], mode='markers', 
                             marker=dict(color='blue', size=14, opacity=0.7), name='Sim Def'), row=1, col=2)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode='markers+text', 
                             text=get_labels(ist_sim[0]), textposition="top center", 
                             marker=dict(color='red', size=14), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', 
                             marker=dict(color='orange', size=10, line=dict(width=2, color='black')), showlegend=False), row=1, col=2)

    # --- Build Animation Frames ---
    frames = []
    for f in range(num_frames):
        frames.append(go.Frame(
            data=[
                # Real (Match trace order above)
                go.Scatter(x=real_def_traj[f,:,0], y=real_def_traj[f,:,1]),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1], text=get_labels(ist_real[f])),
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]),
                # Sim
                go.Scatter(x=sim_def_traj[f,:,0], y=sim_def_traj[f,:,1]),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1], text=get_labels(ist_sim[f])),
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]])
            ],
            name=str(f),
            traces=[0, 1, 2, 3, 4, 5]
        ))
    fig.frames = frames

    # --- Draw the Courts ---
    # Apply to x/y for the left plot, and x2/y2 for the right plot
    shapes_left = draw_plotly_court(xref="x", yref="y")
    shapes_right = draw_plotly_court(xref="x2", yref="y2")
    
    # --- Layout & Controls ---
    # --- Layout & Controls ---
    fig.update_layout(
        title="Interactive IST Audit: Real vs Simulated Pressure",
        shapes=shapes_left + shapes_right,
        xaxis=dict(range=[0, 94], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 50], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False),
        xaxis2=dict(range=[0, 94], showgrid=False, zeroline=False),
        yaxis2=dict(range=[0, 50], scaleanchor="x2", scaleratio=1, showgrid=False, zeroline=False),
        template="plotly_white",
        
        # --- THE FIX: Aspect Ratio & Margins ---
        width=1400,  # Stretch the width to fill modern monitors
        height=450,  # Shrink the height to remove dead vertical space
        margin=dict(l=20, r=20, t=60, b=20), # Remove Plotly's default padding
        # ---------------------------------------
        
        # --- FIXED CONTROLS POSITIONING ---
        updatemenus=[dict(
            type="buttons", 
            showactive=False,
            x=0.0,           # Start at the far left
            y=-0.15,         # Move down below the court
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": False}}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0,
            x=0.1,           # Start the slider to the right of the buttons (10% over)
            y=-0.15,         # Align horizontally with the buttons
            xanchor="left",
            yanchor="top",
            len=0.9,         # Make the slider take up the remaining 90% of the width
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=False))]) 
                   for k in range(num_frames)]
        )]
    )
    
    return fig

def get_base64_image(pid, folder="../images", border_color="#1D428A"):
    path = os.path.join(folder, f"{pid}.png")
    try:
        img = Image.open(path).convert("RGBA")
        
        # 1. FLOOD FILL: Only remove black touching the corners
        # This prevents beards and hair from turning white/transparent
        bg = Image.new("RGBA", img.size, (0, 0, 0, 0))
        # Create a mask where 'black' pixels at the edges are selected
        # We use a tolerance of 40 to catch "mostly black" JPG artifacts
        seed = (0, 0) # Start at top-left corner
        ImageDraw.floodfill(img, seed, (255, 255, 255, 0), thresh=40)

        # 2. CROP & SQUARE
        bbox = img.getbbox()
        if bbox: img = img.crop(bbox)
        size = max(img.size)
        square = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        square.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        
        # 3. CIRCLE MASK
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        output = ImageOps.fit(square, mask.size, centering=(0.5, 0.5))
        output.putalpha(mask)

        # 4. CRISP NAVY BORDER
        border_draw = ImageDraw.Draw(output)
        width = int(size * 0.05)
        border_draw.ellipse((0, 0, size, size), outline=border_color, width=width)

        # 5. DOWNSCALE & ENCODE
        output.thumbnail((80, 80), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        output.save(buf, format="PNG", optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Error on {pid}: {e}")
        return ""

def animate_triple_comparison(off_traj, real_def_traj, base_def_traj, sim_def_traj, 
                              ball_traj, ist_real, ist_base, ist_sim, half_court='left'):
    """
    Generates a clean 1x3 side-by-side interactive comparison without images.
    Columns: [Real] | [Baseline (Kinematic)] | [Optimized (JKO Threat)]
    """
    # --- 1. SHAPE SAFETY ---
    off_traj = np.array(off_traj).reshape(-1, 5, 2)
    real_def_traj = np.array(real_def_traj).reshape(-1, 5, 2)
    base_def_traj = np.array(base_def_traj).reshape(-1, 5, 2)
    sim_def_traj = np.array(sim_def_traj).reshape(-1, 5, 2)
    ball_traj = np.array(ball_traj).reshape(-1, 2)
    ist_real = np.array(ist_real).reshape(-1, 5)
    ist_base = np.array(ist_base).reshape(-1, 5)
    ist_sim = np.array(ist_sim).reshape(-1, 5)
    
    
    num_frames = off_traj.shape[0]
    
    # Colors (Base is Purple, Sim is Blue)
    C_OFF, C_REAL, C_BASE, C_SIM, C_BALL = '#C8102E', '#888888', '#800080', '#1D428A', '#ec7607'
    
    x_range = [-2, 49] if half_court == 'left' else [45, 96]
    y_range = [-2, 54]

    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Real NBA Defense", "Baseline Model", "Your Optimized Model"),
        horizontal_spacing=0.01
    )

    def get_dynamic_labels(vals):
        labels = []
        
        # CSS trick to create a white outline (halo) around the text
        halo = "text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;"
        
        # 1. Safely flatten the array, handling both clean arrays and "ragged" nested arrays
        try:
            vals_flat = np.array(vals, dtype=float).flatten()
        except (ValueError, TypeError):
            vals_flat = np.hstack(vals).astype(float)

        # 2. Loop through the guaranteed flat floats
        for v in vals_flat:
            color_hex = C_OFF if v > 1.0 else 'black'
            labels.append(f"<span style='{halo} color: {color_hex};'><b>{v:.2f}</b></span>")
            
        return labels
            
    fancy_font = dict(family="Arial Black, sans-serif", size=10)
    text_offset = 2.5 

    # --- TRACE SETUP (Order: Offense -> Defense -> Labels -> Ball) ---
    trajs = [real_def_traj, base_def_traj, sim_def_traj]
    ist_vals_list = [ist_real, ist_base, ist_sim]
    names = ['Real Defense', 'Baseline Defense', 'Optimized Defense']
    colors = [C_REAL, C_BASE, C_SIM]

    for i in range(3):
        col = i + 1
        traj = trajs[i]
        ist_vals = ist_vals_list[i]
        name = names[i]
        def_color = colors[i]

        # 1. Offense
        fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode='markers', marker=dict(color=C_OFF, size=12, line=dict(width=1, color='white')), showlegend=False), row=1, col=col)
        # 2. Defense
        fig.add_trace(go.Scatter(x=traj[0,:,0], y=traj[0,:,1], mode='markers', marker=dict(color=def_color, size=12, line=dict(width=2, color='white')), name=name), row=1, col=col)
        # 3. Threat Labels
        fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_offset, mode='text', text=get_dynamic_labels(ist_vals[0]), textfont=fancy_font, showlegend=False), row=1, col=col)
        # 4. Ball (Top layer)
        fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', marker=dict(color=C_BALL, size=9, line=dict(width=1, color='black')), showlegend=False), row=1, col=col)

    # --- ANIMATION FRAMES ---
    frames = []
    for f in range(num_frames):
        frame_data = []
        for i in range(3):
            traj = trajs[i]
            ist_vals = ist_vals_list[i]
            frame_data.extend([
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]), 
                go.Scatter(x=traj[f,:,0], y=traj[f,:,1]), 
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_offset, text=get_dynamic_labels(ist_vals[f])), 
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]])
            ])
        
        # 12 traces total (4 per column)
        frames.append(go.Frame(data=frame_data, name=str(f), traces=list(range(12))))
    fig.frames = frames

    axis_config = dict(range=x_range, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, constrain='domain')
    yaxis_config = dict(range=y_range, scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True)

    # --- CONSOLIDATED LAYOUT ---
    shapes = draw_plotly_court(xref="x", yref="y") + draw_plotly_court(xref="x2", yref="y2") + draw_plotly_court(xref="x3", yref="y3")

    fig.update_layout(
        title=dict(
            text="Triple Comparison: Real vs Baseline vs Optimized", 
            x=0.5, xanchor='center', y=0.98,
            font=dict(size=18, family="Arial Black")
        ),
        shapes=shapes,
        template="plotly_white", 
        width=1000,   
        height=520,   # Adjusted for better fit
        margin=dict(l=5, r=5, t=60, b=120), # Tightened bottom margin
        
        xaxis=axis_config, yaxis=yaxis_config,
        xaxis2=axis_config, yaxis2=dict(range=y_range, scaleanchor="x2", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        xaxis3=axis_config, yaxis3=dict(range=y_range, scaleanchor="x3", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
        
        updatemenus=[dict(
            type="buttons", x=0.5, y=-0.12, # Pulled up
            xanchor="center", yanchor="top", direction="right",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        
        # --- ENHANCED SLIDER ---
        sliders=[dict(
            active=0,
            x=0.5, y=-0.22, # Pulled up from the bottom
            len=0.85, 
            xanchor="center", yanchor="top",
            pad={"t": 20, "b": 10}, # Adds space around the bar
            currentvalue={
                "visible": True, 
                "prefix": "Frame: ", 
                "xanchor": "right", 
                "font": {"size": 14, "color": "#666"}
            },
            transition={"duration": 0}, # Makes it feel snappy
            # Styling the bar itself
            tickcolor="#666",
            font={"color": "#666"},
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=False))]) for k in range(num_frames)]
        )]
    )
    
    return fig

def animate_ontop_comparison(sim_no_ist_traj, sim_ist_traj, real_def_traj, 
                              ist_sim_no_ist, ist_sim_ist, ist_real, 
                              off_traj, ball_traj, off_ids, shot_frame, use_images=True):
    """
    Generates a 1x2 dashboard with independent toggles for all elements.
    Left: Single-court comparison (Legends moved below the court).
    Right: Line chart tracking Total Team IST up to the point of the shot.
    """
    num_frames = off_traj.shape[0]
    
    # Ensure shot_frame doesn't exceed our array bounds
    shot_idx = min(shot_frame, num_frames - 1)
    
    # --- Calculate Total Team IST (Threat) per frame ---
    tot_ist_real = np.sum(ist_real, axis=1)
    tot_ist_base = np.sum(ist_sim_no_ist, axis=1)
    tot_ist_sim = np.sum(ist_sim_ist, axis=1)
    
    # Slice the chart data so it ONLY goes up to the shot
    frames_seq_chart = list(range(shot_idx + 1))
    tot_ist_real_chart = tot_ist_real[:shot_idx + 1]
    tot_ist_base_chart = tot_ist_base[:shot_idx + 1]
    tot_ist_sim_chart = tot_ist_sim[:shot_idx + 1]
    
    # Base the Y-axis maximum purely on the pre-shot data
    max_ist = max(np.max(tot_ist_real_chart), np.max(tot_ist_base_chart), np.max(tot_ist_sim_chart))
    y_max = max_ist * 1.1 
    
    # Handle Optional Images
    encoded_images = {}
    if use_images:
        encoded_images = {int(pid): get_base64_image(int(pid)) for pid in off_ids}
    
    img_size = 8 
    off_mode = 'markers' 

    fig = make_subplots(
        rows=1, cols=2, 
        column_widths=[0.7, 0.3], 
        subplot_titles=("Court View (Click Legends Below to Toggle Layers)", "Total Threat (IST) Before Shot"),
        horizontal_spacing=0.05
    )

    def build_frame_images(f_idx):
        if not use_images: return []
        frame_imgs = []
        for p_idx, pid in enumerate(off_ids):
            pid_int = int(pid) 
            if pid_int in encoded_images and encoded_images[pid_int]: 
                frame_imgs.append(dict(
                    source=encoded_images[pid_int], 
                    xref="x", yref="y",
                    x=off_traj[f_idx, p_idx, 0],
                    y=off_traj[f_idx, p_idx, 1],
                    sizex=img_size, sizey=img_size,
                    xanchor="center", yanchor="middle",
                    layer="below", sizing="stretch"
                ))
        return frame_imgs

    def get_vector_coords(base_traj, target_traj, f):
        x, y = [], []
        for i in range(5):
            x.extend([base_traj[f, i, 0], target_traj[f, i, 0], None])
            y.extend([base_traj[f, i, 1], target_traj[f, i, 1], None])
        return x, y

    def get_labels(vals, color_hex):
        labels = []
        halo = "text-shadow: 1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff;"
        for v in vals:
            labels.append(f"<span style='{halo} color: {color_hex};'><b>{v:.2f}</b></span>")
        return labels

    fancy_font = dict(family="Arial Black, sans-serif", size=14)

    # ==========================================
    # LEFT SUBPLOT: COURT VIEW TRACES (Legend 1)
    # ==========================================
    
    # 0: Offense
    marker_dict = dict(color='red', size=10, opacity=0 if use_images else 1)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode=off_mode, marker=marker_dict, name='Offense Players'), row=1, col=1)
    
    # 1: Ball
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', marker=dict(color='orange', size=8, line=dict(width=2, color='black')), name='Ball'), row=1, col=1)
    
    # 2: Real Defense
    fig.add_trace(go.Scatter(x=real_def_traj[0,:,0], y=real_def_traj[0,:,1], mode='markers', marker=dict(color='blue', size=12, line=dict(width=1, color='white')), name='Model: Real Def', visible=True), row=1, col=1)
    
    # 3: IST Defense
    fig.add_trace(go.Scatter(x=sim_ist_traj[0,:,0], y=sim_ist_traj[0,:,1], mode='markers', marker=dict(color='green', size=12, opacity=0.4, line=dict(color='green', width=3)), name='Model: Enhanced (IST)', visible=True), row=1, col=1)
    
    # 4: Baseline Defense
    fig.add_trace(go.Scatter(x=sim_no_ist_traj[0,:,0], y=sim_no_ist_traj[0,:,1], mode='markers', marker=dict(color='purple', size=12, opacity=0.4, line=dict(color='purple', width=3)), name='Model: Baseline', visible='legendonly'), row=1, col=1)

    # Tethers
    vx_ri, vy_ri = get_vector_coords(real_def_traj, sim_ist_traj, 0)
    vx_rn, vy_rn = get_vector_coords(real_def_traj, sim_no_ist_traj, 0)
    
    # 5: Tether (Real -> IST)
    fig.add_trace(go.Scatter(x=vx_ri, y=vy_ri, mode='lines', line=dict(color='green', dash='dot', width=2), name='Tether: Real ➔ Enhanced', visible='legendonly'), row=1, col=1)
    # 6: Tether (Real -> Baseline)
    fig.add_trace(go.Scatter(x=vx_rn, y=vy_rn, mode='lines', line=dict(color='purple', dash='dot', width=2), name='Tether: Real ➔ Baseline', visible='legendonly'), row=1, col=1)

    text_offset = 4.0 if use_images else 2.0
    
    # 7: IST Labels (Real - Blue)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_offset, mode='text', text=get_labels(ist_real[0], 'blue'), textfont=fancy_font, name='IST Labels: Real', visible='legendonly'), row=1, col=1)
    
    # 8: IST Labels (Enhanced - Green)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] - text_offset, mode='text', text=get_labels(ist_sim_ist[0], 'green'), textfont=fancy_font, name='IST Labels: Enhanced', visible='legendonly'), row=1, col=1)
    
    # 9: IST Labels (Baseline - Purple)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0] + text_offset, y=off_traj[0,:,1], mode='text', text=get_labels(ist_sim_no_ist[0], 'purple'), textfont=fancy_font, name='IST Labels: Baseline', visible='legendonly'), row=1, col=1)


    # ==========================================
    # RIGHT SUBPLOT: CHART TRACES (Legend 2)
    # ==========================================
    
    # 10: Real Line
    fig.add_trace(go.Scatter(x=frames_seq_chart, y=tot_ist_real_chart, mode='lines', line=dict(color='blue', width=2), name='Real Threat Line', legend='legend2'), row=1, col=2)
    # 11: Enhanced Line
    fig.add_trace(go.Scatter(x=frames_seq_chart, y=tot_ist_sim_chart, mode='lines', line=dict(color='green', width=2), name='Enhanced Threat Line', legend='legend2'), row=1, col=2)
    # 12: Baseline Line
    fig.add_trace(go.Scatter(x=frames_seq_chart, y=tot_ist_base_chart, mode='lines', line=dict(color='purple', width=2, dash='dash'), name='Baseline Threat Line', visible='legendonly', legend='legend2'), row=1, col=2)

    # 13: Static Shot Indicator Line
    fig.add_trace(go.Scatter(x=[shot_idx, shot_idx], y=[0, y_max], mode='lines', line=dict(color='gray', width=2, dash='dash'), name='Shot Taken', hoverinfo='none', showlegend=False), row=1, col=2)

    # 14: Moving Time Indicator (Vertical Line)
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, y_max], mode='lines', line=dict(color='black', width=2), showlegend=False, name='Current Frame', hoverinfo='none'), row=1, col=2)


    # ==========================================
    # BUILD ANIMATION FRAMES
    # ==========================================
    frames = []
    for f in range(num_frames):
        vx_ri, vy_ri = get_vector_coords(real_def_traj, sim_ist_traj, f)
        vx_rn, vy_rn = get_vector_coords(real_def_traj, sim_no_ist_traj, f)
        
        frames.append(go.Frame(
            data=[
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]), # 0
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]), # 1
                go.Scatter(x=real_def_traj[f,:,0], y=real_def_traj[f,:,1]), # 2
                go.Scatter(x=sim_ist_traj[f,:,0], y=sim_ist_traj[f,:,1]), # 3
                go.Scatter(x=sim_no_ist_traj[f,:,0], y=sim_no_ist_traj[f,:,1]), # 4
                go.Scatter(x=vx_ri, y=vy_ri), # 5
                go.Scatter(x=vx_rn, y=vy_rn), # 6
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_offset, text=get_labels(ist_real[f], 'blue')), # 7
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] - text_offset, text=get_labels(ist_sim_ist[f], 'green')), # 8
                go.Scatter(x=off_traj[f,:,0] + text_offset, y=off_traj[f,:,1], text=get_labels(ist_sim_no_ist[f], 'purple')), # 9
                go.Scatter(x=[f, f], y=[0, y_max]) # 14 -> mapped to index 10 in update logic
            ],
            layout=go.Layout(images=build_frame_images(f)),
            name=str(f),
            # Notice traces 10, 11, 12, 13 are omitted here because they are static! Saves rendering power.
            traces=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14] 
        ))
    fig.frames = frames

    # ==========================================
    # LAYOUT & HORIZONTAL LEGENDS
    # ==========================================
    axis_clean = dict(range=[-5, 94], showgrid=False, zeroline=False, showticklabels=False)
    yaxis_clean = dict(range=[-5, 55], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)

    fig.update_layout(
        autosize=True,
        title="Interactive IST Geometry Sandbox",
        shapes=draw_plotly_court(xref="x", yref="y"), 
        images=build_frame_images(0),
        template="plotly_white",
        width=None, height=800, # Increased height to fit bottom menus
        margin=dict(l=20, r=20, t=80, b=220), # Huge bottom margin to fit controls/legends
        
        # Court Axis
        xaxis=axis_clean, yaxis=yaxis_clean,
        
        # Line Chart Axis (Bounded dynamically up to just after the shot)
        xaxis2=dict(title="Frame", showgrid=True, range=[0, shot_idx + 5]),
        yaxis2=dict(title="Total Team Threat (IST)", showgrid=True, range=[0, y_max]),
        
        # Legend 1: Court Toggles (Moved BELOW the plot)
        legend=dict(
            title="Court Layers",
            orientation="h",
            yanchor="top", y=-0.05, 
            xanchor="left", x=0.0, 
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1
        ),
        
        # Legend 2: Chart Toggles (Moved BELOW the line chart)
        legend2=dict(
            title="Chart Lines",
            orientation="h",
            yanchor="top", y=-0.05, 
            xanchor="left", x=0.72, 
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1
        ),
        
        # Shifted buttons and sliders down below the legends
        updatemenus=[
            dict(
                type="buttons", x=0.0, y=-0.22, xanchor="left", yanchor="top", direction="right",
                buttons=[
                    # fromcurrent: True ensures hitting play resumes where you paused
                    dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ]
            )
        ],
        sliders=[dict(
            active=0, x=0.0, y=-0.35, len=1.0, xanchor="left", yanchor="top",
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True))]) 
                   for k in range(num_frames)]
        )]
    )
    
    return fig.show(renderer="browser")

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def animate_side_by_side_courts(off_traj, real_def_traj, sim_def_traj, 
                                ball_traj, ist_real, ist_sim, off_ids=[], 
                                half_court='left'):
    # --- 1. SHAPE SAFETY ---
    off_traj = np.array(off_traj).reshape(-1, 5, 2)
    real_def_traj = np.array(real_def_traj).reshape(-1, 5, 2)
    sim_def_traj = np.array(sim_def_traj).reshape(-1, 5, 2)
    ball_traj = np.array(ball_traj).reshape(-1, 2)
    ist_real = np.array(ist_real).reshape(-1, 5)
    ist_sim = np.array(ist_sim).reshape(-1, 5)
    
    num_frames = off_traj.shape[0]
    C_OFF, C_REAL, C_SIM, C_BALL = '#C8102E', '#888888', '#1D428A', '#ec7607'
    
    # Half-court logic
    x_range = [-2, 49] if half_court == 'left' else [45, 96]
    y_range = [-2, 54]

    def get_dynamic_labels(vals):
        labels = []
        halo = "text-shadow: 1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff;"
        for v in vals:
            color_hex = C_OFF if v > 1.0 else 'black'
            labels.append(f"<span style='{halo} color: {color_hex};'><b>{v:.2f}</b></span>")
        return labels

    # horizontal_spacing at 0.04 provides a clean gap for the 1000px width
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5], 
        subplot_titles=("Real NBA Defense", "Optimized JKO Defense"),
        horizontal_spacing=0.04 
    )

    fancy_font = dict(family="Arial Black, sans-serif", size=10)
    text_offset = 2.5 

    # --- TRACE SETUP (Order: Offense -> Defense -> Labels -> Ball) ---
    for col, traj, ist_vals, name, def_color in zip([1, 2], [real_def_traj, sim_def_traj], [ist_real, ist_sim], 
                                                     ['Real Defense', 'Optimized Defense'], [C_REAL, C_SIM]):
        # 1. Offense
        fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode='markers', marker=dict(color=C_OFF, size=12, line=dict(width=1, color='white')), showlegend=False), row=1, col=col)
        # 2. Defense
        fig.add_trace(go.Scatter(x=traj[0,:,0], y=traj[0,:,1], mode='markers', marker=dict(color=def_color, size=12, line=dict(width=2, color='white')), name=name), row=1, col=col)
        # 3. Threat Labels
        fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_offset, mode='text', text=get_dynamic_labels(ist_vals[0]), textfont=fancy_font, showlegend=False), row=1, col=col)
        # 4. Ball (Added last so it's on top)
        fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', marker=dict(color=C_BALL, size=9, line=dict(width=1, color='black')), showlegend=False), row=1, col=col)

    # --- ANIMATION FRAMES ---
    frames = []
    for f in range(num_frames):
        frames.append(go.Frame(data=[
            # Left Court (Traces 0-3)
            go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]), 
            go.Scatter(x=real_def_traj[f,:,0], y=real_def_traj[f,:,1]), 
            go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_offset, text=get_dynamic_labels(ist_real[f])), 
            go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]), 
            # Right Court (Traces 4-7)
            go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]), 
            go.Scatter(x=sim_def_traj[f,:,0], y=sim_def_traj[f,:,1]), 
            go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_offset, text=get_dynamic_labels(ist_sim[f])),
            go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]])
        ], name=str(f), traces=[0, 1, 2, 3, 4, 5, 6, 7]))
    fig.frames = frames

    axis_config = dict(range=x_range, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, constrain='domain')
    yaxis_config = dict(range=y_range, scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True)

    # --- THE "GOLDILOCKS" LAYOUT ---
    fig.update_layout(
        title=dict(text="Side-by-Side Defense Comparison", x=0.5, xanchor='center', font=dict(size=18, family="Arial Black")),
        shapes=draw_plotly_court(xref="x", yref="y") + draw_plotly_court(xref="x2", yref="y2"),
        template="plotly_white", 
        
        # --- FIXED SIZE TO FIT NOTEBOOK CONTAINER ---
        width=1000,   # Standard width for Jupyter cells
        height=550,   # Proportional height for half-courts
        margin=dict(l=20, r=20, t=80, b=120),
        
        xaxis=axis_config, yaxis=yaxis_config,
        xaxis2=axis_config, yaxis2=dict(range=y_range, scaleanchor="x2", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
        
        updatemenus=[dict(type="buttons", x=0.5, y=-0.12, xanchor="center", yanchor="top", direction="right",
                buttons=[dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]),
                         dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
        
        sliders=[dict(active=0, x=0.5, y=-0.22, len=0.5, xanchor="center", yanchor="top",
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=False))]) for k in range(num_frames)])]
    )
    
    return fig

def animate_comparison(sim_ist_traj, real_def_traj, ist_sim_ist, ist_real, 
                       off_traj, ball_traj, off_ids, shot_frame, use_images=True):
    """
    Superimposed 1x2 dashboard with LA Clippers Colors and Dynamic IST Labels.
    Fixed spacing: Legends are stacked vertically and controls pushed down.
    """
    num_frames = off_traj.shape[0]
    shot_idx = min(shot_frame, num_frames - 1)
    
    tot_ist_real = np.sum(ist_real, axis=1)
    tot_ist_sim = np.sum(ist_sim_ist, axis=1)
    
    frames_seq_chart = list(range(shot_idx + 1))
    tot_ist_real_chart = tot_ist_real[:shot_idx + 1]
    tot_ist_sim_chart = tot_ist_sim[:shot_idx + 1]
    
    max_ist = max(np.max(tot_ist_real_chart), np.max(tot_ist_sim_chart))
    y_max = max_ist * 1.1 
    
    c_off = '#C8102E'      
    c_real = '#888888'     
    c_sim = '#1D428A'      
    c_ball = '#ec7607'     
    
    encoded_images = {}
    if use_images:
        encoded_images = {int(pid): get_base64_image(int(pid)) for pid in off_ids}
    
    img_size = 8 
    off_mode = 'markers' 

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4], 
        subplot_titles=("Court: Real vs. Opt", "Threat Metric"),
        horizontal_spacing=0.05
    )

    def build_frame_images(f_idx):
        if not use_images: return []
        frame_imgs = []
        for p_idx, pid in enumerate(off_ids):
            pid_int = int(pid) 
            if pid_int in encoded_images and encoded_images[pid_int]: 
                frame_imgs.append(dict(
                    source=encoded_images[pid_int], xref="x", yref="y",
                    x=off_traj[f_idx, p_idx, 0], y=off_traj[f_idx, p_idx, 1],
                    sizex=img_size, sizey=img_size, xanchor="center", yanchor="middle", layer="below"
                ))
        return frame_imgs

    def get_vector_coords(base_traj, target_traj, f):
        x, y = [], []
        for i in range(5):
            x.extend([base_traj[f, i, 0], target_traj[f, i, 0], None])
            y.extend([base_traj[f, i, 1], target_traj[f, i, 1], None])
        return x, y

    def get_dynamic_labels(vals):
        labels = []
        halo = "text-shadow: 1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff;"
        for v in vals:
            color_hex = c_off if v > 1.0 else 'black'
            labels.append(f"<span style='{halo} color: {color_hex};'><b>{v:.2f}</b></span>")
        return labels

    fancy_font = dict(family="Arial Black, sans-serif", size=14)
    text_offset = 4.0 if use_images else 2.0

    # Traces
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode=off_mode, marker=dict(color=c_off, size=10, opacity=0 if use_images else 1), name='Offense'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', marker=dict(color=c_ball, size=8, line=dict(width=2, color='black')), name='Ball'), row=1, col=1)
    fig.add_trace(go.Scatter(x=real_def_traj[0,:,0], y=real_def_traj[0,:,1], mode='markers', marker=dict(color=c_real, size=12, line=dict(width=1, color='white')), name='Real Defense', visible=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=sim_ist_traj[0,:,0], y=sim_ist_traj[0,:,1], mode='markers', marker=dict(color=c_sim, size=12, opacity=0.8, line=dict(color='white', width=1)), name='Optimized (JKO)', visible=True), row=1, col=1)
    
    vx_ri, vy_ri = get_vector_coords(real_def_traj, sim_ist_traj, 0)
    fig.add_trace(go.Scatter(x=vx_ri, y=vy_ri, mode='lines', line=dict(color=c_sim, dash='dot', width=2), name='Correction Vector', visible='legendonly'), row=1, col=1)

    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_offset, mode='text', text=get_dynamic_labels(ist_real[0]), textfont=fancy_font, name='IST Labels: Real', visible='legendonly'), row=1, col=1)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] - text_offset, mode='text', text=get_dynamic_labels(ist_sim_ist[0]), textfont=fancy_font, name='IST Labels: Optimized', visible='legendonly'), row=1, col=1)

    fig.add_trace(go.Scatter(x=frames_seq_chart, y=tot_ist_real_chart, mode='lines', line=dict(color=c_real, width=3), name='Real Threat', legend='legend2'), row=1, col=2)
    fig.add_trace(go.Scatter(x=frames_seq_chart, y=tot_ist_sim_chart, mode='lines', line=dict(color=c_sim, width=3), name='Optimized Threat', legend='legend2'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[shot_idx, shot_idx], y=[0, y_max], mode='lines', line=dict(color='gray', width=2, dash='dash'), name='Shot Taken', hoverinfo='none', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, y_max], mode='lines', line=dict(color='black', width=2), showlegend=False, name='Current Frame', hoverinfo='none'), row=1, col=2)

    frames = []
    for f in range(num_frames):
        vx_ri, vy_ri = get_vector_coords(real_def_traj, sim_ist_traj, f)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]),
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]),
                go.Scatter(x=real_def_traj[f,:,0], y=real_def_traj[f,:,1]),
                go.Scatter(x=sim_ist_traj[f,:,0], y=sim_ist_traj[f,:,1]),
                go.Scatter(x=vx_ri, y=vy_ri),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_offset, text=get_dynamic_labels(ist_real[f])),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] - text_offset, text=get_dynamic_labels(ist_sim_ist[f])),
                go.Scatter(x=[f, f], y=[0, y_max])
            ],
            layout=go.Layout(images=build_frame_images(f)),
            name=str(f),
            traces=[0, 1, 2, 3, 4, 5, 6, 10] 
        ))
    fig.frames = frames

    # ==========================================
    # LAYOUT (Stacked Legend Fix)
    # ==========================================
    axis_clean = dict(range=[-5, 94], showgrid=False, zeroline=False, showticklabels=False)
    yaxis_clean = dict(range=[-5, 55], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)

    fig.update_layout(
        title=dict(
            text="Interactive Defensive Analysis",
            x=0.5, xanchor='center', font=dict(size=22, family="Arial Black")
        ),
        shapes=draw_plotly_court(xref="x", yref="y"),
        template="plotly_white", width=1400, height=850, 
        margin=dict(l=50, r=50, t=80, b=350), # Balanced side margins
        
        xaxis=dict(range=[-2, 49], showgrid=False, zeroline=False, showticklabels=False, constrain='domain'),
        yaxis=dict(range=[-2, 54], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, constrain='domain'),
        
        xaxis2=dict(title="Frame", showgrid=True, range=[0, shot_idx + 5]),
        yaxis2=dict(title="Total Team Threat (IST)", showgrid=True, range=[0, y_max]),
        
        # --- CENTERED STACKED LEGENDS ---
        legend=dict(
            title="Court Layers", orientation="h", 
            yanchor="top", y=-0.1, xanchor="center", x=0.5, # Centered horizontally
            bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1
        ),
        legend2=dict(
            title="Chart Lines", orientation="h", 
            yanchor="top", y=-0.20, xanchor="center", x=0.5, # Centered horizontally
            bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1
        ),
        
        # --- CENTERED BUTTONS ---
        updatemenus=[dict(
            type="buttons", x=0.5, y=-0.32, xanchor="center", yanchor="top", direction="right",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ])],

        # --- SMALLER CENTERED SLIDER (Timeline) ---
        sliders=[dict(
            active=0, x=0.5, y=-0.44, 
            xanchor="center", yanchor="top",
            len=0.6, # Reduced width from 1.0 to 0.6
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True))]) 
                   for k in range(num_frames)])]
    )
    fig.update_layout(
        width=None, 
        height=700,
        autosize=True,
        margin=dict(l=20, r=20, t=80, b=300),
        xaxis=dict(range=[-2, 49], constrain='domain'), # Zoom to half-court
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1),
        legend2=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
        sliders=[dict(len=0.6, x=0.5, xanchor="center", y=-0.4)]
    )
    
    return fig

# --- Gif Functions --- # 

def save_simulation_gif(sim_ist_traj, real_def_traj, off_traj, ball_traj, 
                         ist_sim, ist_real, 
                         filename='styled_dual_comparison.gif', half_court='left'):
    """
    Saves a 1x2 styled Matplotlib GIF comparing: Real Defense | Simulated Defense
    Matches the modern color scheme, clips floating text, and plays at true 25 FPS.
    """
    print(f"Generating styled 25fps GIF: {filename}...")
    
    # --- STYLING CONSTANTS ---
    C_OFF = '#C8102E'      # Ember Red
    C_REAL = '#888888'     # Grey
    C_SIM = '#1D428A'      # Naval Blue
    C_BALL = '#ec7607'     
    C_COURT = '#BEC0C2'
    
    # 1. Setup Figure (White Background)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor('white') 
    
    x_min, x_max = (-2, 49) if half_court == 'left' else ((45, 96) if half_court == 'right' else (-2, 96))
    
    axes = [ax1, ax2]
    titles = ['Real NBA Defense', 'Simulated Defense (IST Model)']
    
    for ax, title in zip(axes, titles):
        ax.set_facecolor('none')
        draw_court_matplotlib(ax, color=C_COURT, lw=1.5, half_court=half_court)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-2, 54)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=18, fontweight='black', pad=15)
        ax.axis('off')
        
        # Subtle border around the court
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(C_SIM)
            spine.set_linewidth(2)

    # 2. Initialize Visual Elements (With clip_on=True for text)
    def create_elements(ax, is_sim=False):
        return {
            'off': ax.scatter([], [], c=C_OFF, s=200, edgecolors='white', linewidth=1.0, zorder=4),
            'def': ax.scatter([], [], c=C_SIM if is_sim else C_REAL, s=200, 
                              alpha=0.9 if is_sim else 1.0,
                              edgecolors='white' if is_sim else C_SIM, 
                              linewidth=1.0 if is_sim else 2.0, zorder=4),
            'ball': ax.scatter([], [], c=C_BALL, s=120, edgecolors='none', zorder=7),
            
            # TEXT CLIPPING FIX: clip_on=True stops the numbers from floating off-court
            'labels': [ax.text(0, 0, '', fontsize=11, fontweight='bold', ha='center', va='center', 
                               zorder=8, clip_on=True, 
                               bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65, lw=1.5)) 
                       for _ in range(5)]
        }

    e1 = create_elements(ax1, is_sim=False) # Real
    e2 = create_elements(ax2, is_sim=True)  # Sim

    def update(f):
        # Update Positions
        e1['def'].set_offsets(real_def_traj[f])
        e1['off'].set_offsets(off_traj[f])
        e1['ball'].set_offsets(ball_traj[f])
        
        e2['def'].set_offsets(sim_ist_traj[f])
        e2['off'].set_offsets(off_traj[f])
        e2['ball'].set_offsets(ball_traj[f])

        # Update IST Labels and Box Clipping
        for p in range(5):
            ox, oy = off_traj[f, p, 0], off_traj[f, p, 1]
            
            # --- Real Side ---
            val_real = ist_real[f, p]
            text_color_r = C_OFF if val_real > 1.2 else C_SIM
            e1['labels'][p].set_position((ox, oy + 2.8))
            e1['labels'][p].set_text(f"{val_real:.2f}")
            e1['labels'][p].set_color(text_color_r)
            e1['labels'][p].get_bbox_patch().set_edgecolor(text_color_r)
            e1['labels'][p].get_bbox_patch().set_clip_on(True) # Force box to clip
            
            # --- Sim Side ---
            val_sim = ist_sim[f, p]
            text_color_s = C_OFF if val_sim > 1.2 else C_SIM
            e2['labels'][p].set_position((ox, oy + 2.8))
            e2['labels'][p].set_text(f"{val_sim:.2f}")
            e2['labels'][p].set_color(text_color_s)
            e2['labels'][p].get_bbox_patch().set_edgecolor(text_color_s)
            e2['labels'][p].get_bbox_patch().set_clip_on(True) # Force box to clip

        return ()

    # 3. Create Custom Legend
    custom_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_OFF, markersize=12, label='Offense'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BALL, markersize=8, label='Ball'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_REAL, markeredgecolor=C_SIM, markeredgewidth=2, markersize=12, label='Real Defense'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SIM, markersize=12, label='Simulated Defense')
    ]
    
    fig.legend(handles=custom_handles, loc='lower center', ncol=4, fontsize=12, 
               bbox_to_anchor=(0.5, 0.05), frameon=True, facecolor='white', 
               edgecolor=C_SIM, borderpad=0.8)

    # 4. Animate and Save (TRUE 25 FPS REAL-TIME FIX)
    
    # Process every single frame (step = 1)
    frames_to_run = range(0, len(off_traj), 1)
    
    # Interval=40ms perfectly matches 25fps data (1000ms / 25 = 40ms)
    anim = FuncAnimation(fig, update, frames=frames_to_run, interval=40, blit=True)
    
    # Force the GIF writer to output at exactly 25 FPS
    anim.save(filename, writer='pillow', fps=25)
    plt.close()
    print(f"Successfully saved 25fps styled GIF to {filename}")

def save_triple_simulation_gif(sim_no_ist_traj, sim_ist_traj, real_traj,
                               ist_sim_ist, ist_sim_no_ist, ist_real, 
                               offenders_traj, ball_traj, 
                               filename='triple_comparison.gif', half_court='left'):
    """
    Saves a 1x3 high-quality Matplotlib GIF with Plotly-like styling.
    Comparing: Real | Sim (IST) | Sim (Baseline)
    """
    print(f"Generating {filename}...")
    
    # 1. Setup Figure and Axis ranges
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    x_min, x_max = (0, 47) if half_court == 'left' else ((47, 94) if half_court == 'right' else (0, 94))
    
    # Pre-draw courts and set styles
    axes = [ax1, ax2, ax3]
    titles = ['Real Defense', 'Simulated (IST)', 'Simulated (Baseline)']
    
    for ax, title in zip(axes, titles):
        draw_court_matplotlib(ax) # Your helper function
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 55) # Extra 5ft for labels
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')

    # 2. Initialize Visual Elements (using your Plotly styles)
    # Styles: Blue/Red markers (size 10), White edges, Star ball (size 12)
    def create_elements(ax, alpha_def=1.0):
        return {
            'def': ax.scatter([], [], c='blue', s=100, edgecolors='white', linewidth=1, alpha=alpha_def, zorder=3),
            'off': ax.scatter([], [], c='red', s=100, edgecolors='white', linewidth=1, zorder=3),
            'ball': ax.scatter([], [], c='orange', s=150, marker='*', edgecolors='black', linewidth=0.8, zorder=4),
            'labels': [ax.text(0, 0, '', fontsize=9, fontweight='bold', ha='center', va='bottom') for _ in range(5)]
        }

    # Create for each subplot
    e1 = create_elements(ax1)
    e2 = create_elements(ax2, alpha_def=0.8) # Sim IST slightly transparent
    e3 = create_elements(ax3, alpha_def=0.4) # Baseline very transparent

    def update(f):
        # We subsample by 2 inside the FuncAnimation by passing frames=range(0, len, 2)
        
        # Update Real
        e1['def'].set_offsets(real_traj[f])
        e1['off'].set_offsets(offenders_traj[f])
        e1['ball'].set_offsets(ball_traj[f])
        
        # Update Sim IST
        e2['def'].set_offsets(sim_ist_traj[f])
        e2['off'].set_offsets(offenders_traj[f])
        e2['ball'].set_offsets(ball_traj[f])
        
        # Update Sim Baseline
        e3['def'].set_offsets(sim_no_ist_traj[f])
        e3['off'].set_offsets(offenders_traj[f])
        e3['ball'].set_offsets(ball_traj[f])

        # Update IST Labels for all 3
        for i in range(5):
            pos = offenders_traj[f, i]
            # Real labels
            e1['labels'][i].set_position((pos[0], pos[1] + 1.5))
            e1['labels'][i].set_text(f"{ist_real[f, i]:.2f}")
            e1['labels'][i].set_color('red' if ist_real[f, i] > 1.2 else 'black')
            
            # IST labels
            e2['labels'][i].set_position((pos[0], pos[1] + 1.5))
            e2['labels'][i].set_text(f"{ist_sim_ist[f, i]:.2f}")
            e2['labels'][i].set_color('red' if ist_sim_ist[f, i] > 1.2 else 'black')
            
            # Baseline labels
            e3['labels'][i].set_position((pos[0], pos[1] + 1.5))
            e3['labels'][i].set_text(f"{ist_sim_no_ist[f, i]:.2f}")
            e3['labels'][i].set_color('red' if ist_sim_no_ist[f, i] > 1.2 else 'black')

        return ()

    # 3. Animate and Save
    # Subsample frames: range(0, total, 2)
    frames_to_run = range(0, len(real_traj), 2)
    anim = FuncAnimation(fig, update, frames=frames_to_run, interval=80, blit=True)
    
    # Save using Pillow (Native on M1)
    anim.save(filename, writer='pillow', fps=12.5)
    plt.close()
    print(f"Successfully saved {filename}")

import jax
import jax.numpy as jnp
from jax import vmap, jit
import plotly.graph_objects as go
import numpy as np

def create_jax_integrated_simulation(def_traj, off_traj, ball_traj, q_traj, basket_pos, params, filename="defensive_identity.html"):
    """
    JAX-powered simulation that integrates Ball Pressure, IST, and Offender fields.
    Deep Red = Highest Potential (Target Pressure Zones).
    """
    timesteps = def_traj.shape[0]
    # 1. Setup Court Grid
    res_x, res_y = 60, 45
    x_grid = jnp.linspace(0, 47, res_x)
    y_grid = jnp.linspace(0, 50, res_y)
    XX, YY = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    # 2. Define the System-Level Surface Calculation
    @jit
    def compute_surface_frame(defenders, offenders, ball, q_vals):
        def probe_system_energy(point):
            # Evaluate how the system energy changes if a 'probe' moves to [x,y]
            # We use defender 0 as the probe, keeping the rest of the unit static
            test_defs = defenders.at[0].set(point)
            # Use your original integrated function (Field + Ball + IST)
            e_vector = total_energy(test_defs, offenders, q_vals, ball, basket_pos, params)
            return jnp.sum(e_vector)
        
        # Calculate -Energy across the whole grid so that Minima = Red Peaks
        z_values = vmap(probe_system_energy)(grid_points)
        return -z_values.reshape(res_y, res_x)

    # 3. Compute all timesteps
    print(f"Calculating JAX potential surfaces for {timesteps} frames...")
    all_z = []
    for t in range(timesteps):
        z = compute_surface_frame(def_traj[t], off_traj[t], ball_traj[t], q_traj[t])
        all_z.append(np.array(z))
    
    # 4. Construct Plotly Visualization
    fig = go.Figure()
    
    # Theme: Deep Blue (Low) to Deep Red (High/Pressure)
    colorscale = [[0,"rgb(5,48,97)"],[0.5,"rgb(247,247,247)"],[1,"rgb(103,0,31)"]]

    # Initial Traces
    fig.add_trace(go.Heatmap(z=all_z[0], x=np.array(x_grid), y=np.array(y_grid), 
                             colorscale=colorscale, zsmooth='best', name="Pressure Field"))
    fig.add_trace(go.Scatter(x=def_traj[0,:,0], y=def_traj[0,:,1], mode='markers', 
                             marker=dict(color='blue', size=12, line=dict(color='white', width=2)), name='Defenders'))
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode='markers', 
                             marker=dict(color='red', size=10), name='Offenders'))
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', 
                             marker=dict(color='orange', size=15, symbol='star'), name='Ball'))

    # Animation Frames
    frames = [go.Frame(data=[
        go.Heatmap(z=all_z[t]),
        go.Scatter(x=def_traj[t,:,0], y=def_traj[t,:,1]),
        go.Scatter(x=off_traj[t,:,0], y=off_traj[t,:,1]),
        go.Scatter(x=[ball_traj[t,0]], y=[ball_traj[t,1]])
    ], name=str(t)) for t in range(timesteps)]
    
    fig.frames = frames

    # Layout & Sliders
    fig.update_layout(
        title="Topological Defensive Identity: Integrated Potential Surface",
        xaxis=dict(range=[0, 47], visible=False),
        yaxis=dict(range=[0, 50], visible=False, scaleanchor="x", scaleratio=1),
        updatemenus=[{"type": "buttons", "buttons": [{"label": "Play", "method": "animate"}]}],
        sliders=[{"steps": [{"label": str(t), "method": "animate", "args": [[str(t)]]} for t in range(timesteps)]}]
    )

    fig.write_html(filename)
    print(f"Interactive HTML saved as {filename}")
    fig.show(renderer="notebook_connected")

def create_team_pressure_heatmap(def_traj, off_traj, ball_traj, q_values, basket, params, 
                                 filename='team_pressure_surface.html', half_court='left', step=2):
    """
    Plots the combined defensive potential field across the entire court.
    Shows where the defense is successfully taking away space from all offenders.
    """
    print(f"Generating Team Pressure Heatmap: {filename}...")
    
    C_OFF, C_SIM, C_BALL = '#C8102E', '#1D428A', '#ec7607'
    x_min, x_max = (0, 47) if half_court == 'left' else ((47, 94) if half_court == 'right' else (0, 94))
    
    # 1. Setup Grid
    nx, ny = 60, 30
    xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, nx), jnp.linspace(0, 50, ny))
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    frames_data = []
    global_abs_max = 0

    # 2. Frame-by-frame Calculation
    frames_to_run = list(range(0, len(def_traj), step))
    for f in frames_to_run:
        cur_def = def_traj[f]
        cur_off = off_traj[f]
        cur_ball = ball_traj[f]
        cur_q = q_values[f]
        
        # PERSPECTIVE SHIFT: We calculate the potential felt by an 
        # offensive "test point" at every spot on the grid.
        def get_team_field_at_point(point):
            # We treat the 'point' as a proxy for an offensive threat
            # and calculate the IST penalty the defense is applying to it.
            return _calculate_ist_penalty(cur_def, point[None, :], cur_q[0:1], cur_ball, basket, params)

        energies = vmap(get_team_field_at_point)(grid_points)
        zz = np.array(energies.reshape(ny, nx))
        
        global_abs_max = max(global_abs_max, float(np.max(np.abs(zz))))
        
        frames_data.append({
            'name': str(f), 'z': zz,
            'def_x': cur_def[:, 0], 'def_y': cur_def[:, 1],
            'off_x': cur_off[:, 0], 'off_y': cur_off[:, 1],
            'ball_x': [cur_ball[0]], 'ball_y': [cur_ball[1]]
        })

    # 3. Build Plotly Interface
    init = frames_data[0]
    fig = go.Figure(data=[
        go.Heatmap(x=np.linspace(x_min, x_max, nx), y=np.linspace(0, 50, ny), z=init['z'], 
                   colorscale='Blues', # Showing 'Defensive Pressure' in Blue
                   zmin=0, zmax=global_abs_max,
                   opacity=0.7, showscale=True, name='Defensive Pressure'),
        go.Scatter(x=init['def_x'], y=init['def_y'], mode='markers', 
                   marker=dict(color=C_SIM, size=14, line=dict(width=2, color='white')), name='Defenders'),
        go.Scatter(x=init['off_x'], y=init['off_y'], mode='markers', 
                   marker=dict(color=C_OFF, size=14, line=dict(width=1, color='white')), name='Offenders'),
        go.Scatter(x=init['ball_x'], y=init['ball_y'], mode='markers', 
                   marker=dict(color=C_BALL, size=16, symbol='star', line=dict(width=1, color='black')), name='Ball')
    ])

    # Apply Court Shapes
    court_shapes = draw_plotly_court(xref='x', yref='y', half_court=half_court)
    fig.update_layout(shapes=court_shapes)

    # ... [Same Play/Pause and Slider layout logic as previous function] ...

    fig.write_html(filename, include_plotlyjs='cdn', full_html=True)
    print(f"Successfully saved {filename}")
