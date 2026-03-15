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

# ==== Drawing Functions ===== # 
def draw_court_matplotlib(ax, color='#777777', lw=1.5, half_court=None):
    """Matplotlib version of the high-quality Plotly court with half-court support."""
    
    # 1. Court Perimeter (Fixed doubled line by matching line weight and setting zorder)
    if half_court == 'left':
        ax.add_patch(Rectangle((0, 0), 47, 50, color=color, zorder=0, fill=False, lw=lw))
    elif half_court == 'right':
        ax.add_patch(Rectangle((47, 0), 47, 50, color=color, zorder=0, fill=False, lw=lw))
    else:
        ax.add_patch(Rectangle((0, 0), 94, 50, color=color, zorder=0, fill=False, lw=lw))
    
    # 2. Midcourt line and Circle
    ax.plot([47, 47], [0, 50], color=color, lw=lw, zorder=0)
    if half_court == 'left':
        ax.add_patch(Arc((47, 25), 12, 12, theta1=90, theta2=270, color=color, lw=lw, zorder=0))
    elif half_court == 'right':
        ax.add_patch(Arc((47, 25), 12, 12, theta1=-90, theta2=90, color=color, lw=lw, zorder=0))
    else:
        ax.add_patch(Circle((47, 25), 6, color=color, fill=False, lw=lw, zorder=0))

    # 3. Left Side Features (Fixed 68.3 degree 3pt connection)
    if half_court in [None, 'left']:
        ax.add_patch(Rectangle((0, 17), 19, 16, color=color, fill=False, lw=lw, zorder=0))
        # 3PT Arc mathematically connected to the 14ft corner lines
        ax.add_patch(Arc((5.25, 25), 47.5, 47.5, theta1=-68.3, theta2=68.3, color=color, lw=lw, zorder=0))
        ax.plot([0, 14], [3, 3], color=color, lw=lw, zorder=0)
        ax.plot([0, 14], [47, 47], color=color, lw=lw, zorder=0)
        # Hoop & Backboard
        ax.add_patch(Rectangle((4, 22), 0.2, 6, color="#ec7607", lw=2, zorder=0))
        ax.add_patch(Circle((5.25, 25), 0.75, color="#ec7607", fill=False, lw=2, zorder=0))

    # 4. Right Side Features
    if half_court in [None, 'right']:
        ax.add_patch(Rectangle((75, 17), 19, 16, color=color, fill=False, lw=lw, zorder=0))
        # 3PT Arc (180 +/- 68.3)
        ax.add_patch(Arc((94-5.25, 25), 47.5, 47.5, theta1=111.7, theta2=248.3, color=color, lw=lw, zorder=0))
        ax.plot([80, 94], [3, 3], color=color, lw=lw, zorder=0)
        ax.plot([80, 94], [47, 47], color=color, lw=lw, zorder=0)
        # Hoop & Backboard
        ax.add_patch(Rectangle((90, 22), 0.2, 6, color="#ec7607", lw=2, zorder=0))
        ax.add_patch(Circle((94-5.25, 25), 0.75, color="#ec7607", fill=False, lw=2, zorder=0))

def draw_plotly_court(xref='x', yref='y', half_court=None):
    """
    Returns a list of high-quality NBA court lines for Plotly.
    Refactored to return a list of dicts rather than modifying a fig.
    """
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
    
    # Base shapes
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=94, y1=50, line=dict(color=line_col, width=2), layer='below'),
        dict(type="line", x0=47, y0=0, x1=47, y1=50, line=dict(color=line_col, width=2), layer='below')
    ]

    # Midcourt Circle
    if half_court is None:
        shapes.append(dict(type="circle", x0=41, y0=19, x1=53, y1=31, line=dict(color=line_col, width=2), layer='below'))
    elif half_court == 'left':
        shapes.append(dict(type="path", path=ellipse_arc(47, 25, 6, 6, np.pi/2, 3*np.pi/2), line=dict(color=line_col, width=2), layer='below'))
    elif half_court == 'right':
        shapes.append(dict(type="path", path=ellipse_arc(47, 25, 6, 6, -np.pi/2, np.pi/2), line=dict(color=line_col, width=2), layer='below'))

    # Side Features
    if half_court in [None, 'left']:
        shapes += [
            dict(type="rect", x0=0, y0=17, x1=19, y1=33, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=0, y0=3, x1=14, y1=3, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=0, y0=47, x1=14, y1=47, line=dict(color=line_col, width=1), layer='below'),
            dict(type="path", path=ellipse_arc(5.25, 25, three_r, three_r, -1.18, 1.18), line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=4, y0=22, x1=4.2, y1=28, line=dict(color="#ec7607", width=2), fillcolor='#ec7607'),
            dict(type="circle", x0=4.5, y0=24.25, x1=6, y1=25.75, line=dict(color="#ec7607", width=2)),
        ]

    if half_court in [None, 'right']:
        shapes += [
            dict(type="rect", x0=75, y0=17, x1=94, y1=33, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=80, y0=3, x1=94, y1=3, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=80, y0=47, x1=94, y1=47, line=dict(color=line_col, width=1), layer='below'),
            dict(type="path", path=ellipse_arc(94-5.25, 25, three_r, three_r, np.pi - 1.18, np.pi + 1.18), line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=94-4.2, y0=22, x1=94-4, y1=28, line=dict(color="#ec7607", width=2), fillcolor='#ec7607'),
            dict(type="circle", x0=94-6, y0=24.25, x1=94-4.5, y1=25.75, line=dict(color="#ec7607", width=2)),
        ]

    # Assign coordinate references
    for s in shapes:
        s['xref'] = xref
        s['yref'] = yref
    
    return shapes

# --- Animation Functions --- #
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
    
    return fig.show(renderer="browser")

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

def animate_triple_comparison_old(sim_no_ist_traj, sim_ist_traj, real_def_traj, 
                              ist_sim_no_ist, ist_sim_ist, ist_real, 
                              off_traj, ball_traj, off_ids, use_images=True):
    """
    Generates a 1x3 side-by-side interactive comparison.
    Columns: [Real] | [Sim No IST] | [Sim With IST]
    """
    num_frames = off_traj.shape[0]
    
    # 1. Handle Optional Images
    encoded_images = {}
    if use_images:
        encoded_images = {int(pid): get_base64_image(int(pid)) for pid in off_ids}
    
    img_size = 8 
    text_y_offset = (img_size / 2) if use_images else 1.5
    # Use markers+text if images are off, otherwise just text (to avoid overlapping the image)
    off_mode = 'text' if use_images else 'markers+text'

    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Real Defense", "Simulated (No IST / Baseline)", "Simulated (With IST)"),
        horizontal_spacing=0.02
    )

    def get_labels(vals):
        labels = []
        halo = "text-shadow: 2px 2px 0 #fff, -2px -2px 0 #fff, 2px -2px 0 #fff, -2px 2px 0 #fff;"
        for v in vals:
            color = "red" if v > 1.00 else "black"
            labels.append(f"<span style='{halo} color: {color};'><b>{v:.2f}</b></span>")
        return labels

    def build_frame_images(f_idx):
        if not use_images:
            return []
        frame_imgs = []
        # Column mapping: 1=Real, 2=No IST, 3=With IST
        for col, xref, yref in [(1, "x", "y"), (2, "x2", "y2"), (3, "x3", "y3")]:
            for p_idx, pid in enumerate(off_ids):
                pid_int = int(pid) 
                if pid_int in encoded_images and encoded_images[pid_int]: 
                    frame_imgs.append(dict(
                        source=encoded_images[pid_int], 
                        xref=xref, yref=yref,
                        x=off_traj[f_idx, p_idx, 0],
                        y=off_traj[f_idx, p_idx, 1],
                        sizex=img_size, sizey=img_size,
                        xanchor="center", yanchor="middle",
                        layer="below",
                        sizing="stretch"
                    ))
        return frame_imgs

    fancy_font = dict(family="Arial Black, sans-serif", size=15, color="black")

    # --- Column 1: Real Defense ---
    fig.add_trace(go.Scatter(x=real_def_traj[0,:,0], y=real_def_traj[0,:,1], mode='markers', 
                             marker=dict(color='blue', size=12), name='Defense'), row=1, col=1)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_y_offset, mode=off_mode, 
                             text=get_labels(ist_real[0]), textposition="top center", 
                             marker=dict(color='red', size=10), textfont=fancy_font, name='Offense'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', 
                             marker=dict(color='orange', size=8, line=dict(width=2, color='black')), name='Ball'), row=1, col=1)

    # --- Column 2: Simulated (No IST / Baseline) ---
    fig.add_trace(go.Scatter(x=sim_no_ist_traj[0,:,0], y=sim_no_ist_traj[0,:,1], mode='markers', 
                             marker=dict(color='blue', size=12, opacity=0.4), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_y_offset, mode=off_mode, 
                             text=get_labels(ist_sim_no_ist[0]), textposition="top center", 
                             marker=dict(color='red', size=10), textfont=fancy_font, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', 
                             marker=dict(color='orange', size=8, line=dict(width=2, color='black')), showlegend=False), row=1, col=2)

    # --- Column 3: Simulated (With IST) ---
    fig.add_trace(go.Scatter(x=sim_ist_traj[0,:,0], y=sim_ist_traj[0,:,1], mode='markers', 
                             marker=dict(color='blue', size=12, opacity=0.7), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_y_offset, mode=off_mode, 
                             text=get_labels(ist_sim_ist[0]), textposition="top center", 
                             marker=dict(color='red', size=10), textfont=fancy_font, showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', 
                             marker=dict(color='orange', size=8, line=dict(width=2, color='black')), showlegend=False), row=1, col=3)

    # --- Build Animation Frames ---
    frames = []
    for f in range(num_frames):
        frames.append(go.Frame(
            data=[
                # Col 1
                go.Scatter(x=real_def_traj[f,:,0], y=real_def_traj[f,:,1]),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_y_offset, text=get_labels(ist_real[f])),
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]),
                # Col 2 (No IST)
                go.Scatter(x=sim_no_ist_traj[f,:,0], y=sim_no_ist_traj[f,:,1]),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_y_offset, text=get_labels(ist_sim_no_ist[f])),
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]),
                # Col 3 (With IST)
                go.Scatter(x=sim_ist_traj[f,:,0], y=sim_ist_traj[f,:,1]),
                go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_y_offset, text=get_labels(ist_sim_ist[f])),
                go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]])
            ],
            layout=go.Layout(images=build_frame_images(f)),
            name=str(f),
            traces=[0, 1, 2, 3, 4, 5, 6, 7, 8] 
        ))
    fig.frames = frames

    # Court shapes remain unchanged, using same xref/yref logic
    shapes_1 = draw_plotly_court(xref="x", yref="y")
    shapes_2 = draw_plotly_court(xref="x2", yref="y2")
    shapes_3 = draw_plotly_court(xref="x3", yref="y3")
    
    # Common axis settings to hide numbers and grid lines
    axis_clean = dict(range=[-5, 94], showgrid=False, zeroline=False, showticklabels=False)
    yaxis_clean = dict(range=[-5, 55], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False)

    fig.update_layout(
        title="IST Model Comparison: Real vs. Baseline vs. Enhanced",
        shapes=shapes_1 + shapes_2 + shapes_3,
        images=build_frame_images(0),
        template="plotly_white",
        width=1800, height=600,
        margin=dict(l=20, r=20, t=100, b=20),
        xaxis=axis_clean, yaxis=yaxis_clean,
        xaxis2=axis_clean, yaxis2=dict(range=[-5, 55], scaleanchor="x2", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        xaxis3=axis_clean, yaxis3=dict(range=[-5, 55], scaleanchor="x3", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[dict(
            type="buttons", x=0.0, y=-0.15, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0, x=0.1, y=-0.15, len=0.85, xanchor="left", yanchor="top",
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True))]) 
                   for k in range(num_frames)]
        )]
    )
    
    return fig.show(renderer="browser")

def animate_triple_comparison(sim_no_ist_traj, sim_ist_traj, real_def_traj, 
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
def animate_side_by_side_courts(sim_ist_traj, real_def_traj, ist_sim, ist_real, 
                                off_traj, ball_traj, off_ids, 
                                half_court='left', use_images=False):
    # --- 1. SHAPE SAFETY ---
    off_traj = np.array(off_traj).reshape(-1, 5, 2)
    real_def_traj = np.array(real_def_traj).reshape(-1, 5, 2)
    sim_ist_traj = np.array(sim_ist_traj).reshape(-1, 5, 2)
    ball_traj = np.array(ball_traj).reshape(-1, 2)
    ist_real = np.array(ist_real).reshape(-1, 5)
    ist_sim = np.array(ist_sim).reshape(-1, 5)
    
    num_frames = off_traj.shape[0]
    C_OFF, C_REAL, C_SIM, C_BALL = '#C8102E', '#888888', '#1D428A', '#ec7607'
    
    x_range = [-2, 49] if half_court == 'left' else [45, 96]
    y_range = [-2, 54]

    def get_dynamic_labels(vals):
        labels = []
        halo = "text-shadow: 1px 1px 0 #fff, -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff;"
        for v in vals:
            color_hex = C_OFF if v > 1.0 else 'black'
            labels.append(f"<span style='{halo} color: {color_hex};'><b>{v:.2f}</b></span>")
        return labels

    # --- THE "SQUISH" FIX ---
    # horizontal_spacing set to nearly zero (0.005)
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5], 
        subplot_titles=("Real NBA Defense", "Optimized JKO Defense"),
        horizontal_spacing=0.005 # Minimal gap
    )

    fancy_font = dict(family="Arial Black, sans-serif", size=11)
    text_offset = 2.8 

    # Traces
    for col, traj, ist_vals, name, def_color in zip([1, 2], [real_def_traj, sim_ist_traj], [ist_real, ist_sim], 
                                                     ['Real Defense', 'Optimized Defense'], [C_REAL, C_SIM]):
        fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1], mode='markers', marker=dict(color=C_OFF, size=13, line=dict(width=1, color='white')), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=[ball_traj[0,0]], y=[ball_traj[0,1]], mode='markers', marker=dict(color=C_BALL, size=9), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=traj[0,:,0], y=traj[0,:,1], mode='markers', marker=dict(color=def_color, size=13, line=dict(width=2, color=C_SIM if col==1 else 'white')), name=name), row=1, col=col)
        fig.add_trace(go.Scatter(x=off_traj[0,:,0], y=off_traj[0,:,1] + text_offset, mode='text', text=get_dynamic_labels(ist_vals[0]), textfont=fancy_font, showlegend=False), row=1, col=col)

    # Animation Frames
    frames = []
    for f in range(num_frames):
        frames.append(go.Frame(data=[
            go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]), go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]), 
            go.Scatter(x=real_def_traj[f,:,0], y=real_def_traj[f,:,1]), go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_offset, text=get_dynamic_labels(ist_real[f])), 
            go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1]), go.Scatter(x=[ball_traj[f,0]], y=[ball_traj[f,1]]), 
            go.Scatter(x=sim_ist_traj[f,:,0], y=sim_ist_traj[f,:,1]), go.Scatter(x=off_traj[f,:,0], y=off_traj[f,:,1] + text_offset, text=get_dynamic_labels(ist_sim[f])) 
        ], name=str(f), traces=[0, 1, 2, 3, 4, 5, 6, 7]))
    fig.frames = frames

    axis_config = dict(range=x_range, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, constrain='domain')
    yaxis_config = dict(range=y_range, scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, constrain='domain')

    fig.update_layout(
        title=dict(text="Side-by-Side Defense Comparison", x=0.5, xanchor='center', font=dict(size=22, family="Arial Black")),
        shapes=draw_plotly_court(xref="x", yref="y") + draw_plotly_court(xref="x2", yref="y2"),
        template="plotly_white", width=1400, height=750, 
        margin=dict(l=10, r=10, t=100, b=150), # Tighter left/right margins
        xaxis=axis_config, yaxis=yaxis_config,
        xaxis2=axis_config, yaxis2=dict(range=y_range, scaleanchor="x2", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, constrain='domain'),
        
        # Centered Legend
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, bgcolor="white", bordercolor="#1D428A", borderwidth=1),
        
        # Centered Play Buttons
        updatemenus=[dict(type="buttons", x=0.5, y=-0.16, xanchor="center", yanchor="top", direction="right",
                buttons=[dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                         dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
        
        # Smaller (60%) Centered Slider
        sliders=[dict(active=0, x=0.5, y=-0.26, len=0.6, xanchor="center", yanchor="top",
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True))]) for k in range(num_frames)])]
    )

    fig.update_layout(
        title=dict(text="Side-by-Side Defense Comparison", x=0.5, xanchor='center'),
        width=None, # Fixed width to prevent overflow
        height=650, 
        autosize=True,
        margin=dict(l=10, r=10, t=80, b=150),
        xaxis=dict(range=[-2, 49], constrain='domain'),
        xaxis2=dict(range=[-2, 49], constrain='domain'),
        # Legend & Controls centered and smaller
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05),
        sliders=[dict(len=0.6, x=0.5, xanchor="center", y=-0.25)]
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

# --- Poster Functions --- #
def generate_2x3_poster_figure(sim_ist_traj, real_def_traj, off_traj, ball_traj, 
                               ist_real, ist_sim, frame_indices, column_titles, 
                               filename="poster_2x3_clippers.png", 
                               show_tethers=False, vector_frames=20, movement_threshold=1.5):
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=300)
    fig.subplots_adjust(wspace=0.1, hspace=0.1) 
    
    # --- FIGURE BACKGROUND (20% BEC0C2) ---
    fig.patch.set_facecolor('#BEC0C2')
    fig.patch.set_alpha(0.2)
    
    # LA Clippers Colors
    C_OFF = '#C8102E'      # Ember Red
    C_REAL = '#888888'     # Grey
    C_SIM = '#1D428A'      # Naval Blue
    C_BALL = '#ec7607'     
    C_COURT = '#BEC0C2'

    max_f = len(off_traj) - 1

    for col in range(3):
        f = frame_indices[col]
        f_next = min(f + vector_frames, max_f)
        
        # ==========================================
        # ROW 0: REAL DEFENSE
        # ==========================================
        ax_real = axes[0, col]
        ax_real.set_facecolor('none') 
        draw_court_matplotlib(ax_real, color=C_COURT, lw=1.5, half_court='left')
        
        # Plot Entities - Removed white borders (edgecolors='none') for a cleaner modern look
        ax_real.scatter(off_traj[f, :, 0], off_traj[f, :, 1], c=C_OFF, s=250, edgecolors='white', linewidths=1.0, zorder=4)
        # Kept the blue border for Real Defense just to visually tie them to the Sim Model
        ax_real.scatter(real_def_traj[f, :, 0], real_def_traj[f, :, 1], c=C_REAL, s=250, edgecolors=C_SIM, linewidths=2.0, zorder=4)
        ax_real.scatter(ball_traj[f, 0], ball_traj[f, 1], c=C_BALL, s=150, edgecolors='none', zorder=7)
        
        for p in range(5):
            ox, oy = off_traj[f, p, 0], off_traj[f, p, 1]
            rx, ry = real_def_traj[f, p, 0], real_def_traj[f, p, 1]
            
            # Movement Vectors
            if f_next > f:
                ox_next, oy_next = off_traj[f_next, p, 0], off_traj[f_next, p, 1]
                rx_next, ry_next = real_def_traj[f_next, p, 0], real_def_traj[f_next, p, 1]
                
                if ((ox_next - ox)**2 + (oy_next - oy)**2)**0.5 > movement_threshold:
                    ax_real.annotate("", xy=(ox_next, oy_next), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color=C_OFF, lw=2, alpha=0.5), zorder=5)
                
                if ((rx_next - rx)**2 + (ry_next - ry)**2)**0.5 > movement_threshold:
                    ax_real.annotate("", xy=(rx_next, ry_next), xytext=(rx, ry), arrowprops=dict(arrowstyle="->", color=C_REAL, lw=2, alpha=0.5), zorder=5)

            # IST Text - Increased transparency (alpha=0.65)
            val_real = ist_real[f, p]
            text_color = C_OFF if val_real > 1.2 else C_SIM
            ax_real.text(ox, oy + 2.8, f"{val_real:.2f}", color=text_color, fontsize=11, fontweight='bold', ha='center', va='center', zorder=8,
                         bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=text_color, alpha=0.65, lw=1.5))
            
        ax_real.set_xlim(-2, 49)  
        ax_real.set_ylim(-2, 54) 
        ax_real.set_xticks([])
        ax_real.set_yticks([])
        ax_real.set_aspect('equal')
        ax_real.set_title(column_titles[col], fontsize=18, fontweight='bold', pad=15)
        
        if col == 0:
            ax_real.set_ylabel("Real Defense", fontsize=20, fontweight='bold', labelpad=15)

        # ==========================================
        # ROW 1: SIMULATED DEFENSE
        # ==========================================
        ax_sim = axes[1, col]
        ax_sim.set_facecolor('none')
        draw_court_matplotlib(ax_sim, color=C_COURT, lw=1.5, half_court='left')
        
        ax_sim.scatter(off_traj[f, :, 0], off_traj[f, :, 1], c=C_OFF, s=250, edgecolors='white', linewidths=1.0, zorder=4)
        ax_sim.scatter(sim_ist_traj[f, :, 0], sim_ist_traj[f, :, 1], c=C_SIM, s=250, alpha=0.9, edgecolors='white', linewidths=1.0, zorder=4)
        ax_sim.scatter(ball_traj[f, 0], ball_traj[f, 1], c=C_BALL, s=150, edgecolors='none', zorder=7)
        
        for p in range(5):
            rx, ry = real_def_traj[f, p, 0], real_def_traj[f, p, 1]
            sx, sy = sim_ist_traj[f, p, 0], sim_ist_traj[f, p, 1]
            ox, oy = off_traj[f, p, 0], off_traj[f, p, 1]
            
            if show_tethers:
                ax_sim.annotate("", xy=(sx, sy), xytext=(rx, ry), arrowprops=dict(arrowstyle="->", color=C_REAL, lw=1.5, ls="--", alpha=0.5, shrinkA=8, shrinkB=8), zorder=2)
            
            if f_next > f:
                ox_next, oy_next = off_traj[f_next, p, 0], off_traj[f_next, p, 1]
                sx_next, sy_next = sim_ist_traj[f_next, p, 0], sim_ist_traj[f_next, p, 1]
                
                if ((ox_next - ox)**2 + (oy_next - oy)**2)**0.5 > movement_threshold:
                    ax_sim.annotate("", xy=(ox_next, oy_next), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color=C_OFF, lw=2, alpha=0.5), zorder=5)
                
                if ((sx_next - sx)**2 + (sy_next - sy)**2)**0.5 > movement_threshold:
                    ax_sim.annotate("", xy=(sx_next, sy_next), xytext=(sx, sy), arrowprops=dict(arrowstyle="->", color=C_SIM, lw=2, alpha=0.5), zorder=5) 
            
            # IST Text - Increased transparency (alpha=0.65)
            val_sim = ist_sim[f, p]
            text_color = C_OFF if val_sim > 1.2 else C_SIM
            ax_sim.text(ox, oy + 2.8, f"{val_sim:.2f}", color=text_color, fontsize=11, fontweight='bold', ha='center', va='center', zorder=8,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=text_color, alpha=0.65, lw=1.5))
            
        ax_sim.set_xlim(-2, 49)  
        ax_sim.set_ylim(-2, 54)
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])
        ax_sim.set_aspect('equal')
        
        if col == 0:
            ax_sim.set_ylabel("Simulated Defense", fontsize=20, fontweight='bold', labelpad=15)

    # --- ADD SLEEK BORDERS TO EVERY PANEL ---
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(C_SIM)
            spine.set_linewidth(2)

    # ==========================================
    # COOLER, CUSTOM LEGEND
    # ==========================================
    # Build custom handles to guarantee side-by-side placement and consistent styling
    custom_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_OFF, markersize=14, label='Offense'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BALL, markersize=10, label='Ball'),
        # Placed side-by-side
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_REAL, markeredgecolor=C_SIM, markeredgewidth=2, markersize=14, label='Real Defense'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SIM, markersize=14, label='Enhanced Model (IST)'),
        Line2D([0], [0], color=C_REAL, lw=2, alpha=0.5, label=f'Player Movement (+{vector_frames} frames)')
    ]
    
    fig.legend(handles=custom_handles, 
               loc='lower center', 
               ncol=5,                # Forces them into a single horizontal row
               fontsize=15, 
               bbox_to_anchor=(0.5, 0.02), 
               frameon=True,          # Turn on the legend box
               facecolor='white',     # White background to pop off the grey figure
               edgecolor=C_SIM,       # Blue border
               fancybox=True,         # Rounded corners
               shadow=True,           # Cool drop shadow
               borderpad=0.8)         # Padding inside the legend box

    fig.suptitle('Minimizing Spatial Threat: Real vs. Simulated Defensive Positioning', fontsize=26, fontweight='black', y=0.95)

    # Save as PNG
    plt.savefig(filename, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
    print(f"Saved highly-res decorated 2x3 poster figure to {filename}")
    plt.show()

def generate_ist_line_chart(ist_real, ist_sim, shot_frame, filename="ist_threat_chart.png"):
    plt.style.use('default')
    
    # Clippers Brand Colors
    C_OFF = '#C8102E'      # Ember Red
    C_REAL = '#888888'     # Grey
    C_SIM = '#1D428A'      # Naval Blue
    C_COURT = '#BEC0C2'    # Background
    
    # 1. Setup Data
    end_idx = min(shot_frame + 10, len(ist_real))
    tot_real = np.sum(ist_real, axis=1)[:end_idx]
    tot_sim = np.sum(ist_sim, axis=1)[:end_idx]
    time_sec = np.arange(end_idx) / 25.0
    shot_time = shot_frame / 25.0

    # Calculate Totals for the Chart Label
    sum_real = tot_real.sum()
    sum_sim = tot_sim.sum()
    improvement = ((sum_real - sum_sim) / sum_real) * 100

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor(C_COURT); fig.patch.set_alpha(0.2); ax.set_facecolor('none')
    
    # 2. Plot Lines
    ax.plot(time_sec, tot_real, color=C_REAL, lw=5, label='Real Defense (Actual)', zorder=3)
    ax.plot(time_sec, tot_sim, color=C_SIM, lw=5, label='Simulated JKO Defense', zorder=3)
    
    # 3. Fill Recovery Gap
    ax.fill_between(time_sec, tot_sim, tot_real, where=(tot_real >= tot_sim), 
                    interpolate=True, color=C_OFF, alpha=0.2, label='Threat Prevented')
    
    # 4. Scaling & Headroom (Reduced headroom since snapshots are gone)
    max_y = max(np.max(tot_real), np.max(tot_sim))
    ax.set_ylim(0, max_y * 1.3)

    # 5. Titles & High-Threat Metadata
    ax.set_title("SPATIOTEMPORAL THREAT AUDIT", fontsize=28, fontweight='black', pad=35, color=C_SIM)
    
    # Metadata line showing High-Threat category and cumulative IST sums
    category_text = f"Category: High-Threat (>298.16) | Total Real IST: {sum_real:.1f} | Total Sim IST: {sum_sim:.1f}"
    ax.text(0.5, 1.02, category_text, transform=ax.transAxes, fontsize=14, 
            color='#444444', ha='center', fontweight='bold', va='bottom')

    # 6. Shot Indicator
    ax.axvline(x=shot_time, color='black', linestyle='--', lw=3, alpha=0.7, zorder=2)
    ax.text(shot_time + 0.05, max_y * 0.5, 'SHOT RELEASE', rotation=90, 
            color='black', fontsize=14, fontweight='black', va='center')
    
    # 7. Impact Box (Top Right)
    ax.text(0.98, 0.95, f"THREAT REDUCTION: {improvement:.1f}%", 
            transform=ax.transAxes, fontsize=16, fontweight='black', 
            color='white', ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.6", facecolor=C_OFF, edgecolor='none'))

    # Final Polish
    ax.set_xlabel('Time (Seconds)', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Aggregate IST (Team Total)', fontsize=18, fontweight='bold', labelpad=15)
    ax.legend(loc='lower right', fontsize=14, frameon=True, facecolor='white', edgecolor=C_SIM, shadow=True)
    
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(C_SIM); ax.spines[s].set_linewidth(3)
        
    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor())
    plt.show()

def generate_efficiency_bar_chart(filename="efficiency_by_play_type.svg"):
    """
    Generates a highly-styled, Clippers-themed horizontal bar chart 
    showing Efficiency Gain by Play Type for a Google Slides poster.
    """
    plt.style.use('default')
    
    # --- 1. The Data (Top 6 High-Volume Plays) ---
    data = {
        'Play Type': [
            'Pick & Roll (Ball Handler)', 
            'Drive & Kick', 
            'Off Screen', 
            'Isolation', 
            'Catch & Shoot', 
            'Transition'
        ],
        'Efficiency Gain (%)': [43.1, 38.6, 38.2, 37.7, 35.3, 13.6]
    }
    df = pd.DataFrame(data)
    
    # Sort so the highest is at the top of the chart
    df = df.sort_values(by='Efficiency Gain (%)', ascending=True)

    # --- 2. Clippers Brand Colors ---
    C_BASE = '#1D428A'  # Naval Blue (Standard)
    C_HIGH = '#C8102E'  # Ember Red (Highlight the Best)
    C_LOW = '#BEC0C2'   # Clippers Silver (Highlight the Worst/Baseline)
    
    # Assign colors: Red for highest, Silver for lowest, Blue for the rest
    colors = [C_LOW if val < 20 else (C_HIGH if val > 40 else C_BASE) for val in df['Efficiency Gain (%)']]

    # --- 3. Setup the Figure ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # Create horizontal bars
    bars = ax.barh(df['Play Type'], df['Efficiency Gain (%)'], color=colors, height=0.6)

    # --- 4. Add Data Labels to the Bars ---
    for bar in bars:
        width = bar.get_width()
        # Place the text slightly inside the end of the bar (or outside if it's too small)
        if width > 20:
            ax.text(width - 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                    ha='right', va='center', color='white', fontsize=14, fontweight='bold')
        else:
            ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                    ha='left', va='center', color=C_BASE, fontsize=14, fontweight='bold')

    # --- 5. Formatting and Styling ---
    ax.set_title('Threat Reduction by Play Type', fontsize=18, fontweight='black', pad=20, loc='left')
    ax.set_xlabel('Efficiency Gain (%)', fontsize=14, fontweight='bold', labelpad=10)
    
    # Clean up axes and spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Hide the left spine for a floating look
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Style the ticks
    ax.tick_params(axis='y', which='major', labelsize=14, length=0, pad=10) # Hide Y tick marks, just keep text
    ax.tick_params(axis='x', which='major', labelsize=12, width=1.5)
    
    # Add a subtle vertical grid to help read the x-axis
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Set X-axis limit slightly higher than max value for padding
    ax.set_xlim(0, 50)

    # --- 6. Save and Output ---
    plt.savefig(filename, format=filename.split('.')[-1], bbox_inches='tight', transparent=True)
    print(f"Saved Horizontal Bar Chart to {filename}")
    plt.show()

def plot_efficiency_density(summary_df, filename="efficiency_density.png"):
    plt.style.use('default')
    
    # Brand Colors
    C_SIM = '#1D428A' # Naval Blue
    C_OFF = '#C8102E' # Ember Red
    C_BG  = '#BEC0C2' # Silver
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor(C_BG); fig.patch.set_alpha(0.2)
    ax.set_facecolor('none')
    
    # Create the Density (KDE) Plot
    sns.kdeplot(summary_df['Efficiency Gain (%)'], 
                fill=True, color=C_SIM, lw=3, alpha=0.4, ax=ax)
    
    # Add a vertical line for the Median
    median_val = summary_df['Efficiency Gain (%)'].median()
    ax.axvline(median_val, color=C_OFF, ls='--', lw=2, label=f'Median Gain: {median_val:.1f}%')
    
    # Formatting
    ax.set_title("DISTRIBUTION OF DEFENSIVE LIFT", fontsize=20, fontweight='black', pad=20, color=C_SIM)
    ax.set_xlabel("Efficiency Gain (%)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Frequency of Plays", fontsize=14, fontweight='bold')
    
    # Clean spines
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(C_SIM); ax.spines['bottom'].set_color(C_SIM)
    
    ax.legend(fontsize=12, frameon=True, facecolor='white', shadow=True)
    
    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor())
    plt.show()

def plot_ist_population_shift(summary_df, filename="ist_distribution_shift_final.png"):
    plt.style.use('default')
    C_REAL, C_SIM, C_BG, C_OFF = '#888888', '#1D428A', '#BEC0C2', '#C8102E'
    
    # 1. Stats from your 106-play audit
    real_med = summary_df['Total Real IST'].median()
    sim_med = summary_df['Total Sim IST'].median()
    recovery_val = 36.7
    reliability_rate = 92.9  

    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    fig.patch.set_facecolor(C_BG); fig.patch.set_alpha(0.15); ax.set_facecolor('none')
    
    # 2. Plot Populations
    sns.kdeplot(summary_df['Total Real IST'], fill=True, color=C_REAL, 
                lw=6, label='Actual NBA Defense', alpha=0.3, ax=ax)
    sns.kdeplot(summary_df['Our Simulated Defense'] if 'Our Simulated Defense' in summary_df else summary_df['Total Sim IST'], 
                fill=True, color=C_SIM, lw=6, label='Our Simulated Defense', alpha=0.5, ax=ax)
    
    # 3. Median Reference Lines
    ax.axvline(real_med, color=C_REAL, ls=':', lw=4, alpha=0.8)
    ax.axvline(sim_med, color=C_SIM, ls=':', lw=4, alpha=0.8)

    # 4. THE RECOVERY GAP ARROW (Lowered slightly to create more "breathing room" at top)
    y_limit = ax.get_ylim()[1]
    arrow_y = y_limit * 0.40 
    ax.annotate('', xy=(sim_med, arrow_y), xytext=(real_med, arrow_y),
                arrowprops=dict(arrowstyle='<->', color=C_OFF, lw=5))
    ax.text((real_med + sim_med)/2, arrow_y + (y_limit * 0.03), "RECOVERY GAP", 
            color=C_OFF, fontweight='black', fontsize=16, ha='center')

    # 5. CENTERED TITLE
    ax.set_title("How Good Is Our Model?", fontsize=42, fontweight='black', 
                 pad=60, color=C_SIM, loc='center')
    ax.set_xlabel("Team IST Per Play", fontsize=24, fontweight='bold', labelpad=20)
    ax.set_ylabel("Frequency of Outcomes", fontsize=24, fontweight='bold', labelpad=20)
    
    # 6. SPACED DASHBOARD (Top Right)
    # Placing the Legend at the absolute top right
    leg = ax.legend(fontsize=18, loc='upper right', frameon=True, shadow=True, 
                    facecolor='white', edgecolor=C_SIM, borderpad=1.2)

    # Placing the Hero Box significantly lower (0.65 instead of 0.75) to prevent overlap
    stats_text = (f"FRAME WIN RATE: {reliability_rate}%\n"
                  f"MEDIAN RECOVERY: {recovery_val}%")
    
    ax.text(0.97, 0.65, stats_text, transform=ax.transAxes,
            fontsize=20, fontweight='black', color='white', 
            ha='right', va='top',
            bbox=dict(boxstyle="round,pad=1.0", facecolor=C_OFF, edgecolor='none'))

    # 7. Final Polish
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(C_SIM); ax.spines[s].set_linewidth(4)
    
    ax.tick_params(axis='both', labelsize=18, width=3, length=10)
    
    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor())
    plt.show()