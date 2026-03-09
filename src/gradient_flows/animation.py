import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch
import base64
from PIL import Image, ImageOps, ImageDraw
import io
import os


# ==== Drawing Functions ===== # 
def draw_court_matplotlib(ax, color='#777777', lw=1.5, half_court=None):
    """Matplotlib version of the high-quality Plotly court with half-court support."""
    # Court Perimeter
    if half_court == 'left':
        ax.add_patch(Rectangle((0, 0), 47, 50, color=color, zorder=0, fill=False, lw=lw*2))
    elif half_court == 'right':
        ax.add_patch(Rectangle((47, 0), 47, 50, color=color, zorder=0, fill=False, lw=lw*2))
    else:
        ax.add_patch(Rectangle((0, 0), 94, 50, color=color, zorder=0, fill=False, lw=lw*2))
    
    # Midcourt line and Circle
    ax.plot([47, 47], [0, 50], color=color, lw=lw)
    if half_court == 'left':
        ax.add_patch(Arc((47, 25), 12, 12, theta1=90, theta2=270, color=color, lw=lw))
    elif half_court == 'right':
        ax.add_patch(Arc((47, 25), 12, 12, theta1=-90, theta2=90, color=color, lw=lw))
    else:
        ax.add_patch(Circle((47, 25), 6, color=color, fill=False, lw=lw))

    # Left Side Features
    if half_court in [None, 'left']:
        ax.add_patch(Rectangle((0, 17), 19, 16, color=color, fill=False, lw=lw))
        ax.add_patch(Arc((5.25, 25), 47.5, 47.5, theta1=-67.5, theta2=67.5, color=color, lw=lw))
        ax.plot([0, 14], [3, 3], color=color, lw=lw)
        ax.plot([0, 14], [47, 47], color=color, lw=lw)
        ax.add_patch(Rectangle((4, 22), 0.2, 6, color="#ec7607", lw=2))
        ax.add_patch(Circle((5.25, 25), 0.75, color="#ec7607", fill=False, lw=2))

    # Right Side Features
    if half_court in [None, 'right']:
        ax.add_patch(Rectangle((75, 17), 19, 16, color=color, fill=False, lw=lw))
        ax.add_patch(Arc((94-5.25, 25), 47.5, 47.5, theta1=112.5, theta2=247.5, color=color, lw=lw))
        ax.plot([80, 94], [3, 3], color=color, lw=lw)
        ax.plot([80, 94], [47, 47], color=color, lw=lw)
        ax.add_patch(Rectangle((90, 22), 0.2, 6, color="#ec7607", lw=2))
        ax.add_patch(Circle((94-5.25, 25), 0.75, color="#ec7607", fill=False, lw=2))


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
        title="Interactive IST Geometry Sandbox",
        shapes=draw_plotly_court(xref="x", yref="y"), 
        images=build_frame_images(0),
        template="plotly_white",
        width=1400, height=800, # Increased height to fit bottom menus
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

# --- Gif Functions --- # 
def save_simulation_gif(sim_traj, real_traj, ball_traj, offenders_traj, ist_sim, ist_real, filename='sim_comparison.gif'):
    """
    Generates a side-by-side GIF comparing Real vs Sim defense, 
    with their respective IST (pressure) values labeled on each offender.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Defense Comparison: Pressure Analysis (IST)', fontsize=16)

    # --- Setup Courts ---
    for ax in [ax1, ax2]:
        draw_court_matplotlib(ax) # Assuming this helper exists in your notebook
    
    ax1.set_title('Real Defense (Actual Pressure)', fontsize=12)
    ax2.set_title('Simulated JKO Defense (Model Pressure)', fontsize=12)

    # --- Initialize Scatters ---
    # Real Plot
    r_off_sc = ax1.scatter([], [], c='red', s=120, zorder=3)
    r_def_sc = ax1.scatter([], [], c='blue', s=120, zorder=3)
    r_ball_sc = ax1.scatter([], [], c='orange', s=80, zorder=4)
    
    # Sim Plot
    s_off_sc = ax2.scatter([], [], c='red', s=120, zorder=3)
    s_def_sc = ax2.scatter([], [], c='blue', alpha=0.6, s=120, zorder=3)
    s_ball_sc = ax2.scatter([], [], c='orange', s=80, zorder=4)

    # --- Initialize IST Labels (5 for each side) ---
    r_ist_labels = [ax1.text(0, 0, '', color='black', fontsize=10, fontweight='bold', ha='center') for _ in range(5)]
    s_ist_labels = [ax2.text(0, 0, '', color='black', fontsize=10, fontweight='bold', ha='center') for _ in range(5)]

    def update(frame):
        # 1. Update Ball/Player Positions
        r_off_sc.set_offsets(offenders_traj[frame])
        s_off_sc.set_offsets(offenders_traj[frame])
        
        r_def_sc.set_offsets(real_traj[frame])
        s_def_sc.set_offsets(sim_traj[frame])
        
        r_ball_sc.set_offsets(ball_traj[frame])
        s_ball_sc.set_offsets(ball_traj[frame])

        # 2. Update IST Text and Colors
        for i in range(5):
            pos = offenders_traj[frame, i]
            
            # Real Side
            val_real = ist_real[frame, i]
            r_ist_labels[i].set_position((pos[0], pos[1] + 1.8))
            r_ist_labels[i].set_text(f'{val_real:.2f}')
            # Turn text red if high threat
            r_ist_labels[i].set_color('red' if val_real > 1.2 else 'black') 
            
            # Sim Side
            val_sim = ist_sim[frame, i]
            s_ist_labels[i].set_position((pos[0], pos[1] + 1.8))
            s_ist_labels[i].set_text(f'{val_sim:.2f}')
            s_ist_labels[i].set_color('red' if val_sim > 1.2 else 'black')

        return (r_off_sc, r_def_sc, r_ball_sc, s_off_sc, s_def_sc, s_ball_sc, 
                *r_ist_labels, *s_ist_labels)

    anim = FuncAnimation(fig, update, frames=len(sim_traj), interval=40, blit=True)
    anim.save(filename, writer='pillow')
    plt.close()
    print(f"Comparison GIF saved to {filename}")

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

def generate_2x3_poster_figure(sim_ist_traj, real_def_traj, off_traj, ball_traj, 
                               ist_real, ist_sim, frame_indices, column_titles, filename="poster_2x3_figure.svg"):
    """
    Generates a 2x3 static figure for a poster (Left Half Court).
    Row 1: Real Defense (with Real IST)
    Row 2: Simulated Defense (with Sim IST)
    
    Args:
        sim_ist_traj, real_def_traj, off_traj, ball_traj: Coordinate arrays.
        ist_real, ist_sim: The IST threat arrays for both models.
        frame_indices: List of 3 frame integers (e.g., [0, 20, 40]).
        column_titles: List of 3 string titles for the columns.
        filename: Output filename (.svg or .pdf).
    """
    plt.style.use('default')
    
    # Create 2x3 grid.
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=300)
    fig.subplots_adjust(wspace=0.1, hspace=0.1) 
    
    # Colors matching your theme
    C_OFF = '#d62728'   # Red
    C_REAL = '#1f77b4'  # Blue
    C_SIM = '#2ca02c'   # Green
    C_BALL = '#ff7f0e'  # Orange

    for col in range(3):
        f = frame_indices[col]
        
        # ==========================================
        # ROW 0: REAL DEFENSE
        # ==========================================
        ax_real = axes[0, col]
        draw_court_matplotlib(ax_real, color='gray', lw=1.5, half_court='left')
        
        # Plot Entities
        ax_real.scatter(off_traj[f, :, 0], off_traj[f, :, 1], c=C_OFF, s=250, zorder=4, label='Offense' if col==0 else "")
        ax_real.scatter(real_def_traj[f, :, 0], real_def_traj[f, :, 1], c=C_REAL, s=250, edgecolors='black', linewidths=1.5, zorder=4, label='Real Defense' if col==0 else "")
        ax_real.scatter(ball_traj[f, 0], ball_traj[f, 1], c=C_BALL, s=150, edgecolors='black', zorder=5, label='Ball' if col==0 else "")
        
        # Add Real IST Text
        for p in range(5):
            ox, oy = off_traj[f, p, 0], off_traj[f, p, 1]
            val_real = ist_real[f, p]
            text_color = 'darkred' if val_real > 1.2 else 'black'
            ax_real.text(ox, oy + 2.0, f"{val_real:.2f}", color=text_color, fontsize=13, fontweight='bold', ha='center', zorder=6)
            
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
        draw_court_matplotlib(ax_sim, color='gray', lw=1.5, half_court='left')
        
        # Plot Entities
        ax_sim.scatter(off_traj[f, :, 0], off_traj[f, :, 1], c=C_OFF, s=250, zorder=4)
        ax_sim.scatter(sim_ist_traj[f, :, 0], sim_ist_traj[f, :, 1], c=C_SIM, s=250, edgecolors='black', linewidths=1.5, zorder=4, label='Enhanced Model (IST)' if col==0 else "")
        ax_sim.scatter(ball_traj[f, 0], ball_traj[f, 1], c=C_BALL, s=150, edgecolors='black', zorder=5)
        
        # Add Tethers & Sim IST Text
        for p in range(5):
            rx, ry = real_def_traj[f, p, 0], real_def_traj[f, p, 1]
            sx, sy = sim_ist_traj[f, p, 0], sim_ist_traj[f, p, 1]
            ox, oy = off_traj[f, p, 0], off_traj[f, p, 1]
            
            # Draw a faint tether showing where the real defender was
            ax_sim.annotate("", xy=(sx, sy), xytext=(rx, ry),
                            arrowprops=dict(arrowstyle="->", color=C_REAL, lw=2, alpha=0.4, shrinkA=8, shrinkB=8),
                            zorder=2)
            
            val_sim = ist_sim[f, p]
            text_color = 'darkred' if val_sim > 1.2 else 'black'
            ax_sim.text(ox, oy + 2.0, f"{val_sim:.2f}", color=text_color, fontsize=13, fontweight='bold', ha='center', zorder=6)
            
        ax_sim.set_xlim(-2, 49)  
        ax_sim.set_ylim(-2, 54)
        ax_sim.set_xticks([])
        ax_sim.set_yticks([])
        ax_sim.set_aspect('equal')
        
        if col == 0:
            ax_sim.set_ylabel("Simulated Defense", fontsize=20, fontweight='bold', labelpad=15)

    # Add Global Legend
    handles_real, labels_real = axes[0, 0].get_legend_handles_labels()
    handles_sim, labels_sim = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles_real + handles_sim, labels_real + labels_sim, 
               loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, 0.05), frameon=False)
    
    # Global Title
    fig.suptitle('Dynamic Adaptation of Topological Defensive Identity', fontsize=26, fontweight='black', y=0.95)

    plt.savefig(filename, format=filename.split('.')[-1], bbox_inches='tight', transparent=True)
    print(f"Saved highly-res 2x3 poster figure to {filename}")
    
    plt.show()