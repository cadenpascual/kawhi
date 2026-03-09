import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter  # <-- The smoothing tool

def plot_frame(frame, team_colors=None):
    """
    Plot a single NBA frame with players split by team and the ball.
    
    Parameters
    ----------
    frame : dict
        Single frame dictionary with keys: 'ball', 'players', 'frame_id', etc.
    team_colors : dict, optional
        Mapping from teamid to color. Example: {1610612739: 'blue', 1610612744: 'red'}
    """
    
    if team_colors is None:
        team_colors = {}
    
    plt.figure(figsize=(15, 7))
    
    # Draw court boundaries (simplified rectangle)
    plt.plot([0, 50], [0, 0], color='black')   # baseline
    plt.plot([0, 50], [94, 94], color='black') # opposite baseline
    plt.plot([0, 0], [0, 94], color='black')   # sideline
    plt.plot([50, 50], [0, 94], color='black') # opposite sideline
    
    # Draw the ball
    ball = frame["ball"]
    plt.scatter(ball["x"], ball["y"], c='orange', s=200, marker='o', label='Ball', edgecolors='black')
    
    # Plot players
    for player in frame["players"]:
        x, y = player["x"], player["y"]
        teamid = player["teamid"]
        color = team_colors.get(teamid, 'green')  # default green if teamid not in dict
        plt.scatter(x, y, c=color, s=150, label=f'Team {teamid}' if f'Team {teamid}' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(x+0.5, y+0.5, str(player["playerid"]), fontsize=9, color=color)
    
    plt.xlim(0, 50)
    plt.ylim(0, 94)
    plt.xlabel("Court X (ft)")
    plt.ylabel("Court Y (ft)")

    # convert game clock to MM:SS format
    minutes = int(frame['game_clock'] // 60)
    seconds = int(frame['game_clock'] % 60)

    # get shot clock if available
    shot_clock = (frame['shot_clock'])
    shot_str = f"{shot_clock:.1f}s" if shot_clock is not None else "-"

    plt.title(f"Frame {frame['frame_id']} - Game Clock: {minutes:02d}:{seconds:02d} | Shot Clock: {shot_str}")
    plt.legend()
    plt.show()


def draw_half_court(ax=None, color="black", lw=2, outer_lines=False, zorder=2):
    if ax is None:
        ax = plt.gca()

    Y_SHIFT = 47.5  # shift old court coords up so baseline becomes y=0

    # Hoop & backboard
    hoop = Circle((0, 0 + Y_SHIFT), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5 + Y_SHIFT), 60, -1, linewidth=lw, color=color)

    # Paint
    outer_box = Rectangle((-80, -47.5 + Y_SHIFT), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5 + Y_SHIFT), 120, 190, linewidth=lw, color=color, fill=False)

    # Free throw arcs
    top_free_throw = Arc((0, 142.5 + Y_SHIFT), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color)
    bottom_free_throw = Arc((0, 142.5 + Y_SHIFT), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle="dashed")

    # Restricted arc
    restricted = Arc((0, 0 + Y_SHIFT), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Corner 3 lines (14ft = 168 inches tall)
    corner_three_a = Rectangle((-220, -47.5 + Y_SHIFT), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5 + Y_SHIFT), 0, 140, linewidth=lw, color=color)

    # 3pt arc (same as your original; works visually)
    three_arc = Arc((0, 0 + Y_SHIFT), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # Center court arcs
    center_outer_arc = Arc((0, 422.5 + Y_SHIFT), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5 + Y_SHIFT), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

    court_elements = [
        hoop, backboard, outer_box, inner_box, top_free_throw, bottom_free_throw,
        restricted, corner_three_a, corner_three_b, three_arc,
        center_outer_arc, center_inner_arc
    ]

    if outer_lines:
        outer = Rectangle((-250, 0), 500, 470, linewidth=lw, color=color, fill=False)  # baseline now y=0
        court_elements.append(outer)

    for e in court_elements:
        e.set_zorder(zorder)
        ax.add_patch(e)

    return ax

# --- 1. DEFINE THE CUSTOM CLIPPERS COLORMAP ---
C_COLD = '#1D428A'  # Naval Blue
C_HOT = '#C8102E'   # Ember Red
C_COURT = '#BEC0C2' # Clippers Silver / Grey for court lines

clippers_cmap = mcolors.LinearSegmentedColormap.from_list("clippers_diverging", [C_COLD, "white", C_HOT])
clippers_sequential = mcolors.LinearSegmentedColormap.from_list("clippers_sequential", ["white", C_HOT])

# --- 2. THE HIGH-ACCURACY COURT DRAWING FUNCTION ---
def draw_half_court_ft(ax=None, color="black", lw=2, outer_lines=False, zorder=3,
                       baseline_y=-4.75):
    """
    Draw half court in FEET for shot-chart coords:
      rim at (0,0), baseline at y = baseline_y (~ -4.75 ft).
    """
    if ax is None:
        ax = plt.gca()

    # Anchors
    hoop_x, hoop_y = 0.0, 0.0
    half_court_y = baseline_y + 47.0  # half-court line

    # Geometry (feet)
    hoop_r = 0.75            # 9 in
    backboard_w = 6.0
    backboard_h = 0.0833     # thin
    # Backboard plane is ~4 ft from baseline; in this coord rim is at y=0,
    # so baseline is negative. This places board near y ~ -0.75 (looks right).
    backboard_y = baseline_y + 4.0 - backboard_h

    lane_w_outer = 16.0
    lane_w_inner = 12.0
    lane_len = 19.0
    ft_y = baseline_y + lane_len
    ft_circle_r = 6.0

    restricted_r = 4.0

    corner_3_x = 22.0
    corner_3_top_y = baseline_y + 14.0

    three_r = 23.75

    # Patches
    hoop = Circle((hoop_x, hoop_y), radius=hoop_r, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-backboard_w/2, backboard_y), backboard_w, backboard_h,
                          linewidth=lw, color=color)

    outer_box = Rectangle((-lane_w_outer/2, baseline_y), lane_w_outer, lane_len,
                          linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-lane_w_inner/2, baseline_y), lane_w_inner, lane_len,
                          linewidth=lw, color=color, fill=False)

    top_ft = Arc((0, ft_y), 2*ft_circle_r, 2*ft_circle_r, theta1=0, theta2=180,
                 linewidth=lw, color=color)
    bottom_ft = Arc((0, ft_y), 2*ft_circle_r, 2*ft_circle_r, theta1=180, theta2=0,
                    linewidth=lw, color=color, linestyle="dashed")

    restricted = Arc((0, 0), 2*restricted_r, 2*restricted_r, theta1=0, theta2=180,
                     linewidth=lw, color=color)

    y_intersect = np.sqrt(max(three_r**2 - corner_3_x**2, 0))
    theta = np.degrees(np.arctan2(y_intersect, corner_3_x))

    three_arc = Arc((0, 0), 2*three_r, 2*three_r,
                theta1=theta, theta2=180-theta,
                linewidth=lw, color=color)
    three_arc.set_clip_on(False)

    for p in [hoop, backboard, outer_box, inner_box, top_ft, bottom_ft, restricted, three_arc]:
        p.set_zorder(zorder)
        ax.add_patch(p)

    # Corner 3 lines (Line2D)
    ax.plot([-corner_3_x, -corner_3_x], [baseline_y, corner_3_top_y],
            linewidth=lw, color=color, zorder=zorder)
    ax.plot([ corner_3_x,  corner_3_x], [baseline_y, corner_3_top_y],
            linewidth=lw, color=color, zorder=zorder)

    if outer_lines:
        outer = Rectangle((-25, baseline_y), 50, 47.0, linewidth=lw, color=color, fill=False)
        outer.set_zorder(zorder)
        ax.add_patch(outer)

    return ax


# --- 3. THE PLOTTING FUNCTIONS ---

def plot_player_map_on_court(grid, extent=(-25, 25, -5, 42), title=None, alpha=0.85,
                             xlim=(-25, 25), ylim=(-5, 42), transpose=True, 
                             vmin=0.0, vmax=None, midpoint=1.0):
    """
    Plots an absolute xPPS map using a DIVERGING colormap.
    Values below 'midpoint' appear Blue; values above appear Red.
    """
    safe_grid = np.nan_to_num(grid, nan=0.0)
    smoothed_grid = gaussian_filter(safe_grid, sigma=1.2)
    grid_to_plot = smoothed_grid.T if transpose else smoothed_grid

    # If vmax isn't provided, calculate it from the grid
    if vmax is None:
        vmax = np.max(grid_to_plot)
    
    # Ensure midpoint is within the range for the Norm to work
    actual_vmax = max(vmax, midpoint + 0.01)
    
    # TwoSlopeNorm allows us to fix the 'White' color exactly at the midpoint (1.0)
    # even if the range (0.0 to 1.5) is not centered.
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=actual_vmax)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Plot heatmap with Diverging Colormap
    im = ax.imshow(
        grid_to_plot,
        origin="lower",
        extent=extent,
        aspect="equal",
        alpha=alpha,
        zorder=1,
        cmap=clippers_cmap,
        norm=norm
    )

    # Draw high-accuracy court lines (using dark grey for neutral contrast)
    draw_half_court_ft(ax=ax, color='#333333', lw=2, outer_lines=True, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    if title: 
        ax.set_title(title, fontsize=16, fontweight='black', pad=15)

    # Add Colorbar to show the scale
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Shot Quality (xPPS)", fontweight='bold', labelpad=10)
        
    plt.show()

def plot_relative_xpps_map(player_grid, league_grid, extent=(-25, 25, -5, 42), 
                           xlim=(-25, 25), ylim=(-5, 42),
                           title=None, alpha=0.85, transpose=True, max_abs_val=None):
    """
    Plots a relative difference map (Player vs League) ON the court.
    """
    relative_grid = player_grid - league_grid
    grid_to_plot = relative_grid.T if transpose else relative_grid

    if max_abs_val is None:
        max_abs_val = np.max(np.abs(grid_to_plot))

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    
    # Plot the relative heatmap
    im = ax.imshow(
        grid_to_plot,
        origin="lower",
        extent=extent,
        aspect="equal",
        alpha=alpha,
        zorder=1,
        cmap=clippers_cmap, 
        vmin=-max_abs_val,  
        vmax=max_abs_val    
    )

    # Draw the court lines using a dark grey/charcoal to contrast with red/blue
    draw_half_court_ft(ax=ax, color='#333333', lw=2, outer_lines=True, zorder=3)

    # Lock to exact specified grid
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title: 
        ax.set_title(title, fontsize=16, fontweight='black', pad=15)
        
    # Styled colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor('#333333')
    cbar.outline.set_linewidth(1.5)
    cbar.set_label("xPPS vs League Average", rotation=270, labelpad=15, fontweight='bold')
        
    plt.show()