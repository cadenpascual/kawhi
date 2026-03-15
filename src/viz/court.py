import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter  # <-- The smoothing tool
from nba_api.stats.static import players

# --- 1. DEFINE THE COLORMAP ---
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
                             vmin=0.0, vmax=None, midpoint=1.0,
                             ax=None, figsize=(4,3.5),
                             dpi=150, title_size=14):
    
    safe_grid = np.nan_to_num(grid, nan=0.0)
    smoothed_grid = gaussian_filter(safe_grid, sigma=1.2)
    grid_to_plot = smoothed_grid.T if transpose else smoothed_grid

    if vmax is None:
        vmax = np.max(grid_to_plot)
    actual_vmax = max(vmax, midpoint + 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=actual_vmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure 
    
    im = ax.imshow(
        grid_to_plot, origin="lower", extent=extent, aspect="equal",
        alpha=alpha, zorder=1, cmap=clippers_cmap, norm=norm
    )

    draw_half_court_ft(ax=ax, color='#333333', lw=2, outer_lines=True, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    if title: 
        ax.set_title(title, fontsize=title_size, fontweight='black', pad=15)

    # Attach the colorbar specifically to this axis
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Shot Quality (xPPS)", fontweight='bold', labelpad=10)

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

def plot_players_by_name(player_names, maps, pid2row, fixed_vmax=None, baseline=1.0, 
                         size_per_plot=(4, 3.5), dpi=150):
    player_data = []
    
    for name in player_names:
        search_result = players.find_players_by_full_name(name)
        if not search_result: continue
            
        player_id = search_result[0]['id']
        actual_name = search_result[0]['full_name']
        
        if player_id not in pid2row: continue
            
        grid = maps['quality'][pid2row[player_id]]
        player_data.append({"name": actual_name, "grid": grid})
        
    if not player_data:
        print("No valid players found to plot. Exiting.")
        return

    n_players = len(player_data)
    
    if fixed_vmax is None:
        global_vmax = max([np.max(p["grid"]) for p in player_data])
    else:
        global_vmax = fixed_vmax

    # ---> NEW: Calculate total image size and dynamic font size
    total_width = size_per_plot[0] * n_players
    total_height = size_per_plot[1]
    
    # Mathematical scaling: ~3.5x the width in inches, minimum 9pt font
    dynamic_font = max(9, int(size_per_plot[0] * 3.5))

    # ---> NEW: Create a 1-Row, N-Column grid of subplots
    fig, axes = plt.subplots(1, n_players, figsize=(total_width, total_height), dpi=dpi)
    
    # Ensure axes is iterable even if we only plotted 1 player
    if n_players == 1:
        axes = [axes]

    # Draw each player onto their designated subplot in the grid
    for ax, p in zip(axes, player_data):
        plot_player_map_on_court(
            p["grid"], 
            title=f"{p['name']}", 
            vmax=global_vmax, 
            midpoint=baseline,
            ax=ax,                   # Tells it to draw on this specific subplot
            title_size=dynamic_font  # Passes the scaled font size down
        )
        
    # Neatly spaces the charts so colorbars don't overlap
    plt.tight_layout()
    plt.show()