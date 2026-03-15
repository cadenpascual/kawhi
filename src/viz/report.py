import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_release_frame_exemplar(row: pd.Series, fps: int = 25, save_path: str = None):
    """
    A publication-ready plot of the ball's z-trajectory measured in seconds.
    Features CAD-style dimension arrows and physics-based terminology.
    """
    # 1. Extract variables and convert to SECONDS
    z_traj = row['ball_z_traj']
    time_sec = np.arange(len(z_traj)) / fps  
    
    true_release_idx = int(row['local_release_idx'])
    api_delay_idx = int(row['local_pbp_idx'])
    
    true_release_sec = true_release_idx / fps
    api_delay_sec = api_delay_idx / fps
    lag_sec = api_delay_sec - true_release_sec
    
    max_height = np.max(z_traj)
    
    # 2. Setup Figure & Brand Colors
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    C_LINE = '#1D428A' # Naval Blue (Ball)
    C_TRUE = '#059669' # Emerald Green (Algorithm)
    C_API  = '#C8102E' # Ember Red (Error)
    
    # 3. Plot the ball's trajectory
    ax.plot(time_sec, z_traj, color=C_LINE, lw=4, zorder=2)
    
    # 4. Plot vertical dashed lines
    ax.axvline(x=true_release_sec, color=C_TRUE, linestyle='--', lw=2.5, zorder=3)
    ax.axvline(x=api_delay_sec, color=C_API, linestyle='--', lw=2.5, zorder=3)
    
    # 5. Shade the error gap
    ax.axvspan(true_release_sec, api_delay_sec, color=C_API, alpha=0.08, zorder=1)
    
    # ==========================================
    # CAD-STYLE DIMENSION ARROW
    # ==========================================
    # Place arrow 15% above the highest point of the ball
    arrow_y = max_height * 1.15  
    
    # Draw double-headed arrow
    ax.annotate('', xy=(true_release_sec, arrow_y), xytext=(api_delay_sec, arrow_y),
                arrowprops=dict(arrowstyle="<->", color=C_API, lw=2.5), zorder=4)
                
    # Center the text directly ON the arrow line (borderless white background)
    mid_point_sec = true_release_sec + (lag_sec / 2)
    ax.text(mid_point_sec, arrow_y, f"ANNOTATION DELAY: {lag_sec:.2f} SEC", 
            ha='center', va='center', fontsize=12, fontweight='black', color=C_API,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.8), zorder=5)

    # ==========================================
    # DIRECT ANNOTATIONS (Kinematics Theme)
    # ==========================================
    ax.annotate('Kinematic Release',
                xy=(true_release_sec, z_traj[true_release_idx]),
                xytext=(true_release_sec - 0.2, max_height * 1.05), 
                arrowprops=dict(arrowstyle="->", color=C_TRUE, lw=2.5, connectionstyle="arc3,rad=-0.1"),
                fontsize=13, fontweight='bold', color=C_TRUE, ha='right', va='center')
    
    api_y_val = z_traj[api_delay_idx] if api_delay_idx < len(z_traj) else z_traj[-1]
    ax.annotate('Official PBP Timestamp',
                xy=(api_delay_sec, api_y_val),
                xytext=(api_delay_sec + 0.2, max_height * 1.05), 
                arrowprops=dict(arrowstyle="->", color=C_API, lw=2.5, connectionstyle="arc3,rad=0.1"),
                fontsize=13, fontweight='bold', color=C_API, ha='left', va='center')

    # ==========================================
    # FORMATTING & CLEANUP
    # ==========================================
    ax.set_title("Temporal Lag in Raw Play-by-Play Shot Timestamps", 
                 fontsize=18, fontweight='black', pad=25, color='#333333')
    ax.set_xlabel("Time (Seconds)", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel("Ball Height (Feet)", fontsize=14, fontweight='bold', labelpad=10)
    
    # Give the chart 35% vertical headroom
    ax.set_ylim(bottom=np.min(z_traj) - 1, top=max_height * 1.35)
    ax.set_xlim(left=time_sec[0], right=time_sec[-1])
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.grid(axis='x', alpha=0) 
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
        ax.spines[spine].set_color('#333333')
        
    ax.tick_params(axis='both', labelsize=12, width=2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

