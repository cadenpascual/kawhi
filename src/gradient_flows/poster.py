import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

from src.gradient_flows.court import draw_court_matplotlib, draw_plotly_court
from src.viz.court import draw_half_court_ft

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


def plot_ist_optimization_map(final_df, filename="ist_optimization_heatmap.png"):
    # 1. Coordinate Extraction
    def get_val_at_release(row, col):
        try:
            # Accessing the N-1 frame
            idx = int(row['local_release_idx']) - 1
            return row[col][idx]
        except:
            return np.nan

    # Create plotting columns
    final_df['x_release'] = final_df.apply(lambda r: get_val_at_release(r, 'off1_x_traj'), axis=1)
    final_df['y_release'] = final_df.apply(lambda r: get_val_at_release(r, 'off1_y_traj'), axis=1)
    # The 'Dividend': The specific value-add of the IST weights
    final_df['ist_dividend'] = final_df['Base_IST_Total'] - final_df['Sim_IST_Total']

    # 2. Setup Plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 11), dpi=300)
    fig.patch.set_facecolor('#BEC0C2'); fig.patch.set_alpha(0.2)
    ax.set_facecolor('none')

    # 3. Draw your specific court
    draw_half_court_ft(ax, color='#1D428A', lw=3, outer_lines=True)

    # 4. Generate the Heatmap (Hexbin)
    # C=ist_dividend ensures we are mapping the SAVINGS, not just shot frequency
    hb = ax.hexbin(final_df['x_release'], final_df['y_release'], 
               C=final_df['ist_dividend'], gridsize=20, 
               cmap='YlOrRd', reduce_C_function=np.mean, 
               alpha=0.8, zorder=2)

    # 5. Dashboard Elements (No-import Manual Colorbar)
    # Define a small axis inside the figure for the colorbar
    # [x, y, width, height] in axes coordinates (0 to 1)
    cax = ax.inset_axes([1.02, 0.1, 0.03, 0.8]) 

    cb = fig.colorbar(hb, cax=cax)
    cb.set_label('Avg IST Savings (Units)', fontsize=12, fontweight='bold', color='#1D428A')

    # Reset the title and layout as before
    ax.set_title("IST OPTIMIZATION MAP", fontsize=24, fontweight='black', pad=30, color='#1D428A')
        
    # Set limits to match NBA half-court in feet
    ax.set_xlim(-26, 26)
    ax.set_ylim(-6, 45)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor())
    plt.show()

def plot_triple_ist_shift(final_df, filename="ist_triple_shift_dashboard.png"):
    plt.style.use('default')
    
    # 1. Branding & Colors
    # Grey for Real, Dark Grey/Silver for Base, Blue for Sim, Red for Callouts
    C_REAL, C_BASE, C_SIM, C_OFF, C_BG = '#888888', '#555555', '#1D428A', '#C8102E', '#BEC0C2'
    
    # 2. Calculate Stats for the Dashboard
    real_mean = final_df['Real_IST_Total'].mean()
    base_mean = final_df['Base_IST_Total'].mean()
    sim_mean = final_df['Sim_IST_Total'].mean()
    
    total_waste = real_mean - sim_mean   # 76.74
    ist_savings = base_mean - sim_mean   # 37.28

    # 3. Figure Setup
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor(C_BG); fig.patch.set_alpha(0.15); ax.set_facecolor('none')
    
    # 4. Plot Distributions
    # Real: The baseline chaotic reality
    sns.kdeplot(final_df['Real_IST_Total'], fill=True, color=C_REAL, lw=4, 
                label='Actual NBA Defense', alpha=0.2, ax=ax)
    
    # Base: The smart model without physical weighting
    sns.kdeplot(final_df['Base_IST_Total'], fill=False, color=C_BASE, lw=4, ls='--', 
                label='Base Sim Model', ax=ax)
    
    # Sim: The fully optimized model (Final)
    sns.kdeplot(final_df['Sim_IST_Total'], fill=True, color=C_SIM, lw=6, 
                label='Your Sim Model', alpha=0.5, ax=ax)
    
    # 5. Reference Lines (Medians/Means)
    ax.axvline(real_mean, color=C_REAL, ls=':', lw=3, alpha=0.6)
    ax.axvline(sim_mean, color=C_SIM, ls=':', lw=3, alpha=0.8)

    # 6. THE RECOVERY GAP ARROW
    y_limit = ax.get_ylim()[1]
    arrow_y = y_limit * 0.40 
    ax.annotate('', xy=(sim_mean, arrow_y), xytext=(real_mean, arrow_y),
                arrowprops=dict(arrowstyle='<->', color=C_OFF, lw=5))
    ax.text((real_mean + sim_mean)/2, arrow_y + (y_limit * 0.02), "AVG IST SAVED", 
            color=C_OFF, fontweight='black', fontsize=16, ha='center')

    # 7. Titles & Labels
    ax.set_title("Total IST Distribution per Model", fontsize=42, fontweight='black', 
                 pad=60, color=C_SIM, loc='center')
    ax.set_xlabel("Team IST Per Play", fontsize=24, fontweight='bold', labelpad=20)
    ax.set_ylabel("Frequency of Outcomes", fontsize=24, fontweight='bold', labelpad=20)
    
    # 8. THE HERO DASHBOARD (Top Right)
    ax.legend(fontsize=18, loc='upper right', frameon=True, shadow=True, 
              facecolor='white', edgecolor=C_SIM, borderpad=1.2)

    stats_text = (f"Real Model Savings: {total_waste:.2f} Units IST\n"
                  f"Base Model Savings: {ist_savings:.2f} Units IST")
    
    ax.text(0.97, 0.62, stats_text, transform=ax.transAxes,
            fontsize=20, fontweight='black', color='white', 
            ha='right', va='top',
            bbox=dict(boxstyle="round,pad=1.0", facecolor=C_OFF, edgecolor='none'))

    # 9. Final Polish
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(C_SIM); ax.spines[s].set_linewidth(4)
    
    ax.tick_params(axis='both', labelsize=18, width=3, length=10)
    
    plt.tight_layout()
    plt.show()
