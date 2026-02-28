import numpy as np
import jax.numpy as jnp

def extract_quality_trajectories(off_traj, off_pids, maps_npz, pid2row):
    """
    Extracts Q-values for 5 offensive players over the entire trajectory.
    Returns a JAX array of shape (TIMESTEPS, 5).
    """
    timesteps = off_traj.shape[0]
    q_traj = np.ones((timesteps, 5), dtype=float) # Default to 1.0 baseline threat

    xedges = maps_npz["xedges"]
    yedges = maps_npz["yedges"]
    quality_grids = maps_npz["quality"]

    for i, pid in enumerate(off_pids):
        pid_int = int(pid)
        if pid_int in pid2row:
            row_idx = pid2row[pid_int]
            grid = quality_grids[row_idx]

            # Sample the grid for this player at every timestep
            for t in range(timesteps):
                x, y = off_traj[t, i, 0], off_traj[t, i, 1]
                
                if np.isnan(x) or np.isnan(y):
                    continue
                    
                # Find grid bin indices
                x_bin = np.searchsorted(xedges, x) - 1
                y_bin = np.searchsorted(yedges, y) - 1
                
                # Clamp to grid bounds to prevent out-of-bounds errors
                x_bin = np.clip(x_bin, 0, grid.shape[0] - 1)
                y_bin = np.clip(y_bin, 0, grid.shape[1] - 1)
                
                q_traj[t, i] = grid[x_bin, y_bin]
        else:
            print(f"No Q-map found for Player ID {pid_int}. Defaulting to 1.0.")

    return jnp.array(q_traj)


def parse_df_row(row):
    """
    Converts a single row from your DataFrame into trajectory arrays.
    Assumes trajectory columns contain lists or arrays.
    """
    # 1. Ball Trajectory (T, 2)
    ball_traj = jnp.stack([jnp.array(row['ball_x_traj']), 
                           jnp.array(row['ball_y_traj'])], axis=1)
    
    # 2. Offensive Trajectories (T, 5, 2) and Q-Values (T, 5)
    off_list = []
    q_list = []
    off_pids = []
    
    for i in range(1, 6):
        # Extract [x, y] for player i
        pos = jnp.stack([jnp.array(row[f'off{i}_x_traj']), 
                         jnp.array(row[f'off{i}_y_traj'])], axis=1)
        off_list.append(pos)
        
        # Extract Q-rating trajectory for player i
        q_list.append(jnp.array(row[f'off{i}_q_traj']))
        off_pids.append(row[f'off1_pid']) # IDs are usually static
        
    off_traj = jnp.stack(off_list, axis=1)   # Shape: (T, 5, 2)
    q_traj = jnp.stack(q_list, axis=1)       # Shape: (T, 5)
    
    # 3. Real Defensive Trajectories (T, 5, 2)
    def_list = []
    for i in range(1, 6):
        pos = jnp.stack([jnp.array(row[f'def{i}_x_traj']), 
                         jnp.array(row[f'def{i}_y_traj'])], axis=1)
        def_list.append(pos)
    real_def_traj = jnp.stack(def_list, axis=1)
    
    return ball_traj, off_traj, real_def_traj, q_traj, off_pids

def prepare_play_data(row):
    """Extracts and slices data specifically up to the local_release_idx."""
    
    # 1. Get the pre-calculated shot index
    end_idx = int(row['local_release_idx']) + 1 
    
    # 2. Ball
    ball_x = jnp.array(row['ball_x_traj'])[:end_idx] + 25.0
    ball_y = jnp.array(row['ball_y_traj'])[:end_idx] + 5.25
    ball_traj = jnp.stack([ball_y, ball_x], axis=1)

    # 3. Offense & Q-Values
    off_list, q_list = [], []
    for i in range(1, 6):
        x = jnp.array(row[f'off{i}_x_traj'])[:end_idx] + 25.0
        y = jnp.array(row[f'off{i}_y_traj'])[:end_idx] + 5.25
        off_list.append(jnp.stack([y, x], axis=1))
        
        q_vals = jnp.array(row[f'off{i}_q_traj'])[:end_idx]
        q_list.append(q_vals)
        
    off_traj = jnp.stack(off_list, axis=1)
    q_traj = jnp.stack(q_list, axis=1)

    # 4. Real Defense
    def_list = []
    for i in range(1, 6):
        x = jnp.array(row[f'def{i}_x_traj'])[:end_idx] + 25.0
        y = jnp.array(row[f'def{i}_y_traj'])[:end_idx] + 5.25
        def_list.append(jnp.stack([y, x], axis=1))
        
    real_def_traj = jnp.stack(def_list, axis=1)
    init_defenders = real_def_traj[0]

    # 5. Basket Position
    basket_pos = jnp.array([5.25, 25.0]) if row['flipped_coordinates'] else jnp.array([88.75, 25.0])

    # 6. Offender Weights 
    ball_expanded = ball_traj[:, None, :] 
    dist_to_ball = jnp.linalg.norm(off_traj - ball_expanded, axis=2)
    b_traj = 0.4 + 0.6 * (1.0 / (1.0 + jnp.exp(0.3 * (dist_to_ball - 18.0))))
    offender_weights_traj = q_traj * b_traj

    data_slice = (init_defenders, ball_traj, off_traj, real_def_traj, basket_pos)
    return data_slice, offender_weights_traj