import jax
import jax.numpy as jnp
import jax.nn
from jax import jit, vmap
from functools import partial

# --- OPTIMAL TRANSPORT IMPORTS ---
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem 
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear import implicit_differentiation as imp_diff

# --- Default Parameters ---
params = {
    # Blending params
    'blending_radius': jnp.array(15.0),
    'blending_k': jnp.array(0.5),

    # Weights
    'attraction_weight': jnp.array(12.0),
    'basket_gravity_weight': jnp.array(2.0),
    'basket_gravity_sigma': jnp.array(12.0),
    'field_weight': jnp.array(-8.0), # Will be converted to absolute value for OT Cost
    'global_ball_weight': jnp.array(0.05),

    # Threat, Sparsity & Occupancy
    'offender_threat_scale': jnp.array(2.0),
    'offender_threat_dist': jnp.array(15.0),
    'k_softmin': jnp.array(5.0),
    'occupancy_weight': jnp.array(5.0),
    'cohesion_weight': jnp.array(0.5),      # Strength of the 'team rubber band'
    'formation_radius': jnp.array(18.0),   # Natural 'spread' of the shell

    # NEW ELLIPTICAL/ANISOTROPIC PARAMS
    'sigma_long': jnp.array(8.0),   # Long reach towards the perimeter
    'sigma_wide': jnp.array(2.5),   # Narrow width (stay in lane)
    'cushion_dist': jnp.array(3.0), # Stand 3ft off the offender
}

# --- Helper Functions ---

@jit
def softmin(x, k):
    """Smooth, differentiable approximation of the min function."""
    y = -k * x
    y_max = jnp.max(y)
    log_sum_exp = y_max + jnp.log(jnp.sum(jnp.exp(y - y_max)))
    return -log_sum_exp / k

@jit
def _get_rotated_diff(pos, center, rotation_angle):
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    cos_a = jnp.cos(rotation_angle)
    sin_a = jnp.sin(rotation_angle)
    x_prime = dx * cos_a + dy * sin_a
    y_prime = -dx * sin_a + dy * cos_a
    return x_prime, y_prime

@jit
def _calculate_offset_attractor(off_pos, basket_pos, params):
    vec_to_hoop = basket_pos - off_pos
    dist_to_hoop = jnp.sqrt(jnp.sum(vec_to_hoop**2) + 1e-6)
    unit_vec = vec_to_hoop / dist_to_hoop
    target_pos = off_pos + params['cushion_dist'] * unit_vec
    rotation_angle = jnp.arctan2(vec_to_hoop[1], vec_to_hoop[0])
    return target_pos, rotation_angle

@jit
def _calculate_occupancy_penalty(all_defenders_pos, params):
    sigma_sq = jnp.square(params['sigma_wide'])
    idx1, idx2 = jnp.triu_indices(5, k=1)

    def _pair_overlap(i, j):
        pos_i, pos_j = all_defenders_pos[i], all_defenders_pos[j]
        dist_sq = jnp.sum(jnp.square(pos_i - pos_j))
        variance_sum = 2 * sigma_sq
        return jnp.exp(-dist_sq / (2 * variance_sum))

    total_overlap = jnp.sum(vmap(_pair_overlap)(idx1, idx2))
    return params['occupancy_weight'] * total_overlap


@jit
def _calculate_lane_blocking_penalty(defenders, offenders, basket_pos, params):
    """
    Penalizes defenders for not being directly on the line between the offenders and the basket.
    """
    # 1. Calculate the ideal defensive point for every offender
    # Shape of offenders: (5, 2), basket_pos: (2,)
    vec_to_basket = basket_pos - offenders
    dist_to_basket = jnp.linalg.norm(vec_to_basket, axis=-1, keepdims=True)
    
    # Normalize to get direction (add epsilon to prevent div by zero)
    dir_to_basket = vec_to_basket / (dist_to_basket + 1e-6)
    
    # Place ideal points 'cushion_dist' away from offenders, towards the basket
    # (We use jnp.minimum so we don't accidentally put the point behind the basket)
    actual_cushion = jnp.minimum(params['cushion_dist'], dist_to_basket * 0.8)
    ideal_points = offenders + dir_to_basket * actual_cushion
    
    # 2. Calculate distances from every defender to every ideal point
    # Resulting shape: (5, 5) cost matrix
    diffs = defenders[:, None, :] - ideal_points[None, :, :]
    dist_matrix = jnp.linalg.norm(diffs, axis=-1)
    
    # 3. Softmin to find the closest ideal point for each defender
    # This automatically associates a defender with the closest driving lane
    import jax.nn as jnn
    closest_ideal_dists = jnp.sum(jnn.softmax(-params['k_softmin'] * dist_matrix, axis=1) * dist_matrix, axis=1)
    
    # Return squared distance as an energy penalty
    return jnp.sum(closest_ideal_dists ** 2) * params.get('lane_blocking_weight', 2.0)


# --- Potential Energy Calculation ---

@jit
def calculate_dynamic_ist(defenders, offenders, ball_pos, q_values, params):
    """
    Calculates normalized IST dynamically based on simulated defender positions.
    """
    # 1. Openness (O) - Using SIMULATED defenders
    diffs = offenders[:, None, :] - defenders[None, :, :]
    dist_matrix = jnp.linalg.norm(diffs, axis=-1)
    
    # Distance to nearest defender (clipped to prevent zero-division in gradients)
    O = jnp.clip(jnp.min(dist_matrix, axis=1), 0.1, 25.0)
    
    # 2. Ball Factor (B)
    dist_to_ball = jnp.linalg.norm(offenders - ball_pos, axis=-1)
    k = 0.3
    d0 = 18.0
    b_floor = 0.4
    B = b_floor + (1.0 - b_floor) / (1.0 + jnp.exp(k * (dist_to_ball - d0)))
    
    # 3. Pull Learnable Exponents from params
    q_exp = params.get('ist_q_exp', 2.0)
    o_exp = params.get('ist_o_exp', 1.0)
    
    # 4. Calculate Raw IST
    raw_ist = (q_values ** q_exp) * (O ** o_exp) * B
    
    # 5. Normalize so "mass" always equals 5.0
    total_ist = jnp.sum(raw_ist) + 1e-6
    normalized_ist = (raw_ist / total_ist) * 5.0
    
    return normalized_ist

@jit
def _ot_tactical_energy(all_defenders_pos, offenders_pos, basket_pos, params, offender_weights):
    # 1. Ensure weights are NEVER zero or near-zero
    a_weights = jnp.ones(5) / 5.0 
    # Add a tiny epsilon and re-normalize to prevent singular systems
    b_weights = offender_weights + 1e-6
    b_weights = b_weights / jnp.sum(b_weights)
    
    # 2. Setup Geometry
    def get_cushion(off_pos):
        target, _ = _calculate_offset_attractor(off_pos, basket_pos, params)
        return target
    ideal_spots = vmap(get_cushion)(offenders_pos)
    
    # Use a slightly larger epsilon for the PointCloud (blurrier is safer)
    geom = PointCloud(all_defenders_pos, ideal_spots, epsilon=params.get('sinkhorn_epsilon', 1.0))
    prob = linear_problem.LinearProblem(geom, a=a_weights, b=b_weights)
    
    # 3. THE CRITICAL CHANGE: 
    # Set implicit_diff=None to use unrolled differentiation.
    # This avoids the Lineax/Equinox linear solver crash.
    ot_solver = sinkhorn.Sinkhorn(
        threshold=1e-2, 
        max_iterations=100, 
        lse_mode=True,
        implicit_diff=None  # <--- THIS IS THE FIX
    )
    
    out = ot_solver(prob)
    
    # Calculate energy
    # We use stop_gradient on the weights if we only want to optimize defender positions
    return jnp.abs(params.get('field_weight', 5.0)) * jnp.sum(out.matrix * geom.cost_matrix)

@jit
def total_energy(defenders, offenders, ball, basket_pos, params, q_values):
    """
    Calculates the total energy by combining global team structures (Sinkhorn)
    and independent player constraints. Now includes dynamic IST tracking!
    """
    # NEW STEP: Calculate dynamic IST using simulated defenders!
    # Because 'defenders' is passed in here, JAX's autograd will build 
    # gradients that push defenders toward the most dangerous players.
    live_offender_weights = calculate_dynamic_ist(defenders, offenders, ball, q_values, params)

    # 1. Calculate Optimal Transport Assignments (Returns shape (5,))
    # Pass the dynamically calculated live weights instead of static ones
    E_ot_field = _ot_tactical_energy(defenders, offenders, basket_pos, params, live_offender_weights)
    
    # 2. Calculate Individual Defender Potentials
    def single_defender_energy(current_defender_pos):
        # Basket Gravity
        dist_sq_basket = jnp.sum((current_defender_pos - basket_pos)**2)
        E_gravity = -params['basket_gravity_weight'] * jnp.exp(-dist_sq_basket / (2 * params['basket_gravity_sigma']**2))
        
        # Cohesion (Rubber Band)
        team_centroid = jnp.mean(defenders, axis=0)
        dist_to_centroid = jnp.sqrt(jnp.sum((current_defender_pos - team_centroid)**2) + 1e-6)
        E_cohesion = params['cohesion_weight'] * jnp.maximum(0.0, dist_to_centroid - params['formation_radius'])
        
        # Boundaries
        dist_to_left = current_defender_pos[0] - 0.0
        dist_to_right = 94.0 - current_defender_pos[0]
        dist_to_bottom = current_defender_pos[1] - 0.0
        dist_to_top = 50.0 - current_defender_pos[1]
        buffer = 2.0
        p = jnp.sum(jnp.array([
            jnp.maximum(0.0, buffer - dist_to_left)**2,
            jnp.maximum(0.0, buffer - dist_to_right)**2,
            jnp.maximum(0.0, buffer - dist_to_bottom)**2,
            jnp.maximum(0.0, buffer - dist_to_top)**2
        ]))
        E_boundary = p * 10.0
        
        return E_gravity + E_cohesion + E_boundary

    E_indiv = vmap(single_defender_energy)(defenders)
    
    # 3. Calculate Collective Ball Pressure (Who guards the ball?)
    def _calculate_ball_pressure(defender_pos):
        dist_sq_ball = jnp.sum((defender_pos - ball)**2)
        dist_to_ball = jnp.sqrt(dist_sq_ball + 1e-6)
        blend_factor = jax.nn.sigmoid(-params['blending_k'] * (dist_to_ball - params['blending_radius']))
        E_local = -params['attraction_weight'] * jnp.exp(-dist_sq_ball / (2 * params['sigma_wide']**2))
        E_global = params['global_ball_weight'] * dist_sq_ball
        return (blend_factor * E_local) + E_global

    all_ball_energies = vmap(_calculate_ball_pressure)(defenders)
    E_collective_ball = softmin(all_ball_energies, k=params['k_softmin'])
    
    # 4. Calculate Spatial Penalties
    E_occupancy = _calculate_occupancy_penalty(defenders, params)
    E_lane_blocking = _calculate_lane_blocking_penalty(defenders, offenders, basket_pos, params)
    
    # 5. Sum it all up
    # We divide collective team scalars by 5 so they are evenly distributed
    # when the gradient `.sum()` is called later.
    return E_ot_field + E_indiv*0.0 + (E_collective_ball / 5.0) + (E_occupancy / 5.0) + (E_lane_blocking / 5.0)

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)

    basket_pos = jnp.array([5.25, 25.0])
    offenders = jnp.array([
        [30., 25.], # Ball carrier
        [10., 40.], 
        [10., 10.],
        [15., 35.], 
        [15., 15.]
    ])
    ball = offenders[0]

    defenders = jnp.array([
        [34., 28.], # Defender 0
        [8., 38.],
        [8., 12.],
        [20., 35.],
        [20., 15.]
    ])

    print("--- JAX Potentials V6: Optimal Transport Assignments ---")
    
    # Test with skewed weights (Steph Curry effect)
    mock_weights = jnp.array([1.0, 3.0, 0.5, 0.5, 0.5])
    energies = total_energy(defenders, offenders, ball, basket_pos, params, mock_weights)
    grad_fn = jax.grad(lambda d: total_energy(d, offenders, ball, basket_pos, params, mock_weights).sum())
    grads = grad_fn(defenders)

    print(f"\nEnergy per Defender: \n{energies}")
    print(f"\nGradients (Push forces): \n{grads}")
    
    print("\nInitialization Successful. Sinkhorn assignments are active.")