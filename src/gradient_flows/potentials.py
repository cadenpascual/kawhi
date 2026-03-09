import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# --- Default Parameters ---
params = {
    # Blending params
    'blending_radius': jnp.array(15.0),
    'blending_k': jnp.array(0.5),

    # Weights
    'attraction_weight': jnp.array(12.0),
    'basket_gravity_weight': jnp.array(2.0),
    'basket_gravity_sigma': jnp.array(12.0),
    'field_weight': jnp.array(-8.0), # Negative for attraction
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

    # IST Params
    'ist_q_exp': jnp.array(2.16),
    'ist_o_exp': jnp.array(1.03),
}

# --- Helper Functions ---

@jit
def softmin(x, k):
    """
    Smooth, differentiable approximation of the min function.
    """
    y = -k * x
    y_max = jnp.max(y)
    log_sum_exp = y_max + jnp.log(jnp.sum(jnp.exp(y - y_max)))
    return -log_sum_exp / k

@jit
def _get_rotated_diff(pos, center, rotation_angle):
    """
    Rotates the coordinate system so the 'long' axis points to the basket.
    """
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    
    # Rotation Matrix
    cos_a = jnp.cos(rotation_angle)
    sin_a = jnp.sin(rotation_angle)
    
    # Rotate coordinates
    x_prime = dx * cos_a + dy * sin_a
    y_prime = -dx * sin_a + dy * cos_a
    
    return x_prime, y_prime

@jit
def _calculate_offset_attractor(off_pos, basket_pos, params):
    """
    Calculates the 'Ideal' position: slightly shifted toward the basket.
    """
    # Vector from Offender to Basket
    vec_to_hoop = basket_pos - off_pos
    dist_to_hoop = jnp.linalg.norm(vec_to_hoop) + 1e-6
    unit_vec = vec_to_hoop / dist_to_hoop
    
    # The 'Cushion' Target
    target_pos = off_pos + params['cushion_dist'] * unit_vec
    
    # The Rotation Angle (angle of the vector to the hoop)
    rotation_angle = jnp.arctan2(vec_to_hoop[1], vec_to_hoop[0])
    
    return target_pos, rotation_angle

@jit
def _calculate_occupancy_penalty(all_defenders_pos, params):
    """
    Calculates a penalty based on the overlap of influence Gaussians between
    all pairs of defenders.
    We use a simplified circular Gaussian for this.
    """
    # Use 'sigma_wide' as a proxy for personal space radius
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
def _calculate_ist_penalty(all_defenders_pos, offenders_pos, q_values, ball_pos, basket_pos, params):
    def single_ist(off_pos, q_val):
        k_smooth = params.get('ist_k_smooth', 10.0)
        # 1. B-Traj (Distance to ball) 
        ball_dist = jnp.linalg.norm(off_pos - ball_pos)
        b_val = 0.4 + 0.6 * (1.0 / (1.0 + jnp.exp(0.3 * (ball_dist - 18.0))))
        
        # 2. Sim Weights using the REAL q_val passed from the dataframe
        sim_weight = jnp.maximum((q_val ** params['ist_q_exp']) * b_val, 0.35)
        
        # 3. Openness (O) with softmin
        dists_to_defs = jnp.linalg.norm(all_defenders_pos - off_pos, axis=1)
        closest_dist = softmin(dists_to_defs, k=k_smooth)
        openness = jnp.clip(closest_dist / 6.0, 0.5, 1.5)
        
        return sim_weight * (openness ** params['ist_o_exp'])
        
    weight = params.get('ist_weight', 0.0)
    # vmap across all 5 offenders AND their 5 q_values simultaneously
    all_ists = vmap(single_ist)(offenders_pos, q_values)
    return weight * jnp.sum(all_ists)

# --- Potential Energy Calculation ---

@jit
def _total_energy_per_defender(current_defender_pos, all_defenders_pos, offenders_pos, ball_pos, basket_pos, params):
    
    # 1. BASKET ANCHOR (Safe Home)
    dist_sq_basket = jnp.sum((current_defender_pos - basket_pos)**2)
    E_gravity = -params['basket_gravity_weight'] * jnp.exp(-dist_sq_basket / (2 * params['basket_gravity_sigma']**2))

    # 2. OFFENDER ATTRACTION (Anisotropic Field)
    def single_offender_potential(off_pos):
        target_pos, angle = _calculate_offset_attractor(off_pos, basket_pos, params)
        dx_rot, dy_rot = _get_rotated_diff(current_defender_pos, target_pos, angle)
        
        exponent = -((dx_rot**2 / (2 * params['sigma_long']**2)) + 
                     (dy_rot**2 / (2 * params['sigma_wide']**2)))
                       
        dist_off_to_ball = jnp.linalg.norm(off_pos - ball_pos)
        threat_scale = 1.0 + params['offender_threat_scale'] * jnp.exp(-(dist_off_to_ball / params['offender_threat_dist'])**2)
        dynamic_weight = params['field_weight'] * threat_scale
        
        return dynamic_weight * jnp.exp(exponent)
    
    E_field = jnp.sum(vmap(single_offender_potential)(offenders_pos))

    # 3. COLLECTIVE TEAM-LEVEL POTENTIALS

    # This prevents the 'pushed out' problem by pulling rogue players back to the group
    team_centroid = jnp.mean(all_defenders_pos, axis=0)
    dist_to_centroid = jnp.linalg.norm(current_defender_pos - team_centroid)
    
    # Linear pull: 0 if inside the radius, grows linearly if they drift too far
    E_cohesion = params['cohesion_weight'] * jnp.maximum(0, dist_to_centroid - params['formation_radius'])

    def _calculate_ball_pressure(defender_pos):
        dist_sq_ball = jnp.sum((defender_pos - ball_pos)**2)
        dist_to_ball = jnp.sqrt(dist_sq_ball)
        blend_factor = jax.nn.sigmoid(-params['blending_k'] * (dist_to_ball - params['blending_radius']))
        
        E_local = -params['attraction_weight'] * jnp.exp(-dist_sq_ball / (2 * params['sigma_wide']**2))
        E_global = params['global_ball_weight'] * dist_sq_ball
        return (blend_factor * E_local) + E_global
    
    def boundary_penalty(pos):
        # Soft 'spring' at the edges of the 94x50 court
        dist_to_left = pos[0] - 0
        dist_to_right = 94 - pos[0]
        dist_to_bottom = pos[1] - 0
        dist_to_top = 50 - pos[1]
        
        # Penalize getting within 2 feet of any boundary
        buffer = 2.0
        p = jnp.sum(jnp.array([
            jnp.maximum(0, buffer - dist_to_left)**2,
            jnp.maximum(0, buffer - dist_to_right)**2,
            jnp.maximum(0, buffer - dist_to_bottom)**2,
            jnp.maximum(0, buffer - dist_to_top)**2
        ]))
        return p * 10.0 # Weight of the boundary penalty

    all_ball_energies = vmap(_calculate_ball_pressure)(all_defenders_pos)
    E_collective_ball = softmin(all_ball_energies, k=params['k_softmin'])
    
    E_occupancy = _calculate_occupancy_penalty(all_defenders_pos, params)

    E_boundary = jnp.sum(vmap(boundary_penalty)(all_defenders_pos))
    
    return E_gravity + E_field + E_collective_ball + E_occupancy + E_boundary + E_cohesion

@jit
def total_energy(defenders, offenders, q_values, ball, basket_pos, params):
    per_def_energy = vmap(_total_energy_per_defender, in_axes=(0, None, None, None, None, None))(
        defenders, defenders, offenders, ball, basket_pos, params
    )
    # Calculate global IST penalty using the real q_values
    ist_energy = _calculate_ist_penalty(defenders, offenders, q_values, ball, basket_pos, params)

    return per_def_energy + ist_energy

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

    # Defender is slightly behind and to the side of the ideal cushion spot
    defenders = jnp.array([
        [34., 28.], # Defender 0
        [8., 38.],
        [8., 12.],
        [20., 35.],
        [20., 15.]
    ])

    print("--- JAX Potentials V5: Anisotropic Elliptical Wells ---")

    # Calculate the ideal spot for defender 0 to guard offender 0
    target_pos_0, angle_0 = _calculate_offset_attractor(offenders[0], basket_pos, params)
    
    print(f"\nOffender 0 Position: {offenders[0]}")
    print(f"Ideal Cushion Position for D0: {target_pos_0}")
    print(f"Rotation Angle (degrees): {jnp.rad2deg(angle_0):.1f}")
    
    # Calculate energy and gradient
    energies = total_energy(defenders, offenders, ball, basket_pos, params)
    grad_fn = jax.grad(lambda d: total_energy(d, offenders, ball, basket_pos, params).sum())
    grads = grad_fn(defenders)

    print(f"\nDefender 0 Position: {defenders[0]}")
    print(f"Energy for Defender 0: {energies[0]:.4f}")
    print(f"Gradient for Defender 0: {grads[0]}")

    # The gradient should point from the defender's current position toward the target position
    vec_to_target = target_pos_0 - defenders[0]
    dot_product = jnp.dot(grads[0], vec_to_target)

    print(f"\nVector from D0 to Target: {vec_to_target}")
    print(f"Dot product of (Gradient) and (Vector to Target): {dot_product:.4f}")
    assert dot_product < 0 # Negative dot product means gradient points towards the target
    print("\nAssertion passed: Gradient correctly points defender toward the ideal cushion spot.")
