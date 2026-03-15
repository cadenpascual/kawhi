import jax
import jax.numpy as jnp
import optax
from jax import jit
from functools import partial

# Optimal Transport Tools
from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

# Import from our potential function module
from src.gradient_flows.potentials import total_energy

@jit
def wasserstein_distance(x, y, epsilon):
    """
    Calculates the Sinkhorn divergence between two sets of 5 points.
    This serves as a differentiable approximation of the Wasserstein-2 distance.

    Args:
        x (jax.Array): First set of points, shape (5, 2).
        y (jax.Array): Second set of points, shape (5, 2).
        epsilon (float): Entropy regularization parameter.

    Returns:
        float: The Sinkhorn divergence value.
    """
    # OTT-JAX offers a high-level API for Sinkhorn divergence
    out = sinkhorn_divergence(
        PointCloud, x, y, epsilon=epsilon
    )
    return out[0]

@jit
def apply_constraints(defenders_next, defenders_prev, params):
    """
    Applies physical constraints to player movements.

    Args:
        defenders_next (jax.Array): Proposed new positions from the JKO step.
        defenders_prev (jax.Array): Positions from the previous timestep.
        params (dict): Dictionary of simulation parameters.

    Returns:
        jax.Array: Defender positions after applying all constraints.
    """
    # 1. Velocity Cap to prevent "teleportation"
    max_dist_per_step = params['velocity_cap']
    delta = defenders_next - defenders_prev
    distance = jnp.linalg.norm(delta, axis=1)
    
    # Calculate the ratio to scale down movements that are too large
    # Add a small epsilon to avoid division by zero
    ratio = jnp.minimum(1.0, max_dist_per_step / (distance + 1e-6))
    
    # Apply the velocity cap by rescaling the movement vector
    constrained_delta = delta * ratio[:, jnp.newaxis]
    constrained_pos = defenders_prev + constrained_delta
    
    # 2. Court Boundaries to keep players on the court
    court_dims = params['court_dims']  # e.g., [[0, 94], [0, 50]]
    final_pos = jnp.clip(
        constrained_pos,
        a_min=jnp.array([court_dims[0][0], court_dims[1][0]]),
        a_max=jnp.array([court_dims[0][1], court_dims[1][1]])
    )
    
    return final_pos

@partial(jit, static_argnames=['jko_num_steps'])
def run_simulation(init_defenders, ball_traj, offenders_traj, q_traj, basket_pos, params, jko_num_steps):
    """
    Runs the full defensive simulation over a trajectory using JAX's `scan`.
    If q_traj is None, it defaults to a zero-filled array of the appropriate shape.
    """
    # If no q_traj is provided, create a dummy array of zeros 
    # matching the number of timesteps and number of offenders (5).
    if q_traj is None:
        q_traj = jnp.zeros((offenders_traj.shape[0], 5))
    
    trajectory_data = (ball_traj, offenders_traj, q_traj)
    init_carry = (init_defenders, init_defenders)

    def simulation_step(carry, traj_slice):
        defenders_prev, defenders_prev_prev = carry
        ball_pos, offenders_pos, q_values = traj_slice # q_values is now always an array
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(params['max_gradient_norm']),
            optax.adam(learning_rate=params['learning_rate'])
        )

        def loss_fn(defenders_cand):
            # Potential Energy: total_energy now receives q_values (actual or zeros)
            energy = jnp.sum(total_energy(defenders_cand, offenders_pos, q_values, ball_pos, basket_pos, params))
            
            # Kinetic Energy / Optimal Transport Cost
            w2_dist = wasserstein_distance(defenders_cand, defenders_prev, epsilon=params['sinkhorn_epsilon'])
            
            # Acceleration Penalty
            accel = defenders_cand - 2 * defenders_prev + defenders_prev_prev
            acceleration_penalty = jnp.sum(jnp.square(accel))
            
            # Velocity Penalty
            velocity_penalty = jnp.sum(jnp.square(defenders_cand - defenders_prev))
            
            return (energy + 
                    w2_dist / params['jko_lambda'] +
                    params['acceleration_penalty_weight'] * acceleration_penalty +
                    params['velocity_penalty_weight'] * velocity_penalty)

        loss_grad_fn = jax.grad(loss_fn)
        y = defenders_prev
        opt_state = optimizer.init(y)

        def opt_step(i, state):
            y, opt_state = state
            grads = loss_grad_fn(y)
            updates, opt_state = optimizer.update(grads, opt_state, y)
            y_unconstrained = optax.apply_updates(y, updates)
            y_constrained = apply_constraints(y_unconstrained, defenders_prev, params)
            return y_constrained, opt_state

        defenders_next, _ = jax.lax.fori_loop(0, jko_num_steps, opt_step, (y, opt_state))
        
        next_carry = (defenders_next, defenders_prev)
        return next_carry, defenders_next

    final_carry, defenders_trajectory = jax.lax.scan(
        simulation_step,
        init_carry,
        trajectory_data,
    )
    
    return defenders_trajectory

from .potentials import params as potential_params
params = {
        **potential_params,
        'jko_lambda': 0.5,
        'sinkhorn_epsilon': 0.01,
        'velocity_cap': 0.8,
        'court_dims': [[0., 94.], [0., 50.]],
        'max_gradient_norm': 1.0,
        'acceleration_penalty_weight': 2.0,
    }

if __name__ == '__main__':
    # --- Simulation Setup ---
    key = jax.random.PRNGKey(0)
    TIMESTEPS = 50
    JKO_STEPS = 20

    # --- Create Mock Data ---
    basket_pos = jnp.array([5.25, 25.0])
    init_defenders = jnp.array([
        [32., 26.], [8., 38.], [8., 12.], [12., 30.], [12., 20.]
    ])
    init_offenders = jnp.array([
        [10., 40.], [10., 10.], [30., 25.], [15., 35.], [15., 15.]
    ])

    offenders_traj = jnp.zeros((TIMESTEPS, 5, 2)) + init_offenders
    ball_carrier_start = init_offenders[2]
    ball_carrier_end = jnp.array([10., 25.])
    ball_carrier_traj = jnp.linspace(ball_carrier_start, ball_carrier_end, TIMESTEPS)
    
    offenders_traj = offenders_traj.at[:, 2, :].set(ball_carrier_traj)
    ball_traj = ball_carrier_traj

    # --- Run Simulation ---
    print("--- JAX JKO Solver V2 (with internal constraints) ---")
    print(f"Running simulation for {TIMESTEPS} timesteps...")
    
    defensive_trajectory = run_simulation(
        init_defenders,
        ball_traj,
        offenders_traj,
        basket_pos,
        params,
        jko_num_steps=JKO_STEPS
    )

    # --- Print Results ---
    print(f"\nSimulation complete.")
    print(f"Output trajectory shape: {defensive_trajectory.shape}")
    assert defensive_trajectory.shape == (TIMESTEPS, 5, 2)
    print("Output shape is correct.")

    final_pos = defensive_trajectory[-1]
    on_court = jnp.all(
        (final_pos >= jnp.array(params['court_dims'])[:, 0]) &
        (final_pos <= jnp.array(params['court_dims'])[:, 1])
    )
    assert on_court
    print("\nAssertion passed: All defenders remained within court boundaries.")
    
    # Check for NaNs
    has_nans = jnp.any(jnp.isnan(defensive_trajectory))
    assert not has_nans
    print("Assertion passed: Simulation produced no NaN values.")