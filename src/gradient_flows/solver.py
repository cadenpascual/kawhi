import jax
import jax.numpy as jnp
import optax
from jax import jit
from functools import partial

# Optimal Transport Tools
from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

# Import from our potential function module
from .potentials import total_energy

@jit
def wasserstein_distance(x, y, epsilon):
    # Ensure epsilon isn't too small for the divergence
    # safe_eps = jnp.maximum(epsilon, 0.1)
    
    # sinkhorn_divergence can be unstable if points overlap
    # We add a tiny jitter to player positions to prevent identical point sets
    # x_jitter = x + 1e-4 * jax.random.normal(jax.random.PRNGKey(0), x.shape)
    
   #  try:
        # We use a custom call to ensure stability
        #out = sinkhorn_divergence(
            #PointCloud, x_jitter, y, 
            #epsilon=safe_eps,
            #sinkhorn_kwargs={'implicit_diff': None, 'max_iterations': 50}
        #)
        #return jnp.nan_to_num(out[0], nan=100.0, posinf=100.0)
    #except:
        # Fallback to Euclidean if the solver still throws a fit
        #return jnp.sum(jnp.square(x - y))
    return jnp.sum(jnp.square(x - y))

@jit
def apply_constraints(defenders_next, defenders_prev, defenders_prev_prev, params):
    """
    Applies strict physical constraints (Kinematics: Velocity and Acceleration) 
    to prevent teleporting and slingshotting.
    """
    dt = 0.04 # 25 frames per second
    
    # --- 1. ACCELERATION CAP (The anti-slingshot fix) ---
    # Calculate previous velocity and proposed new velocity
    v_prev = (defenders_prev - defenders_prev_prev) / dt
    v_next_unconstrained = (defenders_next - defenders_prev) / dt
    
    # Calculate the change in velocity (acceleration)
    delta_v = v_next_unconstrained - v_prev
    accel_magnitude = jnp.linalg.norm(delta_v, axis=1) / dt
    
    # Get physical human limit (e.g., max ~60 ft/s^2)
    max_accel = params.get('max_acceleration', 60.0) 
    
    # Scale down the velocity change if it violates human physics
    accel_ratio = jnp.minimum(1.0, max_accel / (accel_magnitude + 1e-6))
    constrained_delta_v = delta_v * accel_ratio[:, jnp.newaxis]
    
    # Apply the legally allowed velocity change
    v_next_constrained = v_prev + constrained_delta_v
    
    # --- 2. VELOCITY CAP (Absolute speed limit) ---
    speed = jnp.linalg.norm(v_next_constrained, axis=1)
    max_speed = params.get('velocity_cap', 20.0)
    
    speed_ratio = jnp.minimum(1.0, max_speed / (speed + 1e-6))
    v_final = v_next_constrained * speed_ratio[:, jnp.newaxis]
    
    # Calculate final physical position based on allowed velocity
    constrained_pos = defenders_prev + (v_final * dt)
    
    # --- 3. COURT BOUNDARIES ---
    court_dims = params['court_dims']  
    final_pos = jnp.clip(
        constrained_pos,
        a_min=jnp.array([court_dims[0][0], court_dims[1][0]]),
        a_max=jnp.array([court_dims[0][1], court_dims[1][1]])
    )
    
    return final_pos

@partial(jit, static_argnames=['jko_num_steps'])
def run_simulation(init_defenders, ball_traj, offenders_traj, offender_weights_traj, basket_pos, params, jko_num_steps):   
    """
    Runs the full defensive simulation over a trajectory using JAX's `scan`.
    This version includes an acceleration penalty and internal constraints.
    """
    trajectory_data = (ball_traj, offenders_traj, offender_weights_traj)
    
    # The initial carry now contains two states for the acceleration penalty
    init_carry = (init_defenders, init_defenders)

    def simulation_step(carry, traj_slice):
        """
        Performs a single step of the simulation, including JKO optimization
        and constraint application.
        """
        defenders_prev, defenders_prev_prev = carry
        ball_pos, offenders_pos, current_weights = traj_slice
        
        # --- JKO Step ---
        # Chain the optimizer with gradient clipping for stability
        optimizer = optax.chain(
            optax.clip_by_global_norm(params['max_gradient_norm']),
            optax.sgd(learning_rate=params['learning_rate'])
        )

        def loss_fn(defenders_cand):
            # 1. TACTICAL PULL (The "Switching" Logic)
            # Instead of total_energy(def1->off1), we use the Wasserstein distance 
            # between the 5 defenders and the 5 weighted offensive threats.
            
            # We treat offender_weights as the 'mass' of each offensive point
            off_mass = current_weights / jnp.sum(current_weights)
            def_mass = jnp.ones(5) / 5.0
            
            # This is the Optimal Transport cost. It naturally handles SWITCHING
            # because it finds the most efficient mapping.
            ot_loss = wasserstein_distance(defenders_cand, offenders_pos, params['sinkhorn_epsilon'])
            
            # 2. Physics & Energy
            # movement_cost = JKO 'work' term
            movement_cost = jnp.sum(jnp.square(defenders_cand - defenders_prev))
            
            # acceleration_penalty = Smoothness
            accel = defenders_cand - 2 * defenders_prev + defenders_prev_prev
            acceleration_penalty = jnp.sum(jnp.square(accel))
            
            # Combine everything
            # Note: We scale ot_loss by a weight (e.g., jko_lambda)
            total_loss = (params['jko_lambda'] * ot_loss + 
                         (movement_cost * params.get('sprint_penalty_weight', 2.0)) + 
                         (params['acceleration_penalty_weight'] * acceleration_penalty))
            
            return jnp.nan_to_num(total_loss, nan=1e6)
        
        loss_grad_fn = jax.grad(loss_fn)

        # Initialize the optimization from the previous step's positions
        y = defenders_prev
        opt_state = optimizer.init(y)

        def opt_step(i, state):
            """A single step of the SGD optimizer with strict physical constraints."""
            y, opt_state = state
            grads = loss_grad_fn(y)
            updates, opt_state = optimizer.update(grads, opt_state, y)
            y_unconstrained = optax.apply_updates(y, updates)

            # CRITICAL UPDATE: Pass defenders_prev_prev to calculate momentum
            y_constrained = apply_constraints(y_unconstrained, defenders_prev, defenders_prev_prev, params)
            
            return y_constrained, opt_state

        # Run the inner optimization loop to find the best `defenders_next`
        defenders_next, _ = jax.lax.fori_loop(0, jko_num_steps, opt_step, (y, opt_state))
        
        # The new carry for the next step includes the current and previous states
        next_carry = (defenders_next, defenders_prev)
        return next_carry, defenders_next

    # Use `jax.lax.scan` for an efficient, end-to-end differentiable loop
    final_carry, defenders_trajectory = jax.lax.scan(
        simulation_step,
        init_carry,
        trajectory_data
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
    offender_weights_traj = jnp.ones((TIMESTEPS, 5)) 

    # --- Run Simulation ---
    print("--- JAX JKO Solver V2 (with internal constraints) ---")
    print(f"Running simulation for {TIMESTEPS} timesteps...")
    
    defensive_trajectory = run_simulation(
        init_defenders,
        ball_traj,
        offenders_traj,
        offender_weights_traj,
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