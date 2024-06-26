#!/usr/bin/env python

##
#
# Toy example where a position controlled point mass robot should go around an
# obstacle.
#
##

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import pickle

# Scenario definition variables
OBSTACLE_POSITION = jnp.array([0.0, 0.0])
START_STATE = jnp.array([0.0, -1.5])
GOAL_STATE = jnp.array([0.0, 1.5])

def obstacle_avoidance_cost(x: jnp.ndarray):
    """Cost function for obstacle avoidance.

    Args:
        x: the state of the robot, a 2D vector [px, py]
    
    Returns:
        the cost of being at state x, given an obstacle.
    """
    sigma = 0.3
    cost = jnp.exp(-1/sigma*jnp.linalg.norm(x - OBSTACLE_POSITION)**2)
    return cost

def cost(U: jnp.ndarray):
    """Compute the trajectory cost J(U).

    Args:
        U: the sequence of actions U = [u_0, u_1, ..., u_{T-1}]. Each u_t is a
           2D vector representing the change in position at time t.

    Returns:
        the cost J(U)
    """
    num_steps = U.shape[0]

    def calc_running_cost(carry, i):
        """Compute the running cost of the trajectory."""
        x, J = carry
        u = U[i]
        J += 0.1 * u.dot(u)
        J += 1.0 * obstacle_avoidance_cost(x)
        x = x + U[i]
        return (x, J), None

    # Compute the running cost for each step in the trajectory
    (x, J), _ = jax.lax.scan(calc_running_cost, (START_STATE, 0.0), jnp.arange(num_steps))

    # Add a cost for reaching the goal
    J += (x - GOAL_STATE).dot(x - GOAL_STATE)

    return J

def gradient_descent(U: jnp.ndarray, learning_rate: float, num_steps: int):
    """Perform gradient descent on the cost function J(U).

    Args:
        U: the initial sequence of actions
        learning_rate: the step size for gradient descent
        num_steps: the number of gradient descent steps to take

    Returns:
        the optimized sequence of actions
    """
    jit_grad = jax.jit(jax.value_and_grad(cost))
    for i in range(num_steps):
        J, dJ_dU = jit_grad(U)
        U = U - learning_rate * dJ_dU
        if i % 100 == 0:
            print(f"Step {i}, cost {J}, grad {jnp.linalg.norm(dJ_dU)}")
    return U

def plot_scenario():
    """Make a pretty plot of the task."""
    # Make a contour plot of the obstacle avoidance cost
    x = jnp.linspace(-3, 3, 100)
    y = jnp.linspace(-3, 3, 100)
    X, Y = jnp.meshgrid(x, y)
    Z = jax.vmap(jax.vmap(obstacle_avoidance_cost))(jnp.stack([X, Y], axis=-1))
    plt.contourf(X, Y, Z, cmap="Reds", levels=100)
    #plt.colorbar()
    plt.xlabel("px")
    plt.ylabel("py")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Plot a green star at the goal position
    plt.plot(*GOAL_STATE, "g*", markersize=20)

def plot_trajectory(U, color="blue", alpha=1.0):
    """Plot the trajectory given by the sequence of actions U."""
    X = jnp.cumsum(U, axis=0) + START_STATE
    X = jnp.concatenate([START_STATE.reshape(1, 2), X], axis=0)
    return plt.plot(X[:,0], X[:,1], "o-", color=color, alpha=alpha)

def animate_diffusion_process(samples):
    """Make an animation of the diffusion process."""
    fig, ax = plt.subplots()
    plot_scenario()
    traj = plot_trajectory(samples[0][0], alpha=1.0)

    num_iterations = len(samples)
    num_samples = len(samples[0])

    def update(idx):
        i, j = divmod(idx, num_samples)
        i = i % num_samples
        X = jnp.cumsum(samples[j][i], axis=0) + START_STATE
        X = jnp.concatenate([START_STATE.reshape(1, 2), X], axis=0)
        traj[0].set_data(X[:,0], X[:,1])
        plt.title(f"Seed {i}, Iteration {j}")

    ani = animation.FuncAnimation(
        fig, update, frames=num_samples*num_iterations, interval=100)
    
    # Save the animation to a file
    ani.save("diffusion.mp4", writer="ffmpeg", fps=10)
    plt.show()


def solve_with_gradient_descent():
    """Solve the obstacle avoidance problem using gradient descent."""
    U_guess = jnp.zeros((20, 2))
    U_opt = gradient_descent(U_guess, 0.01, 5000)

    print(f"Initial cost: {cost(U_guess)}")
    print(f"Optimized cost: {cost(U_opt)}")

    plot_scenario()
    plot_trajectory(U_opt)
    plt.show()

def approximate_score_function(
        U: jnp.ndarray, 
        sigma: float, 
        lmbda: float,
        num_samples: int, 
        rng: jax.random.PRNGKey):
    """Approximate the score function ∇_U log p_σ(U) using MPPI-style sampling.

    Args:
        U: the sequence of actions U = [u_0, u_1, ..., u_{T-1}]. Each u_t is a
           2D vector representing the change in position at time t.
        sigma: the noise level for the samples.
        lmbda: the temperature of the energy distribution p(U).
        num_samples: the number of samples to use for the approximation.
        rng: the random number generator key.

    Returns:
        the approximate score function ∇_U log p_σ(U)
    """
    deltas = jax.random.normal(rng, (num_samples,) + U.shape)
    U_new = U + sigma * deltas
    costs = jax.vmap(cost)(U_new)
    min_cost = jnp.min(costs)
    weights = jnp.exp(-1/lmbda * (costs - min_cost))
    weights = weights / jnp.sum(weights)

    return jnp.sum(weights[:, None, None] * deltas, axis=0)

def solve_with_mppi():
    """Solve the trajectory optimization using MPPI-style sampling."""
    rng = jax.random.PRNGKey(0)
    sigma = 0.01
    lmbda = 0.5
    num_samples = 64

    plot_scenario()
    U = jnp.zeros((20, 2))

    N = 100
    for i in range(N):
        rng, score_rng = jax.random.split(rng)
        s = approximate_score_function(U, sigma, lmbda, num_samples, score_rng)
        U = U + 0.1 * s

        if i % 10 == 0:
            plot_trajectory(U, alpha=i/N)
    plot_trajectory(U, color="black")

    plt.show()



def do_langevin_sampling(
        samples: jnp.ndarray,
        learning_rate: float,
        num_iterations: int,
        num_mppi_samples: int,
        sigma: float,
        lmbda: float,
        rng: jax.random.PRNGKey,
):
    """Sample from the distribution p_σ(U) using Langevin dynamics.

    That is, sample 
        U ~ p_σ(U) = ∫ p(U')N(U; U', σ²I)dU'.
    
    These samples can be viewed as noised version of samples from the target
    distribution p(U) = exp(-J(U)/λ), e.g., 
        U = U' + sigma * Z,
    where Z ~ N(0, I) and U' ~ p(U').

    We will approximate the score function ∇_U log p_σ(U) using MPPI-style
    sampling.

    Args: 
        samples: the initial samples U_j that we will transform.
        learning_rate: the step size for Langevin dynamics.
        num_iterations: the number of Langevin dynamics steps to take.
        num_mppi_samples: the number of samples for estimating the score.
        sigma: the noise level for the samples.
        lmbda: the temperature of the energy distribution p(U).
        rng: the random number generator key.
    """

    def _langevin_step(samples, rng):
        # Compute the score functions
        rng, score_rng = jax.random.split(rng)
        scores = jax.vmap(approximate_score_function, in_axes=(0, None, None, None, None))(
            samples, sigma, lmbda, num_mppi_samples, score_rng)
        
        # Sample some random noise
        rng, noise_rng = jax.random.split(rng)
        z = jax.random.normal(noise_rng, samples.shape)

        # Update the samples using Langevin dynamics
        samples = samples + learning_rate * scores + jnp.sqrt(2 * learning_rate * sigma**2) * z

        return samples, rng

    jit_step = jax.jit(_langevin_step)

    for _ in range(num_iterations):
        samples, rng = jit_step(samples, rng)
    
    return samples

def solve_with_diffusion():
    """Use Langevin sampling dynamics to find a bunch of trajectories."""
    horizon = 20
    num_samples = 20

    num_langevin_iterations = 100
    num_mppi_samples = 128
    sigma = 0.1
    lmbda = 0.001

    learning_rate = 0.01
    outer_iterations = 51
    rescale_rate = 0.9

    # Sample some control tapes from a Gaussian distribution
    rng = jax.random.PRNGKey(0)
    rng, init_samples_rng = jax.random.split(rng)
    U = jax.random.normal(init_samples_rng, (num_samples, horizon, 2))

    # Set up a figure to show intermediate solutions
    plt.figure()
    plot_iterations = [0, 3, 5, 10, 30, 50]

    # Use Langevin sampling to refine the samples 
    all_samples = [U]
    jit_cost = jax.jit(jax.vmap(cost))
    for i in range(outer_iterations):
        # Make pretty plots
        if i in plot_iterations:
            ax_idx = plot_iterations.index(i)
            plt.subplot(2, 3, ax_idx+1)
            plot_scenario()
            for j in range(num_samples):
                plot_trajectory(U[j], alpha=0.2)
            plt.title(f"Iteration {i}")

        # Print some stats 
        if i % 5 == 0:
            J = jit_cost(U)
            print(f"  Iteration: {i}, cost: {jnp.mean(J):.4f}, std: {jnp.std(J):.4f}, sigma: {sigma:.4f}")

        # Do the actual update step 
        rng, langevin_rng = jax.random.split(rng)
        U = do_langevin_sampling(
            U, learning_rate, num_langevin_iterations, num_mppi_samples, sigma, lmbda, langevin_rng)

        sigma *= rescale_rate
        all_samples.append(np.asarray(U))

    plt.show()

    return all_samples

def importance_sampling_diffusion():
    """Use importance sampling to estimate several scores using the same data."""
    rng = jax.random.PRNGKey(0)

    # Parameters for dataset generation
    horizon = 20          # The length of the control tape
    lmbda = 0.1           # The temperature of the energy distribution p(U)
    num_rollouts = 1024   # Number of samples from proposal distribution q(U)
    sigma_L = 0.05        # Std deviation of proposal distribution q(U)
    iterations = 101      # Number of times to update the proposal distribution

    # Parameters for Langevin sampling
    num_langevin_samples = 5
    langevin_iterations = 2000
    alpha = 0.1

    def do_rollouts(rng):
        """Roll out samples from the proposal distribution q(U)"""
        # Guess a control tape
        rng, init_samples_rng = jax.random.split(rng)
        mu = 0.01 * jax.random.normal(init_samples_rng, (horizon, 2))
   
        # Roll out samples from the proposal distribution q(U)
        rng, rollout_rng = jax.random.split(rng)
        U = mu + sigma_L * jax.random.normal(rollout_rng, (num_rollouts, horizon, 2))

        jit_cost = jax.jit(jax.vmap(cost))
        for i in range(iterations):
            # Compute the cost of each trajectory
            J = jit_cost(U)

            # Compute MPPI weights
            best = jnp.argmin(J)
            expJ = jnp.exp(-1/lmbda*(J - J[best]))

            # Update the proposal distribution (MPPI-style)
            weights = expJ / jnp.sum(expJ)
            mu = jnp.sum(weights[:, None, None] * U, axis=0)

            # Roll out new samples
            rng, rollout_rng = jax.random.split(rng)
            U = mu + sigma_L * jax.random.normal(rollout_rng, (num_rollouts, horizon, 2))

            # Print some stats 
            if i % 10 == 0:
                print(f"Iteration {i}, best cost: {J[best]}")
    
        # Compute PDF values q(U) for the proposal distribution q(U) = N(μ, σ_L²)
        U_flat = U.reshape(-1, horizon * 2)
        mu_flat = mu.reshape(horizon * 2)
        q = jax.scipy.stats.multivariate_normal.pdf(U_flat, mu_flat, sigma_L)

        return U, expJ, q

    rng = jax.random.PRNGKey(0) 
    U, expJ, q = do_rollouts(rng)
    rng = jax.random.PRNGKey(2)
    U2, expJ2, q2 = do_rollouts(rng)

    # Combine both sets of samples
    U = jnp.concatenate([U, U2], axis=0)
    expJ = jnp.concatenate([expJ, expJ2], axis=0)
    q = jnp.concatenate([q, q2], axis=0)

    U_flat = U.reshape(-1, horizon * 2)

    def calc_noised_score_estimate(sample_mu, sample_sigma):
        """Estimate the noised score function ∇_μ log p_σ(μ), 
        where p_σ(μ) ∝ 𝔼_{U ~ N(μ, σ²)}[exp(−J(U)/λ], using data collected
        from a different proposal distribution q(U) above.
        """
        # PDF values for all data points under the desired distribution N(μ, σ²)
        p_des = jax.scipy.stats.multivariate_normal.pdf(
            U_flat, sample_mu.flatten(), sample_sigma)
        
        # Importance sampling ratio p(U) / q(U)
        ratio = p_des / q

        # Score approximation
        # N.B. this is actually the score times sigma²
        weights = expJ * ratio
        weights = weights / (jnp.sum(weights) + 1e-6)  # Avoid division by zero
        score = jnp.sum(weights[:, None, None] * (U - sample_mu), axis=0)
        return score

    # Do Langevin sampling using the estimated score function
    print("")
    jit_score_estimate = jax.jit(calc_noised_score_estimate)

    plt.figure()
    plt.title("Langevin sampling")
    plot_scenario()

    for j in range(num_langevin_samples):
        rng, init_rng = jax.random.split(rng)
        U_sample = 0.1 * jax.random.normal(init_rng, (horizon, 2))
        sigma = sigma_L
        for i in range(langevin_iterations):
            rng, langevin_rng = jax.random.split(rng)
            eps = jax.random.normal(langevin_rng, (horizon, 2))
            U_sample = U_sample + alpha * jit_score_estimate(U_sample, sigma) + jnp.sqrt(2 * alpha * sigma**2) * eps

            if i % 100 == 0:
                print(f"Iteration {i}, cost: {cost(U_sample)}, sigma: {sigma}")
                sigma *= 0.9  # Rescale the noise for annealed Langevin dynamics

        plot_trajectory(U_sample, color="black")

    plt.figure()
    plt.title("Samples from proposal distribution q(U)")
    plot_scenario()
    for j in range(U.shape[0]):
        plot_trajectory(U[j], alpha=0.02)
    plt.show()


if __name__=="__main__":
    importance_sampling_diffusion()

    # samples = solve_with_diffusion()
    # animate_diffusion_process(samples)

    # solve_with_gradient_descent()
    # solve_with_mppi()

