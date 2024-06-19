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

# Scenario definition variables
OBSTACLE_POSITION = jnp.array([0.0, 0.0])
START_STATE = jnp.array([0.1, -1.5])
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
        J += 0.01 * u.dot(u)
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
    plt.colorbar()
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
    plt.plot(X[:,0], X[:,1], "o-", color=color, alpha=alpha)

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

def do_mppi():
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


if __name__=="__main__":
    # solve_with_gradient_descent()
    # do_mppi()

    horizon = 20
    num_samples = 10

    num_langevin_iterations = 100
    num_mppi_samples = 1024
    sigma = 1.0
    lmbda = 0.1

    # Sample some control tapes from a Gaussian distribution
    rng = jax.random.PRNGKey(0)
    rng, init_samples_rng = jax.random.split(rng)
    U = 1.0*jax.random.normal(init_samples_rng, (num_samples, horizon, 2))

    # Use Langevin sampling to refine the samples 
    jit_cost = jax.jit(jax.vmap(cost))
    for i in range(100):
        print("Sampling with sigma =", sigma)
        learning_rate = 0.1
        rng, langevin_rng = jax.random.split(rng)
        U = do_langevin_sampling(
            U, learning_rate, num_langevin_iterations, num_mppi_samples, sigma, lmbda, langevin_rng)
        J = jit_cost(U)

        print(f"  Cost: {jnp.mean(J)}, std: {jnp.std(J)}")

        sigma *= 0.95

    plot_scenario()
    for i in range(num_samples):
        plot_trajectory(U[i], alpha=0.2)
    plt.show()
