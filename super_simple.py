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
START_STATE = jnp.array([-0.1, -1.5])
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
        J += 0.01 * obstacle_avoidance_cost(x)
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
    x = jnp.linspace(-2, 2, 100)
    y = jnp.linspace(-2, 2, 100)
    X, Y = jnp.meshgrid(x, y)
    Z = jax.vmap(jax.vmap(obstacle_avoidance_cost))(jnp.stack([X, Y], axis=-1))
    plt.contourf(X, Y, Z, cmap="Reds", levels=100)
    plt.colorbar()
    plt.xlabel("px")
    plt.ylabel("py")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    # Plot a green star at the goal position
    plt.plot(*GOAL_STATE, "g*", markersize=20)

def plot_trajectory(U):
    """Plot the trajectory given by the sequence of actions U."""
    X = jnp.cumsum(U, axis=0) + START_STATE
    X = jnp.concatenate([START_STATE.reshape(1, 2), X], axis=0)
    plt.plot(X[:,0], X[:,1], "bo")

if __name__=="__main__":
    plot_scenario()

    U_guess = jnp.zeros((20, 2))
    U_opt = gradient_descent(U_guess, 0.01, 5000)

    print(f"Initial cost: {cost(U_guess)}")
    print(f"Optimized cost: {cost(U_opt)}")

    plot_trajectory(U_opt)

    plt.show()


    