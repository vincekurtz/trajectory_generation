#!/usr/bin/env python

##
#
# A toy exmaple for exploring generative models for trajectory planning. A robot
# with double integrator dynamics should go around an obstacle. 
#
##

import numpy as np
from pydrake.all import *
import matplotlib.pyplot as plt

class ObstacleAvoidanceProblem:
    def __init__(self, obstacle_position=[0, 0], obstacle_radius=1.0,
            target_position=[0, 2], num_steps=20, time_step=1e-2):
        """
        A simple obstacle avoidance problem for a robot with double integrator
        dynamics.

        Args:
            obstacle_position: position of the center of the obstacle
            obstacle_radius:   size of the (spherical) obstacle
            target_position:   the robot's goal
            num_steps:         number of time steps to optimize over
            time_step:         time step for discretization
        """
        self.obstacle_position = obstacle_position
        self.obstacle_radius = obstacle_radius
        self.target_position = target_position
        self.num_steps = num_steps

        # Set up a Drake MathematicalProgram for the optimization problem
        self.mp = MathematicalProgram()
        q = self.mp.NewContinuousVariables(2, num_steps, "q")
        v = np.empty((2, num_steps), dtype=object)
        a = np.empty((2, num_steps-1), dtype=object)   # accelerations are the control input

        v[:,0] = np.array([0.0, 0.0])  # assume (for now) initial velocity = 0
        for t in range(1, num_steps):
            v[:,t] = (q[:,t] - q[:,t-1]) / time_step
        
        for t in range(num_steps - 1):
            # Accelerations depend on q[t-1], q[t], q[t+1]
            a[:,t] = (v[:,t+1] - v[:,t]) / time_step

        # Initial condition constraint: we'll replace this later
        self.initial_condition = self.mp.AddLinearEqualityConstraint(
                np.eye(2), np.zeros(2), q[:,0])
       
        # Cost function
        accel_weight = 1e-4
        velocity_weight = 1e-4
        for t in range(num_steps - 1):
            self.mp.AddQuadraticCost(accel_weight * a[:,t].dot(a[:,t]))
        for t in range(num_steps):
            self.mp.AddQuadraticCost(velocity_weight * v[:,t].dot(v[:,t]))

        # Obstacle avoidance constraint
        for t in range(num_steps):
            squared_distance_to_obstacle = (q[:,t] -
                    obstacle_position).dot(q[:,t] - obstacle_position)
            self.mp.AddConstraint(
                    squared_distance_to_obstacle >= obstacle_radius**2)

        # Hard constraint on final position
        final_err = q[:,num_steps-1] - np.asarray(target_position)
        self.mp.AddConstraint(eq(final_err, 0))

        # Hard constraint on final velocity
        self.mp.AddConstraint(eq(v[:,num_steps-1], 0))

        self.q = q  # allows us to extract the solution later

    def Solve(self, q0, initial_guess):
        """
        Solve the optimization problem from the given initial condition and
        initial guess. 

        Args:
            q0: initial position of the robot, size 2
            initial_guess: guess for the position trajectory, size 2xnum_steps
        """
        assert len(q0) == 2
        assert initial_guess.shape == (2, self.num_steps)

        self.initial_condition.evaluator().UpdateCoefficients(
                Aeq=np.eye(2), beq=q0)

        self.mp.SetInitialGuess(self.q, initial_guess)

        solver = SnoptSolver()
        res = solver.Solve(self.mp)
        print(f"Solved with {res.get_solver_id().name()}")
        if not res.is_success():
            print(f"Solver failed! Getting the solution anyway")

        print(f"Cost: {res.get_optimal_cost()}")

        return res.GetSolution(self.q), res.get_optimal_cost()
        

    def plot_scenario(self):
        """
        Make a matplotlib plot of the obstacle and target position
        """
        ax = plt.gca()
        ax.axis("equal")
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))

        plt.plot(*self.target_position, "g*", markersize=20)
        obstacle = plt.Circle(self.obstacle_position, self.obstacle_radius, color="red")
        ax.add_patch(obstacle)

    def plot_trajectory(self, position_sequence, **kwargs):
        """
        Add the given trajectory to the matplotlib plot

        Args:
            position_sequence: robot positions, array of size 2xnum_steps
        """
        assert position_sequence.shape == (2, self.num_steps)
        plt.plot(position_sequence[0,:], position_sequence[1,:], **kwargs)


if __name__=="__main__":
    # Set up the scenario
    prob = ObstacleAvoidanceProblem()
 
    # Solve from a particular initial guess
    q0 = np.array([0.5, -2])
    guess = np.array([q0 + [-0.2*t, 0.2*t] for t in range(20)]).T
    traj, cost = prob.Solve(q0, guess)

    # Plot the solution
    prob.plot_scenario()
    prob.plot_trajectory(guess, color="b", marker="o", alpha=0.2, label="guess")
    prob.plot_trajectory(traj, color="b", marker="o", label="solution")
    plt.legend()

    plt.show()

