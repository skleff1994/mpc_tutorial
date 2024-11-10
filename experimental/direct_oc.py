import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import casadi as ca

# Pendulum parameters
G = 9.81
L = 1.0
N = 20   # Number of time steps
DT = 1e-2

# Initial and final conditions
# theta0 = 1.0  # Initial angle
# theta_dot0 = 0.0  # Initial angular velocity
x0 = np.array([1.,0.])
# theta_f = 0.0  # Final angle
# theta_dot_f = 0.0  # Final angular velocity

# Define the pendulum dynamics
def pendulum_dynamics(t, y, u):
    theta, theta_dot = y
    theta_ddot = -G / L * np.sin(theta) + u / L
    return np.array([theta_dot, theta_ddot])

### Direct Single Shooting ###

# Solve the dynamics for a given control trajectory
def simulate_single_shooting(u_traj):
    def dynamics(t, y):
        u = np.interp(t, np.linspace(0, N*DT, len(u_traj)), u_traj)
        return pendulum_dynamics(t, y, u)
    
    sol = integrate.solve_ivp(dynamics, [0, N*DT], x0, t_eval=np.linspace(0, N*DT, N))
    return sol.y.T  # Return the states (theta, theta_dot) over time

# Cost function for single shooting
def cost_single_shooting(u_traj):
    states = simulate_single_shooting(u_traj)
    theta_traj, theta_dot_traj = states[:, 0], states[:, 1]
    control_cost = np.sum(0.001 * u_traj**2) * DT  # Regularize control effort
    return control_cost + np.sum(theta_traj**2 + 0.1 * theta_dot_traj**2) * DT

# Perform optimization using single shooting
def optimize_single_shooting(u_guess):
    result = optimize.minimize(cost_single_shooting, u_guess, method='SLSQP', options={'disp': True})
    u_opt = result.x
    return u_opt, simulate_single_shooting(u_opt)

### Direct Multiple Shooting ###

# Dynamics over small time intervals
def dynamics_multiple_shooting(y, u):
    return integrate.solve_ivp(lambda t, y: pendulum_dynamics(t, y, u), [0, DT], y, method='RK45').y[:, -1]

# Cost function for multiple shooting
def cost_multiple_shooting(vars):
    states = vars[:2*N].reshape((N, 2))
    controls = vars[2*N:]
    cost = 0
    for i in range(N-1):
        cost += np.sum(states[i, 0]**2 + 0.1 * states[i, 1]**2 + 0.001 * controls[i]**2) * DT
    return cost

# Constraints for multiple shooting
def constraints_multiple_shooting(vars):
    states = vars[:2*N].reshape((N, 2))
    controls = vars[2*N:]
    constraints = []
    for i in range(N-1):
        y_next = dynamics_multiple_shooting(states[i], controls[i])
        constraints.append(states[i+1] - y_next)  # Enforce continuity
    return np.concatenate(constraints)

# Perform optimization using multiple shooting
def optimize_multiple_shooting(initial_guess):
    result = optimize.minimize(cost_multiple_shooting, initial_guess, constraints={'type': 'eq', 'fun': constraints_multiple_shooting}, method='SLSQP', options={'disp': True})
    vars_opt = result.x
    states_opt = vars_opt[:2*N].reshape((N, 2))
    controls_opt = vars_opt[2*N:]
    return controls_opt, states_opt

### Direct Collocation ###

def optimize_collocation():
    opti = ca.Opti()

    # Variables
    X = opti.variable(2, N+1)  # State variables (theta, theta_dot)
    U = opti.variable(1, N)  # Control variables
    T_s = np.linspace(0, N*DT, N+1)  # Time discretization

    # Pendulum dynamics in CasADi
    def dynamics_collocation(x, u):
        theta, theta_dot = x[0], x[1]
        theta_ddot = -G/L * ca.sin(theta) + u / L
        return ca.vertcat(theta_dot, theta_ddot)

    # Cost function
    J = 0
    for k in range(N):
        J += ca.sumsqr(X[:,k]) + 0.001 * ca.sumsqr(U[:,k])
        x_next = X[:, k] + DT * dynamics_collocation(X[:, k], U[:, k])
        opti.subject_to(X[:, k+1] == x_next)  # Collocation constraints

    # Boundary conditions
    opti.subject_to(X[:, 0] == ca.vertcat(theta0, theta_dot0))
    opti.subject_to(X[:, -1] == ca.vertcat(theta_f, theta_dot_f))

    # Control constraints
    opti.subject_to(opti.bounded(-2, U, 2))  # Control bounds

    # Solve the problem
    opti.minimize(J)
    opti.solver('ipopt')  # Use IPOPT solver
    sol = opti.solve()

    x_opt = sol.value(X)
    u_opt = sol.value(U)
    
    return u_opt, x_opt

### Main ###
if __name__ == "__main__":
    # Single Shooting
    print("Solving using single shooting")
    u_guess = np.ones(N)
    u_opt_single, states_single = optimize_single_shooting(u_guess)

    # Multiple Shooting
    print("Solving using multiple shooting")
    initial_guess = np.ones(2*N + N)  # State and control guess
    for i in range(N):
        initial_guess[2*i:2*i+2] = x0
    u_opt_multiple, states_multiple = optimize_multiple_shooting(initial_guess)

    # # Collocation
    # print("Solving using collocation")
    # u_opt_collocation, states_collocation = optimize_collocation()

    # Plot results
    t = np.linspace(0, N*DT, N)
    
    # Theta
    plt.figure()
    plt.plot(t, states_single[:, 0], label='Single Shooting - Theta', color='b')
    plt.plot(t[0], states_single[0, 0], 'bo')
    plt.plot(t, states_multiple[:, 0], label='Multiple Shooting - Theta', color='r')
    plt.plot(t[0], states_multiple[0, 0], 'ro')
    # plt.plot(np.linspace(0, T, N+1), states_collocation[0, :], label='Collocation - Theta')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.legend()
    # theta dot
    plt.figure()
    plt.plot(t, states_single[:, 1], label='Single Shooting - theta dot', color='b')
    plt.plot(t[0], states_single[0, 1], 'bo')
    plt.plot(t, states_multiple[:, 1], label='Multiple Shooting - theta dot', color='r')
    plt.plot(t[0], states_multiple[0, 1], 'ro')
    # plt.plot(np.linspace(0, T, N+1), states_collocation[0, :], label='Collocation - Theta')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta dot (rad/s)')
    plt.legend()
    # Phase
    plt.figure()
    plt.plot(states_single[:, 0], states_single[:, 1], label='Single Shooting - Theta', color='b')
    plt.plot(states_single[0, 0], states_single[0, 1], 'bo')
    plt.plot(states_multiple[:, 0], states_multiple[:, 1], label='Multiple Shooting - Theta', color='r')
    plt.plot(states_multiple[0, 0], states_multiple[0, 1], 'ro')
    # plt.plot(np.linspace(0, T, N+1), states_collocation[0, :], label='Collocation - Theta')
    plt.xlabel('Theta (rad/s)')
    plt.ylabel('Omega (rad)')
    plt.legend()
    # Control
    plt.figure()
    plt.plot(t, u_opt_single, label='Single Shooting - Control', color='b')
    plt.plot(t[0], u_opt_single[0], 'bo')
    plt.plot(t, u_opt_multiple, label='Multiple Shooting - Control', color='r')
    plt.plot(t[0], u_opt_multiple[0], 'ro')
    # plt.plot(t, u_opt_collocation, label='Collocation - Control')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input (u)')
    plt.legend()
    plt.show()
