# import numpy as np
# import matplotlib.pyplot as plt

# Create the running models
DT = 1e-3
G = 9.81
L = 1.
def pendulum_dynamics(th, thd, u):
    # acceleration
    thddot = -G * np.sin(th) / L + u/L
    return thd, thddot

# def running_cost(th, thd, u):
#     return 0.1 * np.sin(th)**2 + 0.01 * thd**2 + 0.1 * u**2
#     # return 1. * np.sin(th)**2 + 1. *(1-np.cos(th)) **2 + 0.1 * thd**2 + 0.001 * u**2
#     # return 0.001 * u**2

# def terminal_cost(th, thd):
#     return 1. * np.sin(th)**2 + 1. *(1-np.cos(th)) **2 + 0.1 * thd**2 

def find_nearest_index(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

# # Discretrize state space and control space
# # NX1 = 51 ; X1_MIN = -np.pi ; X1_MAX = np.pi
# # X1_SPACE = np.linspace(X1_MIN, X1_MAX, NX1)
# # NX2 = 51 ; X2_MIN = -5 ; X2_MAX = 5
# # X2_SPACE = np.linspace(X2_MIN, X2_MAX, NX2)
# # NU  = 21 ; U_MIN = -1 ; U_MAX = 1
# # U_SPACE  = np.linspace(U_MIN, U_MAX, NU)
# # MAXITER = 10
# # Discretrize state space and control space
# NX1 = 101 ; X1_MIN = -np.pi ; X1_MAX = np.pi
# X1_SPACE = np.linspace(X1_MIN, X1_MAX, NX1)
# NX2 = 101 ; X2_MIN = -10 ; X2_MAX = 10
# X2_SPACE = np.linspace(X2_MIN, X2_MAX, NX2)
# NU  = 101 ; U_MIN = -2 ; U_MAX = 2
# U_SPACE  = np.linspace(U_MIN, U_MAX, NU)
# MAXITER = 10
# V = np.zeros((NX1, NX2))
# # Set final state costs (e.g., zero if you reach the goal)
# for i,x1 in enumerate(X1_SPACE):
#     for j,x2 in enumerate(X2_SPACE):
#         V[i, j] = terminal_cost(x1, x2)
# # Finite difference method for derivatives
# eps_x1 = (X1_MAX - X1_MIN)/NX1
# eps_x2 = (X2_MAX - X2_MIN)/NX2
# def finite_difference(V, i, j, x1_vals, x2_vals):
#     # i_eps = find_nearest_index(x1+eps_x1, X1_SPACE)
#     # j_eps = find_nearest_index(x2+eps_x2, X1_SPACE)
#     dV_dx1 = (V[min(i + 1, NX1 - 1), j] - V[max(i - 1, 0), j])  / (2*eps_x1)
#     dV_dx2 = (V[i, min(j + 1, NX2 - 1)] - V[i, max(j - 1, 0)])  / (2*eps_x2)
#     # dV_dx1 = (V[min(i + 1, NX1 - 1), j] - V[max(i - 1, 0), j]) / (x1_vals[min(i + 1, NX1 - 1)] - x1_vals[max(i - 1, 0)])
#     # dV_dx2 = (V[i, min(j + 1, NX2 - 1)] - V[i, max(j - 1, 0)]) / (x2_vals[min(j + 1, NX2 - 1)] - x2_vals[max(j - 1, 0)])
#     # assert np.linalg.norm(dV_dx1) + np.linalg.norm(dV_dx2) < 1000
#     return dV_dx1, dV_dx2
# # def finite_difference(V, i, j, x1_vals, x2_vals):
# #     dV_dx1 = (V[min(i+1, NX1-1), j] - V[max(i-1, 0), j]) / (2*eps_x1)
# #     dV_dx2 = (V[i, min(j+1, NX2-1)] - V[i, max(j-1, 0)]) / (2*eps_x2)
# #     # Apply slight averaging to reduce noise
# #     dV_dx1 = 0.5 * (dV_dx1 + (V[i, j] - V[max(i-1, 0), j]) / (eps_x1))
# #     dV_dx2 = 0.5 * (dV_dx2 + (V[i, j] - V[i, max(j-1, 0)]) / (eps_x2))
# #     return dV_dx1, dV_dx2

# # V[i(x1+d),j] - V[i(x1),j] / d

# for iter in range(MAXITER):
#     print("iter ", iter)
#     V_new = np.copy(V)
#     # For each state
#     for i,x1 in enumerate(X1_SPACE):
#         for j,x2 in enumerate(X2_SPACE):
#             # Skip goal state
#             if V[i, j] == 0:
#                 print("skipping goal state")
#                 continue
#             # Compute optimal cost using Bellman principle
#             min_value = np.inf
#             for k,u in enumerate(U_SPACE):
#                 # Compute finite differences for partial derivatives
#                 dV_dx1, dV_dx2 = finite_difference(V, i, j, X1_SPACE, X2_SPACE)
#                 # Compute dynamics
#                 dx1_dt, dx2_dt = pendulum_dynamics(x1, x2, u)
#                 # Compute HJB residual
#                 value = dV_dx1 * dx1_dt + dV_dx2 * dx2_dt + running_cost(x1, x2, u)
#                 if(value < min_value):
#                     min_value = value
#             # Update value function with minimal cost-to-go
#             V_new[i, j] = V[i, j] - DT * min_value
#     # Convergence check
#     if np.max(np.abs(V_new - V)) < 1e-4:
#         print(f"Converged after {iter} iterations")
#         break
#     V = V_new


# # Plot the final cost-to-go function
# plt.imshow(V, origin='lower', extent=[X1_MIN, X1_MAX, X2_MIN, X2_MAX], aspect='auto')
# plt.colorbar(label='Value function')
# plt.xlabel('Theta')
# plt.ylabel('Theta dot')
# plt.title('Value function for Simple Pendulum')


# # Avoid abrupt changes in control by regularizing the control selection
# def smooth_policy(V, i, j, U_SPACE):
#     min_value = np.inf
#     best_u = 0
#     for u in U_SPACE:
#         dV_dx1, dV_dx2 = finite_difference(V, i, j, X1_SPACE, X2_SPACE)
#         dx1_dt, dx2_dt = pendulum_dynamics(X1_SPACE[i], X2_SPACE[j], u)
#         value = dV_dx1 * dx1_dt + dV_dx2 * dx2_dt + running_cost(X1_SPACE[i], X2_SPACE[j], u)
#         if value < min_value:
#             min_value = value
#             best_u = u
#     return best_u

# COMPUTE_POLICY = True
# if(COMPUTE_POLICY):
#     # Initialize policy array (same shape as value function)
#     optimal_policy = np.zeros((NX1, NX2))
#     # Function to retrieve optimal policy
#     for i, x1 in enumerate(X1_SPACE):
#         for j, x2 in enumerate(X2_SPACE):
#             # Store the best control input at state (x1, x2)
#             optimal_policy[i, j] = smooth_policy(V, i, j, U_SPACE)

#     # Plot the optimal feedback policy
#     plt.figure()
#     plt.imshow(optimal_policy, origin='lower', extent=[X1_MIN, X1_MAX, X2_MIN, X2_MAX], aspect='auto')
#     plt.colorbar(label='Optimal Control Input u(th, thd)')
#     plt.xlabel('Theta (rad)')
#     plt.ylabel('Theta dot (rad/s)')
#     plt.title('Optimal Feedback Policy from DP')
#     plt.show()

# x = np.array([1.,0])
# TSIM = 100
# x_traj = np.zeros((TSIM+1, 2))
# u_traj = np.zeros(TSIM)
# x_traj[0,:] = x
# print("Simulating optimal policy")
# for t in range(TSIM):
#     ix = find_nearest_index(x[0], X1_SPACE)
#     jx = find_nearest_index(x[1], X2_SPACE)
#     print("Measured ", x)
#     u = optimal_policy[ix, jx]
#     u_traj[t] = u.copy()
#     print("Input", u)
#     x = pendulum_dynamics(x[0], x[1], u)
#     x_traj[t+1,:] = x

# # plot trajs
# tlin = np.linspace(0, DT*TSIM, TSIM+1)
# plt.figure('Phase plot')
# plt.plot(x_traj[:,0], x_traj[:,1])
# plt.plot(x_traj[0,0], x_traj[0,1], 'ro')
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# ax1.plot(tlin, x_traj[:,0], label='Th')
# ax2.plot(tlin, x_traj[:,1], label='Th dot')
# ax3.plot(tlin[:-1], u_traj, label='u')
# fig.legend()
# fig.suptitle('Trajectories')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters for the pendulum
g = 9.81  # gravitational acceleration (m/s^2)
l = 1.0   # length of the pendulum (m)
m = 1.0   # mass of the pendulum (kg)
dt = 0.01 # time step for discretization
num_iters = 10 # Number of iterations

# Discretize state space (theta and omega)
theta_min, theta_max = -np.pi, np.pi
omega_min, omega_max = -8, 8
num_theta = 100
num_omega = 100
theta_vals = np.linspace(theta_min, theta_max, num_theta)
omega_vals = np.linspace(omega_min, omega_max, num_omega)
u_vals = np.linspace(-10, 10, 21)  # Control inputs (torque)

# Running cost: quadratic cost on control input
def cost(theta, omega, u):
    return 0.1 * theta**2 + 0.1 * omega**2 + 0.01 * u**2

# Finite difference method for derivatives
def finite_difference(V, i, j, theta_vals, omega_vals):
    dV_dtheta = (V[min(i + 1, num_theta - 1), j] - V[max(i - 1, 0), j]) / (theta_vals[1] - theta_vals[0])
    dV_domega = (V[i, min(j + 1, num_omega - 1)] - V[i, max(j - 1, 0)]) / (omega_vals[1] - omega_vals[0])
    return dV_dtheta, dV_domega

# Initialize value function
V = np.zeros((num_theta, num_omega))

# HJB Iteration loop
for iteration in range(num_iters):
    V_new = np.copy(V)
    print("iter ", iteration)
    for i, theta in enumerate(theta_vals):
        for j, omega in enumerate(omega_vals):
            min_value = np.inf
            
            # Loop over control inputs
            for u in u_vals:
                # Compute finite differences for partial derivatives
                dV_dtheta, dV_domega = finite_difference(V, i, j, theta_vals, omega_vals)
                
                # Compute dynamics
                theta_dot = omega
                omega_dot = - (g / l) * np.sin(theta) + u / (m * l**2)
                
                # Compute the HJB equation
                value = dV_dtheta * theta_dot + dV_domega * omega_dot + cost(theta, omega, u)
                
                # Minimize over control inputs
                if value < min_value:
                    min_value = value
            
            # Update value function with minimal cost-to-go
            V_new[i, j] = V[i, j] - dt * min_value
    
    # Convergence check
    if np.max(np.abs(V_new - V)) < 1e-4:
        print(f"Converged after {iteration} iterations")
        break
    
    V = V_new

# Plot the value function
plt.imshow(V, origin='lower', extent=[theta_min, theta_max, omega_min, omega_max], aspect='auto')
plt.colorbar(label='Value Function V(θ, ω)')
plt.xlabel('Theta (rad)')
plt.ylabel('Omega (rad/s)')
plt.title('Value Function for Simple Pendulum (HJB PDE)')
plt.show()


# Initialize policy array (same shape as value function)
optimal_policy = np.zeros((num_theta, num_omega))

# Function to retrieve optimal policy
for i, theta in enumerate(theta_vals):
    for j, omega in enumerate(omega_vals):
        min_value = np.inf
        best_u = 0
        
        for u in u_vals:
            # Compute finite differences for partial derivatives
            dV_dtheta, dV_domega = finite_difference(V, i, j, theta_vals, omega_vals)
            
            # Compute system dynamics (theta_dot and omega_dot)
            theta_dot = omega
            omega_dot = - (g / l) * np.sin(theta) + (u / (m * l**2))
            
            # Hamiltonian (without control costs)
            value = dV_dtheta * theta_dot + dV_domega * omega_dot + cost(theta, omega, u)
            
            # Minimize the Hamiltonian by finding the best control input
            if value < min_value:
                min_value = value
                best_u = u
        
        # Store the best control input at state (theta, omega)
        optimal_policy[i, j] = best_u

# Plot the optimal feedback policy
plt.imshow(optimal_policy, origin='lower', extent=[theta_min, theta_max, omega_min, omega_max], aspect='auto')
plt.colorbar(label='Optimal Control Input u(θ, ω)')
plt.xlabel('Theta (rad)')
plt.ylabel('Omega (rad/s)')
plt.title('Optimal Feedback Policy from HJB PDE')
plt.show()

x = np.array([1.,0])
TSIM = 100
x_traj = np.zeros((TSIM+1, 2))
u_traj = np.zeros(TSIM)
x_traj[0,:] = x
print("Simulating optimal policy")
for t in range(TSIM):
    ix = find_nearest_index(x[0], theta_vals)
    jx = find_nearest_index(x[1], omega_vals)
    print("Measured ", x)
    u = optimal_policy[ix, jx]
    u_traj[t] = u.copy()
    print("Input", u)
    x = pendulum_dynamics(x[0], x[1], u)
    x_traj[t+1,:] = x

# plot trajs
tlin = np.linspace(0, DT*TSIM, TSIM+1)
plt.figure('Phase plot')
plt.plot(x_traj[:,0], x_traj[:,1])
plt.plot(x_traj[0,0], x_traj[0,1], 'ro')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(tlin, x_traj[:,0], label='Th')
ax2.plot(tlin, x_traj[:,1], label='Th dot')
ax3.plot(tlin[:-1], u_traj, label='u')
fig.legend()
fig.suptitle('Trajectories')
plt.show()
