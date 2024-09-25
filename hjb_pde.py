import numpy as np
import matplotlib.pyplot as plt

# Create the running models
DT = 2e-2
G = 9.81
L = 1.
def pendulum_dynamics(th, thd, u):
    # acceleration
    thddot = -G * np.sin(th) / L + u/L
    return thd, thddot

COST_WEIGHTS = [1, 1, 1e-2, 1e-3]
def running_cost(th, thd, u):
    # return 0.1 * th**2 + 0.1 * thd**2 + 0.01 * u**2
    cost = COST_WEIGHTS[0] * (1 - np.cos(th))  # cos(th)
    cost += COST_WEIGHTS[1] * (np.sin(thd))    # sin(th)
    cost += COST_WEIGHTS[2] * thd ** 2         # thdot
    cost += COST_WEIGHTS[3] * u ** 2           # u
    return 0.5*cost

# # Updated cost function with hard state constraint
# def running_cost_with_constraints(th, thd, u):
#     penalty = 1e6  
#     # Add a high cost when the state exceeds the constraints
#     if abs(omega) > 5.:
#         return penalty
#     else:
#         return running_cost(th, thd, u)
    
def find_nearest_index(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

# Discretrize state space and control space
NX1 = 51 ; X1_MIN = -np.pi ; X1_MAX = np.pi
X1_SPACE = np.linspace(X1_MIN, X1_MAX, NX1)
NX2 = 51 ; X2_MIN = -10 ; X2_MAX = 10
X2_SPACE = np.linspace(X2_MIN, X2_MAX, NX2)
NU  = 51 ; U_MIN = -10 ; U_MAX = 10
U_SPACE  = np.linspace(U_MIN, U_MAX, NU)
MAXITER = 10
V = np.zeros((NX1, NX2))
# Finite difference method for derivatives
def finite_difference(V, i, j, x1_vals, x2_vals):
    dV_dx1 = (V[min(i + 1, NX1 - 1), j] - V[max(i - 1, 0), j]) / (x1_vals[1] - x1_vals[0])
    dV_dx2 = (V[i, min(j + 1, NX2 - 1)] - V[i, max(j - 1, 0)]) / (x2_vals[1] - x2_vals[0])
    return dV_dx1, dV_dx2

for iter in range(MAXITER):
    print("iter ", iter)
    V_new = np.copy(V)
    # For each state
    for i, x1 in enumerate(X1_SPACE):
        for j,x2 in enumerate(X2_SPACE):
            # Compute optimal cost using Bellman principle
            min_value = np.inf
            for k,u in enumerate(U_SPACE):
                # Compute finite differences for partial derivatives
                dV_dx1, dV_dx2 = finite_difference(V, i, j, X1_SPACE, X2_SPACE)
                # Compute dynamics
                dx1_dt, dx2_dt = pendulum_dynamics(x1, x2, u)
                # Compute HJB residual
                value = dV_dx1 * dx1_dt + dV_dx2 * dx2_dt + running_cost(x1, x2, u)
                if(value < min_value):
                    min_value = value
            # Update value function with minimal cost-to-go
            V_new[i, j] = V[i, j] - DT * min_value
    # Convergence check
    if np.max(np.abs(V_new - V)) < 1e-4:
        print(f"Converged after {iter} iterations")
        break
    V = V_new


# Plot the final cost-to-go function
plt.imshow(V, origin='lower', extent=[X1_MIN, X1_MAX, X2_MIN, X2_MAX], aspect='auto')
plt.colorbar(label='Value function')
plt.xlabel('Theta')
plt.ylabel('Theta dot')
plt.title('Value function for Simple Pendulum')

# Initialize policy array (same shape as value function)
optimal_policy = np.zeros((NX1, NX2))
# Function to retrieve optimal policy
for i, theta in enumerate(X1_SPACE):
    for j, omega in enumerate(X2_SPACE):
        min_value = np.inf
        best_u = 0
        for k,u in enumerate(U_SPACE):
            # Compute finite differences for partial derivatives
            dV_dx1, dV_dx2 = finite_difference(V, i, j, X1_SPACE, X2_SPACE)
            # Compute dynamics
            dx1_dt, dx2_dt = pendulum_dynamics(x1, x2, u)
            # Compute HJB residual
            value = dV_dx1 * dx1_dt + dV_dx2 * dx2_dt + running_cost(x1, x2, u)
            if(value < min_value):
                min_value = value
                best_u = u
        # Store the best control input at state (theta, omega)
        optimal_policy[i, j] = best_u

# Plot the optimal feedback policy
plt.figure()
plt.imshow(optimal_policy, origin='lower', extent=[X1_MIN, X1_MAX, X2_MIN, X2_MAX], aspect='auto')
plt.colorbar(label='Optimal Control Input u(th, thd)')
plt.xlabel('Theta (rad)')
plt.ylabel('Theta dot (rad/s)')
plt.title('Optimal Feedback Policy from DP')
# plt.show()

x = np.array([0.1,0])
TSIM = 100
x_traj = np.zeros((TSIM+1, 2))
u_traj = np.zeros(TSIM)
x_traj[0,:] = x
print("Simulating optimal policy")
for t in range(TSIM):
    ix = find_nearest_index(x[0], X1_SPACE)
    jx = find_nearest_index(x[1], X2_SPACE)
    print("Measured ", x)
    u = optimal_policy[ix, jx]
    u_traj[t] = u.copy()
    print("Input", u)
    x = pendulum_dynamics(x[0], x[1], u)
    x_traj[t+1,:] = x
tlin = np.linspace(0, DT*TSIM, TSIM+1)
plt.figure('Phase plot')
plt.plot(x_traj[:,0], x_traj[:,1])
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(tlin, x_traj[:,0], label='Th')
ax2.plot(tlin, x_traj[:,1], label='Th dot')
ax3.plot(tlin[:-1], u_traj, label='u')
fig.legend()
fig.suptitle('Trajectories')
plt.show()