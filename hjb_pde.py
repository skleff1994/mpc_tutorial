import numpy as np
import matplotlib.pyplot as plt

# Create the running models
DT = 1e-3
G = 9.81
L = 1.
def pendulum_dynamics(th, thd, u):
    # acceleration
    thddot = -G * np.sin(th) / L + u/L
    return thd, thddot

def running_cost(th, thd, u):
    return 1. * np.sin(th)**2 + 1. *(1-np.cos(th)) **2 + 0.1 * thd**2 + 0.001 * u**2

def terminal_cost(th, thd):
    return 1. * np.sin(th)**2 + 1. *(1-np.cos(th)) **2 + 0.1 * thd**2 

def find_nearest_index(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

# Discretrize state space and control space
NX1 = 51 ; X1_MIN = -np.pi ; X1_MAX = np.pi
X1_SPACE = np.linspace(X1_MIN, X1_MAX, NX1)
NX2 = 51 ; X2_MIN = -5 ; X2_MAX = 5
X2_SPACE = np.linspace(X2_MIN, X2_MAX, NX2)
NU  = 21 ; U_MIN = -1 ; U_MAX = 1
U_SPACE  = np.linspace(U_MIN, U_MAX, NU)
MAXITER = 10
V = np.zeros((NX1, NX2))
# Set final state costs (e.g., zero if you reach the goal)
for i,x1 in enumerate(X1_SPACE):
    for j,x2 in enumerate(X2_SPACE):
        V[i, j] = terminal_cost(x1, x2)
# Finite difference method for derivatives
eps_x1 = (X1_MAX - X1_MIN)/NX1
eps_x2 = (X2_MAX - X2_MIN)/NX2
def finite_difference(V, i, j, x1_vals, x2_vals):
    # i_eps = find_nearest_index(x1+eps_x1, X1_SPACE)
    # j_eps = find_nearest_index(x2+eps_x2, X1_SPACE)
    # dV_dx1 = (V[i_eps, j] - V[i, j]) / eps_x1
    # dV_dx2 = (V[i, j_eps] - V[i, j]) / eps_x2
    dV_dx1 = (V[min(i + 1, NX1 - 1), j] - V[max(i - 1, 0), j]) / (x1_vals[min(i + 1, NX1 - 1)] - x1_vals[max(i - 1, 0)])
    dV_dx2 = (V[i, min(j + 1, NX2 - 1)] - V[i, max(j - 1, 0)]) / (x2_vals[min(j + 1, NX2 - 1)] - x2_vals[max(j - 1, 0)])
    # assert np.linalg.norm(dV_dx1) + np.linalg.norm(dV_dx2) < 1000
    return dV_dx1, dV_dx2


# V[i(x1+d),j] - V[i(x1),j] / d

for iter in range(MAXITER):
    print("iter ", iter)
    V_new = np.copy(V)
    # For each state
    for i,x1 in enumerate(X1_SPACE):
        for j,x2 in enumerate(X2_SPACE):
            # Skip goal state
            if V[i, j] == 0:
                print("skipping goal state")
                continue
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

COMPUTE_POLICY = True

if(COMPUTE_POLICY):
    # Initialize policy array (same shape as value function)
    optimal_policy = np.zeros((NX1, NX2))
    # Function to retrieve optimal policy
    for i, x1 in enumerate(X1_SPACE):
        for j, x2 in enumerate(X2_SPACE):
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
            # Store the best control input at state (x1, x2)
            optimal_policy[i, j] = best_u

    # Plot the optimal feedback policy
    plt.figure()
    plt.imshow(optimal_policy, origin='lower', extent=[X1_MIN, X1_MAX, X2_MIN, X2_MAX], aspect='auto')
    plt.colorbar(label='Optimal Control Input u(th, thd)')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Theta dot (rad/s)')
    plt.title('Optimal Feedback Policy from DP')
    plt.show()

x = np.array([1.,0])
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