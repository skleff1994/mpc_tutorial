import numpy as np
import matplotlib.pyplot as plt

# Create the running models
DT = 5e-2
G = 9.81
L = 1.
def pendulum_dynamics(th, thd, u):
    # acceleration
    thddot = -G * np.sin(th) / L + u/L
    # euler integration
    thd_next = thd + thddot*DT
    th_next = th + thd_next*DT
    return th_next, thd_next

def running_cost(th, thd, u):
    # return 1. * np.sin(th)**2 + 1. *(1-np.cos(th)) **2 + 0.1 * thd**2 + 0.001 * u**2
    return 0.001 * u**2

def terminal_cost(th, thd):
    # return 1. * np.sin(th)**2 + 1. *(1-np.cos(th)) **2 + 0.1 * thd**2 
    return np.linalg.norm(np.array([th, thd])) 

def find_nearest_index(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

# Discretrize state space and control space
NX1 = 101 ; X1_MIN = -np.pi ; X1_MAX = np.pi
X1_SPACE = np.linspace(X1_MIN, X1_MAX, NX1)
NX2 = 101 ; X2_MIN = -10 ; X2_MAX = 10
X2_SPACE = np.linspace(X2_MIN, X2_MAX, NX2)
NU  = 51 ; U_MIN = -20 ; U_MAX = 20
U_SPACE  = np.linspace(U_MIN, U_MAX, NU)
MAXITER = 10
# i_target = find_nearest_index(0., X1_SPACE)
# j_target = find_nearest_index(0., X2_SPACE)
# cost_to_go = np.full((NX1, NX2), 1e12)
# cost_to_go[i_target, j_target] = 0  # Goal state: upright position
cost_to_go = np.zeros((NX1, NX2))
# cost_to_go = np.full((NX1, NX2), 0)
# Set final state costs (e.g., zero if you reach the goal)
for i,x1 in enumerate(X1_SPACE):
    for j,x2 in enumerate(X2_SPACE):
        cost_to_go[i, j] = terminal_cost(x1, x2)
for iter in range(MAXITER):
    print("iter ", iter)
    new_cost_to_go = np.copy(cost_to_go)
    # For each state
    for i, x1 in enumerate(X1_SPACE):
        for j,x2 in enumerate(X2_SPACE):
            # Skip goal state
            if cost_to_go[i, j] == 0:
                print("skipping goal state")
                continue
            # Compute optimal cost using Bellman principle
            min_cost = np.inf
            for k,u in enumerate(U_SPACE):
                x1_next, x2_next = pendulum_dynamics(x1, x2, u)
                # print(x1_next, x2_next)
                # J(x,u) = l(x,u) + V(f(x,u))
                i_next = find_nearest_index(x1_next, X1_SPACE)
                j_next = find_nearest_index(x2_next, X2_SPACE)
                cost = running_cost(x1, x2, u) + cost_to_go[i_next, j_next] 
                if(cost < min_cost):
                    min_cost = cost
            # Update cost to go with best cost
            new_cost_to_go[i,j] = min(min_cost, new_cost_to_go[i, j])
    # Convergence check
    if np.max(np.abs(new_cost_to_go - cost_to_go)) < 1e-3:
        print(f"Converged after {iter} iterations")
        break
    cost_to_go = new_cost_to_go

# Plot the final cost-to-go function
plt.figure()
plt.imshow(cost_to_go, origin='lower', extent=[X1_MIN, X1_MAX, X2_MIN, X2_MAX], aspect='auto')
plt.colorbar(label='Value function')
plt.xlabel('Theta')
plt.ylabel('Theta dot')
plt.title('Value function for Simple Pendulum')
# plt.show()

COMPUTE_POLICY = True

if(COMPUTE_POLICY):
    # Initialize policy array (same shape as value function)
    optimal_policy = np.zeros((NX1, NX2))
    # Function to retrieve optimal policy
    print("Computing optimal policy")
    for i, x1 in enumerate(X1_SPACE):
        for j, x2 in enumerate(X2_SPACE):
            min_cost = np.inf
            best_u = 0
            for u in U_SPACE:
                # Compute next state based on dynamics
                x1_next, x2_next = pendulum_dynamics(x1, x2, u)
                # Find nearest indices in the discretized state space
                i_next = find_nearest_index(x1_next, X1_SPACE)
                j_next = find_nearest_index(x2_next, X2_SPACE)
                # Compute cost based on Bellman equation
                cost = running_cost(x1, x2, u) + cost_to_go[i_next, j_next]
                # Update the optimal control input if we find a lower cost
                if cost < min_cost:
                    min_cost = cost
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
    print("State ", x)
    # Compute optimal control (online)
    if(COMPUTE_POLICY):
        ix = find_nearest_index(x[0], X1_SPACE)
        jx = find_nearest_index(x[1], X2_SPACE)
        u = optimal_policy[ix, jx]
        print("offline ctrl = ", u)
    else:
        min_cost = np.inf
        for u in U_SPACE:
            # Compute next state based on dynamics
            x1_next, x2_next = pendulum_dynamics(x[0], x[1], u)
            # Find nearest indices in the discretized state space
            i_next = find_nearest_index(x1_next, X1_SPACE)
            j_next = find_nearest_index(x2_next, X2_SPACE)
            # Compute cost based on Bellman equation
            cost = running_cost(x[0], x[1], u) + cost_to_go[i_next, j_next]
            # Update the optimal control input if we find a lower cost
            if cost < min_cost:
                min_cost = cost
                best_u = u
        u = best_u.copy()
        print("online ctrl = ", u)

    # record control + simulate + record state
    u_traj[t] = u.copy()
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