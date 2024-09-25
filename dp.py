import numpy as np
import matplotlib.pyplot as plt

# Create the running models
DT = 1e-2
G = 9.81
L = 1.
def pendulum_dynamics(th, thd, u):
    # acceleration
    thddot = -G * np.sin(th) / L + u/L
    # euler integration
    thd_next = thd + thddot*DT
    th_next = th + thd_next*DT
    return th_next, thd_next

COST_WEIGHTS = [1, 1, 1e-2, 1e-3]
def running_cost(th, thd, u):
    cost = COST_WEIGHTS[0] * (1 - np.cos(th))  # cos(th)
    cost += COST_WEIGHTS[1] * (np.sin(thd))    # sin(th)
    cost += COST_WEIGHTS[2] * thd ** 2         # thdot
    cost += COST_WEIGHTS[3] * u ** 2           # u
    return 0.5*cost

def find_nearest_index(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

# Discretrize state space and control space
NX1 = 51 ; X1_MIN = 0 ; X1_MAX = 2*np.pi
X1_SPACE = np.linspace(X1_MIN, X1_MAX, NX1)
NX2 = 31 ; X2_MIN = -1 ; X2_MAX = 1
X2_SPACE = np.linspace(X2_MIN, X2_MAX, NX2)
NU  = 31 ; U_MIN = -5 ; U_MAX = 5
U_SPACE  = np.linspace(U_MIN, U_MAX, NU)
MAXITER = 10
i_target = find_nearest_index(0., X1_SPACE)
j_target = find_nearest_index(0., X2_SPACE)
cost_to_go = np.full((NX1, NX2), 10)
cost_to_go[i_target, j_target] = 0  # Goal state: upright position
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
                cost = + running_cost(x1, x2, u) + cost_to_go[i_next, j_next] 
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
plt.imshow(cost_to_go, origin='lower', extent=[X1_MIN, X1_MAX, X2_MIN, X2_MAX], aspect='auto')
plt.colorbar(label='Value function')
plt.xlabel('Theta')
plt.ylabel('Theta dot')
plt.title('Value function for Simple Pendulum')
plt.show()