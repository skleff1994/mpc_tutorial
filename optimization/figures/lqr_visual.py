import numpy as np
import matplotlib.pyplot as plt

COST_COLOR = '#1E961E'   # Green
VARS_COLOR = '#3C32A0'   # Blue
CSTR_COLOR = '#FF783C'   # Red

CONSTRAINTS = True
TRAJECTORY = False
# Define linear system dynamics (discrete-time for illustration)
A = np.array([[1.0, 0.1],
              [0.0, 1.0]])
B = np.array([[0.0],
              [0.1]])

# Quadratic costs
Q = np.diag([1.0, 0.1])
R = np.array([[0.01]])

# Define grid for state space
x1 = np.linspace(-2, 2, 50)
x2 = np.linspace(-2, 2, 50)
X1, X2 = np.meshgrid(x1, x2)
states = np.stack([X1, X2], axis=-1)

# Value function approximation: V(x) = x^T Q x (stage cost proxy)
V = np.einsum('...i,ij,...j', states, Q, states)

# Compute LQR gain K (discrete Riccati solution)
# For simplicity, solve finite-horizon Riccati iteration (backward for N=100)
N = 50
P = Q.copy()
for _ in range(N):
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    P = Q + A.T @ P @ (A - B @ K)

K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# Compute control policy: u = -Kx, show vector field of closed-loop dynamics
U = np.zeros_like(X1)
Vfield = np.zeros_like(X2)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        u = -K @ x
        dx = A @ x + B.flatten() * u
        U[i, j] = dx[0] - x[0]
        Vfield[i, j] = dx[1] - x[1]

plt.figure(figsize=(8, 10))

# Plot level sets of quadratic cost
CS = plt.contour(X1, X2, V, levels=10, cmap="Greens_r", linewidths=3)
plt.clabel(CS, inline=1, fontsize=8)

# Plot vector field (policy effect)
plt.quiver(X1, X2, U, Vfield, color=VARS_COLOR, alpha=0.6)

if CONSTRAINTS:
    # # Velocity constraint |x2| <= 1
    # plt.fill_between(x1, 1, 2, color="#FF783C", alpha=0.2, label="Infeasible velocity")
    # plt.fill_between(x1, -2, -1, color="#FF783C", alpha=0.2)
    # # Constraint boundary lines
    # plt.axhline(1, color="#FF783C", linestyle="--", linewidth=2, label="Velocity limits")
    # plt.axhline(-1, color="#FF783C", linestyle="--", linewidth=2)

    # Compute unconstrained control u=-Kx over grid
    Uvals = - (K @ np.stack([X1.ravel(), X2.ravel()])).reshape(-1)
    Uvals = Uvals.reshape(X1.shape)
    # Boolean mask for infeasible inputs
    infeasible = np.abs(Uvals) > 5.
    # Shade infeasible states
    plt.contourf(X1, X2, infeasible, levels=[0.5, 1], colors=["#FF783C"], alpha=0.2)
    plt.contour(X1, X2, infeasible, levels=[0.5], colors="#FF783C", linewidths=2)



plt.xlabel(r"$x_1$ (position)", fontsize=14)
plt.ylabel(r"$x_2$ (velocity)", fontsize=14)
plt.title("LQR = Quadratic Cost + Linear Dynamics", fontsize=20)

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
# Legend handles
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
handles = [
    Line2D([], [], color=COST_COLOR, linestyle='-', label='Cost level sets'),
    Line2D([], [], color=VARS_COLOR, linestyle='-', label='Optimal policy'),
    # FancyArrow(0, 0, 0.2, 0, width=0.05, color=VARS_COLOR, label='Optimal policy'),
]
plt.legend(
    handles=handles,
    loc='upper right',
    fontsize=18
)
plt.tight_layout()

if TRAJECTORY:
    # Simulate one optimal trajectory
    x_traj = []
    x = np.array([-1.5, -0])  # Initial condition
    x_traj.append(x.copy())
    for t in range(1000):  # simulate for 50 steps
        u = -K @ x
        x = A @ x + B.flatten() * u
        x_traj.append(x.copy())

    x_traj = np.array(x_traj)

    # Plot the trajectory
    plt.plot(x_traj[:, 0], x_traj[:, 1], color="#429EDB", linewidth=3, label='Optimal trajectory')
    plt.scatter(x_traj[0, 0], x_traj[0, 1], color="#429EDB", marker='o', s=80, label='Start')
    plt.scatter(x_traj[-1, 0], x_traj[-1, 1], color='#429EDB', marker='x', s=80, label='End')

    # Update legend
    handles.extend([
        Line2D([], [], color="#429EDB", linestyle='-', linewidth=3, label='Optimal trajectory'),
        Line2D([], [], color="#429EDB", marker='o', linestyle='None', markersize=12, label='Start'),
        Line2D([], [], color="#429EDB", marker='x', linestyle='None', markersize=12, label='End'),
    ])
    plt.legend(handles=handles, loc='upper right', fontsize=14)



plt.show()