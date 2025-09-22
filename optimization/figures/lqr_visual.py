import numpy as np
import matplotlib.pyplot as plt

#1E961E COST GREEN
#3C32A0 VARS BLUE
#FF783C CSTR RED

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
plt.quiver(X1, X2, U, Vfield, color="#3C32A0", alpha=0.6)

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
handles = [
    Line2D([], [], color='#1E961E', linestyle='-', label='Cost level sets'),
    Line2D([], [], color='#3C32A0', linestyle='-', label='Optimal policy'),
]
plt.legend(
    handles=handles,
    loc='upper right',
    fontsize=18
)
plt.tight_layout()
plt.show()