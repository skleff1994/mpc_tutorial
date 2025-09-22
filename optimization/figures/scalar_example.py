import numpy as np
import matplotlib.pyplot as plt


# Define a nonlinear objective function (bumpy bowl)
def objective(x, y):
    return 2*(x - 1)**2 + (y - 1)**2

# Define a nonlinear constraint function: constraint(x, y) = 0
# E.g., sin(x) + y^2 = 1 => constraint(x, y) = sin(x) + y^2 - 1
def constraint(x, y):
    return  1.5 - (x + y) 

# Create a grid of (x, y) points
x = np.linspace(-2, 3, 400)
y = np.linspace(-2, 3, 400)
X, Y = np.meshgrid(x, y)

# Evaluate objective and constraint on the grid
Z = objective(X, Y)
C = constraint(X, Y)

# Unconstrained minimum
x_star = np.array([1, 1])
# Constrained optimum (projection of (1,1) onto x1+x2=1.5 line)
# Since unconstrained optimum (1,1) is infeasible (sum=2 >1.5)
# Projection onto line x1+x2=1.5
A = np.array([1, 1])
b = 1.5
x_proj = x_star - (np.dot(A, x_star) - b) * A / np.dot(A, A)

# Plot
plt.figure(figsize=(8, 10))

# Plot level curves of the objective
contours = plt.contour(X, Y, Z, levels=25, cmap='Purples_r', linewidths=3)
plt.clabel(contours, inline=True, fontsize=10)

# Plot nonlinear constraint (red curve where constraint == 0)
constraint_contour = plt.contour(X, Y, C, levels=[0], colors='orange', linewidths=3)
plt.clabel(constraint_contour, fmt='Linear Constraint', inline=True, fontsize=12, manual=[(1.5, 0.3)])

# Optional: Shade infeasible region (e.g., constraint > 0)
plt.contourf(X, Y, C, levels=[0, C.max()], colors=['#ffe6e6'], alpha=0.6)

# Points
plt.plot(*x_star, "bo", markersize=10, label="Unconstrained minimum")
plt.plot(*x_proj, "g*", markersize=15, label="Constrained optimum")


# Labels and title
plt.title("QP = Quadratic Cost + Linear Constraint", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Legend handles
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], color='purple', linestyle='-', label='Cost level sets'),
    Line2D([], [], color='orange', linestyle='--', linewidth=2, label='Constraint Boundary'),
    Patch(facecolor='#ffe6e6', edgecolor='none', alpha=0.6, label='Infeasible Region'),
    Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Unconstrained minimum'),
    Line2D([], [], marker='*', color='w', markerfacecolor='green', markersize=15, label='Constrained optimum')
]
plt.legend(
    handles=handles,
    loc='upper right',
    fontsize=18
)

plt.tight_layout()
plt.show()


