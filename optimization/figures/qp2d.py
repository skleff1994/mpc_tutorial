import numpy as np
import matplotlib.pyplot as plt

#1E961E COST GREEN
#3C32A0 VARS BLUE
#FF783C CSTR RED

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
contours = plt.contour(X, Y, Z, levels=25, cmap='Greens_r', linewidths=3)
plt.clabel(contours, inline=True, fontsize=10)

# Plot nonlinear constraint (red curve where constraint == 0)
constraint_contour = plt.contour(X, Y, C, levels=[0], colors='#FF783C', linewidths=3)
plt.clabel(constraint_contour, fmt='Linear Constraint', inline=True, fontsize=12, manual=[(1.5, 0.3)])

# Optional: Shade infeasible region (e.g., constraint > 0)
plt.contourf(X, Y, C, levels=[0, C.max()], colors=['#FF783C'], alpha=0.2)

# Points
plt.plot(*x_star, marker="o", color='#3C32A0', markersize=10, alpha=0.5, label="Unconstrained minimum")
plt.plot(*x_proj, marker="*", color='#3C32A0', markersize=20, label="Constrained optimum")


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
    Line2D([], [], color='#1E961E', linestyle='-', label='Cost level sets'),
    Line2D([], [], color='#FF783C', linestyle='--', linewidth=2, label='Constraint Boundary'),
    Patch(facecolor='#FF783C', edgecolor='none', alpha=0.2, label='Infeasible Region'),
    Line2D([], [], marker='o', color='w', markerfacecolor='#3C32A0', markersize=10, alpha=0.5, label='Unconstrained minimum'),
    Line2D([], [], marker='*', color='w', markerfacecolor='#3C32A0', markersize=20, label='Constrained optimum')
]
plt.legend(
    handles=handles,
    loc='upper right',
    fontsize=18
)

plt.tight_layout()
plt.show()


