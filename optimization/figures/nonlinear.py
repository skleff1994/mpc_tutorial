import numpy as np
import matplotlib.pyplot as plt

#1E961E COST GREEN
#3C32A0 VARS BLUE
#FF783C CSTR RED

# Define a nonlinear objective function (bumpy bowl)
def objective(x, y):
    return 0.5 * (x**2 + y**2) + 0.3 * np.sin(3*x) * np.cos(3*y)

# Define a nonlinear constraint function: constraint(x, y) = 0
# E.g., sin(x) + y^2 = 1 => constraint(x, y) = sin(x) + y^2 - 1
def constraint(x, y):
    return np.sin(x) + y**2 - 1

# Create a grid of (x, y) points
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

# Evaluate objective and constraint on the grid
Z = objective(X, Y)
C = constraint(X, Y)

# Plotting
plt.figure(figsize=(8, 10))

# Plot level curves of the objective
contours = plt.contour(X, Y, Z, levels=30, cmap='Greens_r', linewidths=3)
plt.clabel(contours, inline=True, fontsize=10)

# Plot nonlinear constraint (red curve where constraint == 0)
constraint_contour = plt.contour(X, Y, C, levels=[0], colors='#FF783C', linewidths=3)
plt.clabel(constraint_contour, fmt='Nonlinear Constraint', inline=True, fontsize=12)

# Optional: Shade infeasible region (e.g., constraint > 0)
plt.contourf(X, Y, C, levels=[0, C.max()], colors=['#FF783C'], alpha=0.2)

# Labels and title
plt.title("Nonlinear Cost + Nonlinear Constraint", fontsize=20)
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
]
plt.legend(
    handles=handles,
    loc='upper right',
    fontsize=18
)

plt.tight_layout()
plt.show()
