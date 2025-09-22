import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid for paraboloid (cost function)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # paraboloid: quadratic cost

# Define half-plane constraint: y >= -1 (feasible region)
constraint_boundary = -1

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot paraboloid surface
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor='none')

# Highlight feasible region (half-plane)
Y_feasible = np.where(Y >= constraint_boundary, Y, np.nan)
ax.plot_surface(X, Y_feasible, Z, color="lightblue", alpha=0.4)

# Mark unconstrained minimum (0,0,0)
ax.scatter(0, 0, 0, color="red", s=80, label="Unconstrained minimum")

# Mark constrained optimum (0, -1, Z at y=-1)
opt_x, opt_y = 0, -1
ax.scatter(opt_x, opt_y, opt_x**2 + opt_y**2, color="blue", s=80, label="Constrained optimum")

# Formatting
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("Cost")
ax.set_title("Quadratic Program: bowl + half-plane constraint")
ax.legend()

plt.show()
