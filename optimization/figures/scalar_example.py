import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# === Color codes (consistent with your QP script) ===
COST_COLOR = '#1E961E'   # Green
VARS_COLOR = '#3C32A0'   # Blue
CSTR_COLOR = '#FF783C'   # Red

WITH_CSTR = True
# === Define the 1D parabola cost function ===
def cost(x):
    return (x - 2)**2

# === Create the domain ===
x = np.linspace(-1, 4, 400)
y = cost(x)

# Unconstrained minimum
x_star = 2
y_star = cost(x_star)

# Add a constraint: x <= 1.5
x_max = 1.5
y_proj = cost(x_max)

# === Plot ===
plt.figure(figsize=(10, 8))

# Plot the parabola
plt.plot(x, y, color=COST_COLOR, linewidth=3, label='Cost')

if(WITH_CSTR):
    # Feasible region shading
    plt.axvspan(x_max, x[-1], color=CSTR_COLOR, alpha=0.2, label='Infeasible Region')
    # Constraint boundary
    plt.axvline(x_max, color=CSTR_COLOR, linestyle='--', linewidth=2, label='Constraint Boundary')
    # Mark unconstrained minimum
    plt.plot(x_star, y_star, marker='o', markersize=10, color=VARS_COLOR, alpha=0.5, label='Unconstrained minimum')
    # Mark constrained optimum
    plt.plot(x_max, y_proj, marker='*', markersize=20, color=VARS_COLOR, label='Constrained optimum')
else:
    plt.plot(x_star, y_star, marker='*', markersize=20, color=VARS_COLOR, label='Optimum')


# Axes and grid
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)

# Labels
if(WITH_CSTR):
    plt.title(r"$\min_x~(x - 2)^2 \quad \text{s.t.}~x \leq 1.5$", fontsize=18)
else:
    plt.title(r"$\min_x~(x - 2)^2$", fontsize=18)
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"Cost", fontsize=16)

# Custom legend
if(WITH_CSTR):
    handles = [
        Line2D([], [], color=COST_COLOR, linewidth=3, label='Cost'),
        Line2D([], [], color=CSTR_COLOR, linestyle='--', linewidth=2, label='Constraint Boundary'),
        Patch(facecolor=CSTR_COLOR, edgecolor='none', alpha=0.2, label='Infeasible Region'),
        Line2D([], [], marker='o', color='w', markerfacecolor=VARS_COLOR, markersize=10, alpha=0.5, label='Unconstrained minimum'),
        Line2D([], [], marker='*', color='w', markerfacecolor=VARS_COLOR, markersize=20, label='Constrained optimum')
    ]
else:
    handles = [
        Line2D([], [], color=COST_COLOR, linewidth=3, label='Cost'),
        Line2D([], [], marker='*', color='w', markerfacecolor=VARS_COLOR, markersize=20, label='Optimum'),
    ]
plt.legend(handles=handles, fontsize=14, loc='upper center', ncol=2)

plt.tight_layout()
plt.show()
