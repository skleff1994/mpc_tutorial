import matplotlib.pyplot as plt

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 3))

# Number of timesteps
T = 5
x_positions = range(T + 1)

# Plot states x_k as circles
for t in x_positions:
    ax.plot(t, 1, 'o', color="tab:blue", markersize=10)
    ax.text(t, 1.15, f"$x_{t}$", ha="center", fontsize=12)

# Plot controls u_k as squares between states
for t in range(T):
    ax.plot(t + 0.5, 0, 's', color="tab:orange", markersize=10)
    ax.text(t + 0.5, -0.25, f"$u_{t}$", ha="center", fontsize=12)

# Draw arrows from x_k and u_k to x_{k+1}
for t in range(T):
    # arrow from x_t and u_t to x_{t+1}
    ax.annotate("",
                xy=(t + 1, 1), xycoords='data',
                xytext=(t + 0.5, 0.1), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.annotate("",
                xy=(t + 1, 1), xycoords='data',
                xytext=(t, 1.05), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="black"))

# Add running cost boxes
for t in range(T):
    ax.text(t + 0.25, 1.5, r"$\ell(x_{%d},u_{%d})$" % (t, t),
            fontsize=10, bbox=dict(boxstyle="round", fc="wheat", ec="0.7"))

# Add terminal cost
ax.text(T, 1.5, r"$\ell_T(x_T)$",
        fontsize=10, bbox=dict(boxstyle="round", fc="lightgreen", ec="0.7"))

# Decorations
ax.set_ylim(-0.5, 2)
ax.axis("off")
ax.set_title("Trajectory optimization: states $x$, controls $u$, costs", fontsize=14)

plt.tight_layout()
# plt.savefig("/mnt/data/ocp_timeline.png", dpi=150)
plt.show()
