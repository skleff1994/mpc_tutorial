import matplotlib.pyplot as plt
import numpy as np


def plot_matrix_structure(ax, dense=True, N=15, color='gray'):
    M = np.zeros((N, N))
    if dense:
        M[:, :] = 1
    else:
        for i in range(N):
            M[i, i] = 1
            if i > 0:
                M[i, i-1] = 1
            if i < N-1:
                M[i, i+1] = 1
    ax.spy(M, markersize=10, color=color)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Left: dense
# plt.subplot(1,2,1)
plot_matrix_structure(axes[0], dense=True, color='gray')
# axes[0].set_title("Dense QP\n$O(T^3)$", color="red", fontsize=20)

# Right: sparse
# plt.subplot(1,2,2)
plot_matrix_structure(axes[1], dense=False, color='black')
# axes[1].set_title("Structured QP\n$O(T)$", color="blue", fontsize=20)


for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
