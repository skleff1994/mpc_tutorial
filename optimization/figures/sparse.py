import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10   # horizon length
nx = 2   # state dimension
nu = 1   # control dimension

# Dense Hessian (all entries filled)
dense_matrix = np.ones((N*(nx+nu), N*(nx+nu)))

# Sparse block-banded Hessian (MPC structure)
sparse_matrix = np.zeros_like(dense_matrix)
block_size = nx + nu
for k in range(N):
    idx = slice(k*block_size, (k+1)*block_size)
    # Diagonal block
    sparse_matrix[idx, idx] = 1
    # Off-diagonal coupling to next stage (due to dynamics)
    if k < N-1:
        next_idx = slice((k+1)*block_size, (k+2)*block_size)
        sparse_matrix[idx, next_idx] = 1
        sparse_matrix[next_idx, idx] = 1

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].spy(dense_matrix, markersize=8)
axes[0].set_title("Dense Hessian\n(naive optimization)", fontsize=18)

axes[1].spy(sparse_matrix, markersize=8)
axes[1].set_title("Banded Sparse Hessian\n(MPC structure)", fontsize=18)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
