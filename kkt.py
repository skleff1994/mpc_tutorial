import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import block_diag, csr_matrix

# Pendulum parameters
G = 9.81
L = 1.0
T = 2.0  # Time horizon
N = 50   # Number of time steps
dt = T / N

# Initial and final conditions
theta0 = 1.0  # Initial angle
theta_dot0 = 0.0  # Initial angular velocity
theta_f = 0.0  # Final angle
theta_dot_f = 0.0  # Final angular velocity

# Pendulum dynamics discretized
def pendulum_dynamics_discrete(y, u):
    theta, theta_dot = y
    theta_ddot = -G / L * np.sin(theta) + u / L
    theta_new = theta + dt * theta_dot
    theta_dot_new = theta_dot + dt * theta_ddot
    return np.array([theta_new, theta_dot_new])

### Single Shooting KKT Matrix ###

def kkt_single_shooting():
    # Hessian of the Lagrangian: Includes control regularization and state regularization.
    H = np.zeros((N, N))  # Control Hessian (diagonal since cost is u^2)

    for i in range(N):
        H[i, i] = 0.001  # Control cost regularization term

    # Constraint Jacobian (dynamics)
    A = np.zeros((2 * N, 2 * N))  # State Jacobian
    B = np.zeros((2 * N, N))      # Control Jacobian

    # Initial conditions constraint
    A[0, 0] = 1
    A[1, 1] = 1

    # Dynamics constraints
    for i in range(1, N):
        A[2 * i, 2 * (i - 1)] = -1  # theta(t+1) - theta(t)
        A[2 * i + 1, 2 * (i - 1) + 1] = -1  # theta_dot(t+1) - theta_dot(t)
        # Add derivatives w.r.t control
        B[2 * i + 1, i] = dt / L

    # KKT matrix construction
    KKT = np.block([[H, B.T], [B, np.zeros((2 * N, 2 * N))]])

    return KKT

### Multiple Shooting KKT Matrix ###

def kkt_multiple_shooting():
    # Block-diagonal Hessian of Lagrangian (since each control affects only its own interval)
    H_blocks = [np.array([[0.001]]) for _ in range(N)]  # Each control has its own block

    # Stack to form Hessian
    H = block_diag(H_blocks)

    # Jacobian for state dynamics (block structure)
    A_blocks = []
    B_blocks = []
    for i in range(N):
        A_i = np.zeros((2, 2))
        B_i = np.zeros((2, 1))

        if i > 0:
            # Dynamic continuity constraint (theta and theta_dot)
            A_i[0, 0] = -1
            A_i[1, 1] = -1

        # Dynamic control effect
        B_i[1, 0] = dt / L

        A_blocks.append(A_i)
        B_blocks.append(B_i)

    # State Jacobian A (block diagonal) and control Jacobian B
    A = block_diag(A_blocks)
    B = block_diag(B_blocks)

    # KKT matrix for multiple shooting (same form as single, but block-sparse)
    KKT = csr_matrix(np.block([[H.toarray(), B.T.toarray()], [B.toarray(), np.zeros(A.shape)]]))

    return KKT

### Visualizing Sparsity Pattern ###

def plot_sparsity(matrix, title):
    plt.figure()
    plt.spy(matrix, markersize=1)
    plt.title(title)

### Main ###
if __name__ == "__main__":
    # Single Shooting KKT Matrix
    kkt_single = kkt_single_shooting()
    plot_sparsity(kkt_single, "KKT Matrix - Single Shooting")

    # Multiple Shooting KKT Matrix
    kkt_multiple = kkt_multiple_shooting()
    plot_sparsity(kkt_multiple.toarray(), "KKT Matrix - Multiple Shooting")
    plt.show()
