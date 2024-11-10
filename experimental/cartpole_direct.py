import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import block_diag, csr_matrix
from scipy.optimize import minimize

# Cartpole parameters
m_cart = 1.0  # mass of the cart (kg)
m_pole = 0.1  # mass of the pole (kg)
L = 0.5       # length of the pole (m)
g = 9.81      # acceleration due to gravity (m/s^2)
T = 1.0       # time horizon (s)
N = 10        # number of time steps
dt = T / N    # time step size

# Initial and final conditions
x0 = np.array([0.0, 0.0, np.pi, 0.0])  # [cart position, cart velocity, pole angle, pole angular velocity]
xf = np.array([0.0, 0.0, 0.0, 0.0])     # [final position, final velocity, final angle, final angular velocity]

# Dynamics of the cartpole system
def cartpole_dynamics(state, u):
    x, x_dot, theta, theta_dot = state
    # Equations of motion
    theta_ddot = (g * np.sin(theta) + np.cos(theta) * (-u - m_pole * L * theta_dot**2 * np.sin(theta)) / (m_cart + m_pole)) / (L * (4/3 - m_pole * np.cos(theta)**2 / (m_cart + m_pole)))
    x_ddot = (u + m_pole * L * (theta_dot**2 * np.sin(theta) - theta_ddot * np.cos(theta))) / (m_cart + m_pole)
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

# Cost function: Quadratic cost on state and control
def cost_function(states, controls):
    Q = np.diag([10, 1, 100, 10])  # State cost weights
    R = 0.01                        # Control cost weight
    cost = 0.0
    for i in range(N):
        cost += (states[i] @ Q @ states[i]) + R * controls[i]**2
    return cost

### Single Shooting KKT Matrix ###

def kkt_single_shooting():
    # Hessian of the Lagrangian (including control cost)
    H = np.zeros((N * 4 + N, N * 4 + N))  # size nx*N + nu*N
    print("hessian shape = ", H.shape)
    for i in range(N):
        H[i * 4 + 2, i * 4 + 2] = 100  # State cost for theta
        H[i * 4 + 3, i * 4 + 3] = 10   # State cost for theta_dot
        H[N * 4 + i, N * 4 + i] = 0.01  # Control cost
    #z = [ (x, xd, th, thd)_0, ]
    # Constraint Jacobian
    A = np.zeros((4 * N, N * 4))  # Only states, no control inputs in A for now
    B = np.zeros((4 * N, N))       # Constraint Jacobian should match states
    print("A shape = ", A.shape)
    print("B shape = ", B.shape)
    # Initial conditions constraint
    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 2] = 1
    A[3, 3] = 1

    # Dynamics constraints
    for i in range(1, N):
        A[4 * i:4 * i + 4, 4 * (i - 1):4 * i] = -np.eye(4)
    
    # KKT matrix construction
    KKT = np.zeros((N * 4 + N, N * 4 + N + N))  # Adjusted to add space for B
    print("KKT shape = ", KKT.shape)
    KKT[:N * 4 + N, :N * 4 + N] = H
    # KKT[:N * 4 + N, N * 4 + N:N * 4 + N + N] = B.T  # Correct shape for B.T
    # KKT[N * 4 + N:, :N * 4 + N] = B  # Correct shape for B
    # KKT[N * 4 + N:, N * 4 + N:] = np.zeros((N, N))  # Zero block for additional constraints

    return KKT





### Multiple Shooting KKT Matrix ###

def kkt_multiple_shooting():
    H_blocks = [np.array([[0.01]]) for _ in range(N)]  # Control cost
    
    # Stack to form Hessian
    H = block_diag(H_blocks)

    # Jacobian for state dynamics (block structure)
    A_blocks = []
    for i in range(N):
        A_i = np.zeros((4, 4))
        if i > 0:
            # Dynamic continuity constraint
            A_i[0, 0] = -1
            A_i[1, 1] = -1
            A_i[2, 2] = -1
            A_i[3, 3] = -1
        A_blocks.append(A_i)

    # State Jacobian A (block diagonal)
    A = block_diag(A_blocks)

    # KKT matrix for multiple shooting (same form as single, but block-sparse)
    KKT = csr_matrix(np.block([[H.toarray(), np.zeros((N, 4 * N))], [np.zeros((4 * N, N)), np.zeros((4 * N, 4 * N))]]))
    return KKT

### Visualization of Sparsity Patterns ###

def plot_sparsity(matrix, title):
    plt.figure()
    plt.spy(matrix, markersize=1)
    plt.title(title)
    plt.show()

### Sequential Quadratic Programming (SQP) Resolution ###

def sqp_resolution(initial_guess):
    # Placeholder for SQP iterations
    def objective(x):
        # Unroll states and controls
        states = x[:N * 4].reshape((N, 4))
        controls = x[N * 4:].reshape((N,))
        return cost_function(states, controls)

    constraints = {'type': 'eq', 'fun': lambda x: cartpole_constraints(x)}

    res = minimize(objective, initial_guess, constraints=constraints, method='SLSQP', options={'disp': True})
    return res

def cartpole_constraints(x):
    states = x[:N * 4].reshape((N, 4))
    controls = x[N * 4:].reshape((N,))
    constraints = np.zeros((4 * N))
    
    for i in range(N):
        if i > 0:
            # Enforce dynamics constraints
            dynamics_error = states[i] - cartpole_dynamics(states[i - 1], controls[i - 1]) * dt
            constraints[4 * i: 4 * i + 4] = dynamics_error

    # Initial conditions
    constraints[0:4] = states[0] - x0
    return constraints

### Main ###
if __name__ == "__main__":
    # Single Shooting KKT Matrix
    kkt_single = kkt_single_shooting()
    plot_sparsity(kkt_single, "KKT Matrix - Single Shooting")

    # Multiple Shooting KKT Matrix
    kkt_multiple = kkt_multiple_shooting()
    plot_sparsity(kkt_multiple.toarray(), "KKT Matrix - Multiple Shooting")

    # Initial guess for SQP resolution
    initial_guess = np.zeros(N * 4 + N)  # States + controls
    initial_guess[:N * 4] = np.tile(x0, N)  # Initial state guess for all time steps

    # Run SQP
    sqp_result = sqp_resolution(initial_guess)

    # Extract and visualize iterates
    states_opt = sqp_result.x[:N * 4].reshape((N, 4))
    controls_opt = sqp_result.x[N * 4:].reshape((N,))
    
    # Visualize the optimal control input and states
    t = np.linspace(0, T, N)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, states_opt[:, 0], label='Cart Position (m)')
    plt.plot(t, states_opt[:, 2], label='Pole Angle (rad)')
    plt.legend()
    plt.title('States over Time')
    
    plt.subplot(2, 1, 2)
    plt.step(t[:-1], controls_opt, label='Control Input (Force)', where='post')
    plt.legend()
    plt.title('Control Input over Time')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()
