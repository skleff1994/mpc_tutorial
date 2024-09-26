import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Pendulum and control parameters
g = 9.81  # Gravity
L = 1.0   # Pendulum length
R = 0.01  # Control penalty weight

# Time horizon and discretization
T = 10.0
N = 100
t = np.linspace(0, T, N)

# Dynamics: [theta, omega, lambda1, lambda2]
def dynamics(t, y):
    theta, omega, lambda1, lambda2 = y
    u = -lambda2 / (R * L)  # Optimal control law

    # State equations
    dtheta_dt = omega
    domega_dt = -g / L * np.sin(theta) + u / L
    
    # Co-state equations
    dlambda1_dt = -theta + lambda2 * g / L * np.cos(theta)
    dlambda2_dt = -omega - lambda1

    return np.array([dtheta_dt, domega_dt, dlambda1_dt, dlambda2_dt])

# Boundary conditions
def boundary_conditions(ya, yb):
    theta_a, omega_a, lambda1_a, lambda2_a = ya
    theta_b, omega_b, lambda1_b, lambda2_b = yb
    
    # Initial conditions (pendulum starts from rest)
    bc1 = theta_a - np.pi/2  # Start with theta = pi/2
    bc2 = omega_a            # Start with zero angular velocity

    # Final conditions: free terminal state, hence co-states must be zero
    bc3 = lambda1_b
    bc4 = lambda2_b

    return np.array([bc1, bc2, bc3, bc4])

# Initial guess for the solution (linear interpolation for states and co-states)
y0 = np.zeros((4, N))
y0[0] = np.linspace(np.pi/2, 0, N)  # Initial guess for theta
y0[1] = np.linspace(0, 0, N)        # Initial guess for omega
# p = np.zeros(1)

# Solve the boundary value problem using solve_bvp
sol = solve_bvp(dynamics, boundary_conditions, t, y0)

# Extract the solution
theta_sol = sol.y[0]
omega_sol = sol.y[1]
lambda1_sol = sol.y[2]
lambda2_sol = sol.y[3]

# Compute the control input from the solution
u_sol = -lambda2_sol / (R * L)

# Plot results
plt.figure()
plt.subplot(3,1,1)
plt.plot(np.linspace(0, T, len(theta_sol)), theta_sol, label='Theta (rad)')
plt.ylabel('Theta (rad)')
plt.grid()

plt.subplot(3,1,2)
plt.plot(np.linspace(0, T, len(omega_sol)), omega_sol, label='Omega (rad/s)')
plt.ylabel('Omega (rad/s)')
plt.grid()

plt.subplot(3,1,3)
plt.plot(np.linspace(0, T, len(u_sol)), u_sol, label='Control input u')
plt.ylabel('Control Input u')
plt.xlabel('Time (s)')
plt.grid()

plt.tight_layout()
plt.show()
