import numpy as np
import matplotlib.pyplot as plt

# Constants
m1 = 1.0  # mass of the cart
m2 = 0.1  # mass of the pole
l = 0.5   # length of the pole
g = 9.81  # gravity
dt = 0.02 # time step for simulation
N = 100   # number of iterations
theta_max = np.pi 
p_max = 1.0

# State discretization
p_vals = np.linspace(-p_max, p_max, 11)
theta_vals = np.linspace(-theta_max, theta_max, 11)
p_dot_vals = np.linspace(-1, 1, 11)
theta_dot_vals = np.linspace(-1, 1, 11)
f_vals = np.linspace(-5, 5, 11)

# Value function grid
V = np.zeros((len(p_vals), len(theta_vals), len(p_dot_vals), len(theta_dot_vals)))
optimal_policy = np.zeros_like(V)

def cost_function(p, theta, p_dot, theta_dot, f):
    # Define the quadratic cost function
    return 0.01*p**2 + 0.1*(np.sin(theta))**2 + 0.1*(1 - np.cos(theta))**2 + 0.001*p_dot**2 + 0.001*theta_dot**2 + 0.001*f**2

def terminal_cost_function(p, theta, p_dot, theta_dot):
    # Define the terminal cost function
    return 0.01*p**2 + 100*(np.sin(theta))**2 + 100*(1 - np.cos(theta))**2 + 100*p_dot**2 + 100*theta_dot**2

def mu(theta):
    return m1 + m2 * (np.sin(theta) ** 2)

def dynamics(p, theta, p_dot, theta_dot, f):
    mu_theta = mu(theta)
    ddot_theta = (1 / mu_theta) * (np.cos(theta) * f + (m1 + m2) * g * np.sin(theta) - m2 * np.cos(theta) * np.sin(theta) * theta_dot**2)
    ddot_p = (1 / mu_theta) * (f + m2 * g * np.cos(theta) * np.sin(theta) - m2 * l * np.sin(theta) * theta_dot)
    return ddot_p, ddot_theta

def dt_dynamics(x, u):
    ddot_p, ddot_theta = dynamics(x[0], x[1], x[2], x[3], u)
    new_p = x[0] + x[2] * dt + 0.5 * ddot_p * dt**2
    new_theta = x[1] + x[3] * dt + 0.5 * ddot_theta * dt**2
    new_p_dot = x[2] + ddot_p * dt
    new_theta_dot = x[3] + ddot_theta * dt
    return np.array([new_p, new_theta, new_p_dot, new_theta_dot])

# Initialize value function with terminal cost
for i, p in enumerate(p_vals):
    for j, theta in enumerate(theta_vals):
        for k, p_dot in enumerate(p_dot_vals):
            for l, theta_dot in enumerate(theta_dot_vals):
                V[i, j, k, l] = terminal_cost_function(p, theta, p_dot, theta_dot)

# Iterative computation of value function
for iteration in range(N):
    print("iter =", iteration)
    V_new = np.copy(V)
    for i, p in enumerate(p_vals):
        print("   i = ", i)
        for j, theta in enumerate(theta_vals):
            for k, p_dot in enumerate(p_dot_vals):
                for l, theta_dot in enumerate(theta_dot_vals):
                    min_value = np.inf
                    best_u = 0
                    for f in f_vals:
                        # Estimate future values
                        new_state = dt_dynamics(np.array([p, theta, p_dot, theta_dot]), f)
                        
                        # Boundary conditions
                        new_p = np.clip(new_state[0], -p_max, p_max)
                        new_theta = np.clip(new_state[1], -theta_max, theta_max)
                        new_p_dot = np.clip(new_state[2], -5, 5)
                        new_theta_dot = np.clip(new_state[3], -5, 5)
                        
                        # Find indices of new state
                        p_idx = np.digitize(new_p, p_vals) - 1
                        theta_idx = np.digitize(new_theta, theta_vals) - 1
                        p_dot_idx = np.digitize(new_p_dot, p_dot_vals) - 1
                        theta_dot_idx = np.digitize(new_theta_dot, theta_dot_vals) - 1
                        
                        # Compute the cost and update the value function
                        value = cost_function(p, theta, p_dot, theta_dot, f) + V[p_idx, theta_idx, p_dot_idx, theta_dot_idx]
                        if value < min_value:
                            min_value = value
                            best_u = f
                            
                    # Update the value function
                    V_new[i, j, k, l] = min(min_value, V_new[i, j, k, l])
                    # Update policy 
                    optimal_policy[i, j, k, l] = best_u
    V = V_new

# Visualization of the value function for a specific state
fixed_p_dot_idx = 5  # Middle of the p_dot range
fixed_theta_dot_idx = 5  # Middle of the theta_dot range

plt.imshow(V[:, :, fixed_p_dot_idx, fixed_theta_dot_idx], extent=(-p_max, p_max, -theta_max, theta_max), origin='lower', aspect='auto')
plt.colorbar(label='Value Function')
plt.xlabel('Cart Position (p)')
plt.ylabel('Pole Angle (theta)')
plt.title('Value Function V(p, theta) with fixed velocities')
plt.show()

# Simulating optimal policy
x = np.array([0., -1, 0, 0])  # Initial state
TSIM = 1000
x_traj = np.zeros((TSIM + 1, 4))
u_traj = np.zeros(TSIM)
x_traj[0, :] = x

print("Simulating optimal policy")
for t in range(TSIM):
    # Compute optimal control (online)
    p_idx = np.digitize(x[0], p_vals) - 1
    theta_idx = np.digitize(x[1], theta_vals) - 1
    p_dot_idx = np.digitize(x[2], p_dot_vals) - 1
    theta_dot_idx = np.digitize(x[3], theta_dot_vals) - 1
    u = optimal_policy[p_idx, theta_idx, p_dot_idx, theta_dot_idx]
    
    # Record control + simulate + record state
    u_traj[t] = u
    x = dt_dynamics(x, u)
    x_traj[t + 1, :] = x

# Visualization of the trajectory
time_lin = np.linspace(0, dt * (TSIM + 1), TSIM + 1)

fig, axs = plt.subplots(4)
labels = ['p (m)', 'th (rad)', 'pdot (m/s)', 'thdot (rad/s)']
for i in range(4):
    axs[i].plot(time_lin, x_traj[:, i], label=labels[i])
    axs[i].set_ylabel(labels[i])
    axs[i].grid()
plt.xlabel('time (s)')
fig.suptitle("State trajectory")

plt.figure()
plt.plot(time_lin[:-1], u_traj[:])
plt.title("Control trajectory")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.grid()

plt.show()

from IPython.display import HTML
from cartpole_utils import animateCartpole
anim = animateCartpole(x_traj) 
# HTML(anim.to_jshtml())
HTML(anim.to_html5_video())