import numpy as np
import matplotlib.pyplot as plt

# Define the cost function and its gradient
def cost_function(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def cost_gradient(x):
    return np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])

# Define the constraint function and its Jacobian
def constraint_function(x):
    return x[0]**2 + x[1]**2 - 1  # Circle constraint

def constraint_jacobian(x):
    return np.array([[2 * x[0], 2 * x[1]]])  # Gradient of the constraint

# Merit function to evaluate
def merit_function(x):
    return cost_function(x) + 1e3 * max(0, constraint_function(x))**2

# Newton's method with line search
def newton_method(x0, tol=1e-6, max_iter=100, alpha=0.5, beta=0.5, max_ls_step=100):
    x = x0
    iterates = [x]
    logs = []

    for iteration in range(max_iter):
        grad_f = cost_gradient(x)
        jac_g = constraint_jacobian(x)
        g = constraint_function(x)

        # Construct the KKT matrix
        KKT_matrix = np.block([[2 * np.eye(2), jac_g.T],
                                [jac_g, np.zeros((1, 1))]])

        # Construct the KKT vector
        KKT_vector = np.hstack([-grad_f, -g])

        # Solve the linear system
        delta = np.linalg.solve(KKT_matrix, KKT_vector)

        # Initialize step size
        t = 1.0
        x_new = x + t * delta[:2]

        # Line search to ensure descent
        ls_step = 0
        while merit_function(x_new) >= merit_function(x) + alpha * t * np.dot(grad_f, delta[:2]) and ls_step < max_ls_step:
            t *= beta
            x_new = x + t * delta[:2]
            ls_step += 1

        # Update the solution
        iterates.append(x_new)

        # Log values
        residual = np.abs(g)
        cost = cost_function(x_new)
        step_size = np.linalg.norm(delta[:2]) * t
        logs.append((iteration + 1, residual, cost, step_size))
        
        # Print the log for this iteration
        print(f"Iteration {iteration + 1}: Residual = {residual:.6f}, Cost = {cost:.6f}, Step Size = {step_size:.6f}")

        # Check convergence
        if step_size < tol:
            break

        x = x_new

    return x, iterates, logs

# Initial guess
x0 = np.array([-0.5, 0.5])

# Solve the optimization problem
solution, iterates, logs = newton_method(x0)

# Prepare to plot
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = cost_function(np.array([X, Y]))
G = constraint_function(np.array([X, Y]))

# Plotting
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)
plt.contour(X, Y, G, levels=[0], colors='black', linewidths=2, label='Constraint')
iterates = np.array(iterates)

# Plot iterates
plt.plot(x0[0], x0[1], color='g', marker='o', markersize=12, label='Initial guess')  # green dots
plt.plot(iterates[:, 0], iterates[:, 1], 'o-', color='red', label='Iterates')
plt.plot(solution[0], solution[1], 'bo', label='Solution', markersize=10)

plt.title('Newton Method with Line Search for Nonlinear Equality Constrained Optimization')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
