import numpy as np
import matplotlib.pyplot as plt 
import pylab

A = 1
B = 100
NEWTON = False

def rosenbrock(x):
    """Compute the Rosenbrock function."""
    return (A - x[0]) ** 2 + B * (x[1] - x[0] ** 2) ** 2

def rosenbrock_grad(x):
    """Compute the gradient of the Rosenbrock function."""
    dfdx0 = -2 * (A - x[0]) - 4 * B * x[0] * (x[1] - x[0] ** 2)
    dfdx1 = 2 * B * (x[1] - x[0] ** 2)
    return np.array([dfdx0, dfdx1])

def rosenbrock_hessian(x):
    """Compute the Hessian of the Rosenbrock function."""
    d2fdx02 = 2 + 4 * B * (x[1] - x[0] ** 2) + 12 * B * x[0]**2
    d2fdx01 = -4 * B * x[0]
    d2fdx11 = 2 * B
    return np.array([[d2fdx02, d2fdx01], [d2fdx01, d2fdx11]])


def F(z1, z2):
    '''
    This the objective function we are trying to minimize
    '''
    return (A- z1)**2 + B * (z2 - z1**2)**2

def F_grad(z1, z2):
    '''
    Gradient of the objective function
    '''
    F_z1 = -2 * (A - z1) - 4 * B * (z2 - z1**2) * z1
    F_z2 =  2 * B * (z2 - z1**2)
    return np.array([F_z1, F_z2])


def F_hess(z1, z2):
    '''
    Hessian of the objective function
    '''
    F_z1z1 = 2 - 4 * z2 + 12 * z1**2
    F_z1z2 = -4 * B * z1
    F_z2z2 = 2 * B
    return np.array([[F_z1z1, F_z1z2],[F_z1z2, F_z2z2]])

def backtracking_line_search(func, grad, x, p, alpha=1, beta=0.5):
    """Perform backtracking line search."""
    while func(x + alpha * p) > func(x) + 0.1 * alpha * np.dot(grad(x), p):
        alpha *= beta
    print("alpha_bt = ", alpha)
    return alpha

def gradient_step(z):
    return -rosenbrock_grad(z)
   
def newton_step(z):
    det_hess = np.linalg.det(rosenbrock_hessian(z))
    if det_hess == 0:
        raise ValueError("Hessian is singular; cannot invert.")
    return -np.linalg.solve(rosenbrock_hessian(z), rosenbrock_grad(z))

MAXIT = 10000
z_iterates         = np.zeros((MAXIT, 2))
dz_iterates        = np.zeros((MAXIT, 2))
alphas             = np.zeros(MAXIT)
F_values           = np.zeros(MAXIT)
F_grad_norm_values = np.zeros(MAXIT)
z_iterates[0] = np.array([0.5, -0.5])

TOL = 1e-3
iter = 0
grad_norm = np.inf
while iter < MAXIT-1 and grad_norm > TOL:
    # Compute current cost and grad norm
    z = z_iterates[iter,:].copy()
    F_values[iter] = rosenbrock(z) 
    grad_norm = np.linalg.norm(rosenbrock_grad(z), 1)
    F_grad_norm_values[iter] = grad_norm
    print("Iteration ", iter+1, " cost = ", F_values[iter], ", ||grad|| = ", grad_norm)
    if grad_norm < TOL:
        print("Converged in ", iter+1, " iterations !")
        z_opt = z.copy()
        z_iterates[iter:,:] = z_opt
    else:    
        # Compute search direction and step lenght
        if(NEWTON == False):
            dz = gradient_step(z_iterates[iter,:])
            alpha = backtracking_line_search(rosenbrock, rosenbrock_grad, z, dz, alpha=0.01, beta=0.5) 
            # assert alpha == line_search(z_iterates[iter,:], dz, alpha0=0.01)
        else:
            dz = newton_step(z_iterates[iter,:])
            alpha = backtracking_line_search(rosenbrock, rosenbrock_grad, z, dz, alpha=1, beta=0.5)
            # assert alpha == line_search(z_iterates[iter,:], dz, alpha0=1)
        dz_iterates[iter,:] = dz
        print("alpha = ", alpha)
        alphas[iter] = alpha
        # print("alpha", alpha)
        # Take the step 
        z_iterates[iter+1,:] = z_iterates[iter,:] + alpha * dz
        iter += 1

print("Final cost = ", rosenbrock(z_iterates[iter-1]))
print("Local minimizer = ", z)


# Visualization
z1 = np.arange(-1., 2, 0.01)
z2 = np.arange(-1., 2, 0.01)
Z1, Z2 = pylab.meshgrid(z1, z2)
Z = F(Z1, Z2)

# Plot level sets
contour_levels = np.linspace(0, 100, 50)
pylab.contour(Z1, Z2, Z, levels=contour_levels) # cmap='viridis')
pylab.colorbar(label='Objective Function Value')

# Plotting the gradient descent path
pylab.plot(z_iterates[:, 0], z_iterates[:, 1], 'ro-', markersize=5, label='Gradient Descent Path')  # green dots and line
pylab.scatter(z_iterates[:, 0], z_iterates[:, 1], color='r')  # green dots
pylab.plot(z_iterates[0, 0], z_iterates[0, 1], color='g', marker='o', markersize=12, label='Initial guess')  # green dots
pylab.plot(1, 1, color='b', marker='*', markersize=12, label='Global optimum')  # green dots

# Labels and legend
pylab.xlabel('z1')
pylab.ylabel('z2')
pylab.title('Gradient Descent Iterates with Level Sets')
pylab.legend()
pylab.xlim(-1, 2.)
pylab.ylim(-1, 2.)
pylab.grid()
pylab.show()



