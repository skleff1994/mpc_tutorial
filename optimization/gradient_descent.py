import numpy as np
import matplotlib.pyplot as plt 
import pylab

COEFFS = [1, 1, 0.4]

def F(z1, z2):
    '''
    This the objective function we are trying to minimize
    '''
    return COEFFS[0]*z1**2 + COEFFS[1]*z2**2 + COEFFS[2]*z1*z2 

def F_grad(z1, z2):
    '''
    Gradient of the objective function
    '''
    F_z1 = COEFFS[0]*2*z1 + COEFFS[2]*z2 
    F_z2 = COEFFS[1]*2*z2 + COEFFS[2]*z1
    return np.array([F_z1, F_z2])

def F_hess(z1, z2):
    '''
    Hessian of the objective function
    '''
    F_z1z1 = COEFFS[0]*2 
    F_z2z2 = COEFFS[1]*2 
    F_z1z2 = COEFFS[2]
    return np.array([[F_z1z1, F_z1z2],[F_z1z2, F_z2z2]])

# z1 = np.arange(-3.0,3.0,0.1)
# z2 = np.arange(-3.0,3.0,0.1)
# Z1,Z2 = pylab.meshgrid(z1, z2) # grid of point
# Z = F(Z1, Z2) # evaluation of the function on the grid

# im = pylab.imshow(Z,cmap=pylab.cm.RdBu) # drawing the function
# # adding the Contour lines with labels
# cset = pylab.contour(Z,np.arange(0.,5,0.5),linewidths=2,cmap=pylab.cm.Set2)
# pylab.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# pylab.colorbar(im) # adding the colobar on the right
# # latex fashion title
# # pylab.title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
# pylab.show()


def line_search(z, dz, alpha0=1., gamma=0.5, beta=1., max_bt=10):
    '''
    Backtracking line search 
        z      : current iterate
        dz     : search (descent) direction
        alpha0 : initial alpha value
        gamma  : contraction rate of alpha
        beta   : sufficient decrease condition parameter (Armijo)
        max_bt : max number of backtracking steps
    '''
    alpha = alpha0
    znext = z + alpha * dz
    Fz = gradient_step(z).T
    bt = 0
    decrease = alpha * gamma * beta * Fz @ dz
    while F(znext[0], znext[1]) > F(z[0], z[1]) + decrease and bt < max_bt:
        alpha = gamma*alpha
        bt += 1
    print("BLTS alpha = ", alpha, " (bt =  ", bt, ")")
    return alpha

def gradient_step(z):
    return -F_grad(z[0], z[1])
   
def newton_step(z):
    return -np.linalg.inv(F_hess(z[0], z[1])) @ F_grad(z[0], z[1])

MAXIT = 10
z_iterates  = np.zeros((MAXIT+1, 2))
dz_iterates = np.zeros((MAXIT, 2))
alphas      = np.zeros(MAXIT)
F_values    = np.zeros(MAXIT)
# initial guess
z_iterates[0] = np.array([2., -2])



for k in range(MAXIT):
    # Compute current cost
    z = z_iterates[k,:]
    F_values[k] = F(z[0], z[1]) 
    print("Iteration ", k+1, " cost = ", F_values[k])
    # Compute search direction
    dz = newton_step(z_iterates[k])
    dz_iterates[k,:] = dz
    # Compute step length 
    alpha = line_search(z_iterates[k], dz)
    alphas[k] = alpha
    # Take the step 
    z_iterates[k+1,:] = z_iterates[k] + alpha * dz

print("Final cost = ", F(z_iterates[k+1,0], z_iterates[k+1,1]))
    



# Visualization
z1 = np.arange(-3.0, 3.0, 0.1)
z2 = np.arange(-3.0, 3.0, 0.1)
Z1, Z2 = pylab.meshgrid(z1, z2)
Z = F(Z1, Z2)

# Plot level sets
contour_levels = np.linspace(0, 10, 25)
pylab.contour(Z1, Z2, Z, levels=contour_levels) # cmap='viridis')
pylab.colorbar(label='Objective Function Value')

# Plotting the gradient descent path
pylab.plot(z_iterates[:, 0], z_iterates[:, 1], 'ro-', markersize=5, label='Gradient Descent Path')  # green dots and line
pylab.scatter(z_iterates[:, 0], z_iterates[:, 1], color='r')  # green dots

# Labels and legend
pylab.xlabel('z1')
pylab.ylabel('z2')
pylab.title('Gradient Descent Iterates with Level Sets')
pylab.legend()
pylab.xlim(-3, 3)
pylab.ylim(-3, 3)
pylab.grid()
pylab.show()

# # Visualization
# z1 = np.arange(-2.0, 2.0, 0.01)
# z2 = np.arange(-2.0, 2.0, 0.01)
# Z1, Z2 = pylab.meshgrid(z1, z2)
# Z = F(Z1, Z2)

# im = pylab.imshow(Z, cmap=pylab.cm.RdBu)

# cset = pylab.contour(Z, np.arange(0., 5, 0.5), linewidths=2) #, cmap=pylab.cm.Set2)
# pylab.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
# pylab.colorbar(im)

# # Setting the tick labels to the actual z1 and z2 values
# pylab.xticks(np.arange(-2, 2, 1))  # Adjust as needed for the x-axis ticks
# pylab.yticks(np.arange(-2, 2, 1))  # Adjust as needed for the y-axis ticks


# # Plotting the gradient descent path
# pylab.plot(z_iterates[:, 0], z_iterates[:, 1], 'go-', markersize=5, label='Gradient Descent Path')  # green dots and line
# pylab.scatter(z_iterates[:, 0], z_iterates[:, 1], color='green')  # green dots
# pylab.legend()
# pylab.title('Gradient Descent Iterates')
# pylab.show()
