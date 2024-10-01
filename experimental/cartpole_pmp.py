import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import crocoddyl
from IPython.display import HTML
from cartpole_utils import animateCartpole

class DifferentialActionModelCartpole(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isTerminal=False):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, crocoddyl.StateVector(4), 1, 6
        )  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        self.isTerminal = isTerminal

        # Dynamics model parameters
        self.m1 = 1.0
        self.m2 = 0.1
        self.l = 0.5
        self.g = 9.81
        self.R = 0.000001 # control REG

        self.costWeights = [
            0.1,   # sin(th)
            0.1,   # 1-cos(th)
            0.01,   # y
            0.001, # ydot
            0.001, # thdot
            self.R,   # f
        ]  

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone

        # Getting the state and control variables
        y, th, ydot, thdot = x[0].item(), x[1].item(), x[2].item(), x[3].item()
        f = u[0].item()

        # Shortname for system parameters
        s, c = np.sin(th), np.cos(th)
        m1, m2, le, g = self.m1, self.m2, self.l, self.g
        m = m1 + m2
        mu = m1 + m2 * s**2

        # Computing the system acceleration using the equation of motions
        xddot = (f + m2 * c * s * g - m2 * le * s * thdot) / mu
        thddot = (c * f / le + m * g * s / le - m2 * c * s * thdot**2) / mu
        data.xout = np.matrix([xddot, thddot]).T

        # Computing the cost residual and value
        data.r = np.matrix(self.costWeights * np.array([s, 1 - c, y, ydot, thdot, f])).T
        data.cost = 0.5 * sum(np.asarray(data.r) ** 2).item()

    def calcDiff(self, data, x, u=None):
        '''
        Analytical derivatives of the cartpole dynamics and cost
        '''
        if u is None:
            u = self.unone

        # Getting the state and control variables
        y, th, ydot, thdot = x[0].item(), x[1].item(), x[2].item(), x[3].item()
        f = u[0].item()

        # Shortname for system parameters
        m1, m2, lcart, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)
        m = m1 + m2
        mu = m1 + m2 * s**2
        w = self.costWeights

        # derivative of xddot by x, theta, xdot, thetadot
        # derivative of thddot by x, theta, xdot, thetadot
        data.Fx[:, :] = np.array(
            [
                [
                    0.0,
                    (m2 * g * c * c - m2 * g * s * s - m2 * lcart * c * thdot) / mu,
                    0.0,
                    -m2 * lcart * s / mu,
                ],
                [
                    0.0,
                    (
                        (-s * f / lcart)
                        + (m * g * c / lcart)
                        - (m2 * c * c * thdot**2)
                        + (m2 * s * s * thdot**2)
                    )
                    / mu,
                    0.0,
                    -2 * m2 * c * s * thdot,
                ],
            ]
        )
        # derivative of xddot and thddot by f
        data.Fu[:] = np.array([1 / mu, c / (lcart * mu)])
        # first derivative of data.cost by x, theta, xdot, thetadot
        data.Lx[:] = np.array(
            [
                y * w[2] ** 2,
                s * ((w[0] ** 2 - w[1] ** 2) * c + w[1] ** 2),
                ydot * w[3] ** 2,
                thdot * w[4] ** 2,
            ]
        )
        # first derivative of data.cost by f
        data.Lu[:] = np.array([f * w[5] ** 2])
        # second derivative of data.cost by x, theta, xdot, thetadot
        data.Lxx[:] = np.array(
            [
                w[2] ** 2,
                w[0] ** 2 * (c**2 - s**2) + w[1] ** 2 * (s**2 - c**2 + c),
                w[3] ** 2,
                w[4] ** 2,
            ]
        )
        # second derivative of data.cost by f
        data.Luu[:] = np.array([w[5] ** 2])

model = crocoddyl.IntegratedActionModelEuler(DifferentialActionModelCartpole(isTerminal=False), 1e-3)
data  = model.createData()

# Dynamics: [x, theta, x_dot, theta_dot, lambda1, lambda2, lambda3, lambda4]
def dynamics(t, y):
    xdot = np.zeros((4, y.shape[1]))
    lbdot = np.zeros((4, y.shape[1]))
    for i in range(y.shape[1]):
        p, th, pd, thd, l1, l2, l3, l4 = y[:,i]
        # State 
        x = np.array([p,th, pd, thd])
        # Co-state
        lb = np.array([l1, l2, l3, l4])
        # Clamp theta to avoid numerical issues
        th = np.clip(th, 0, 2*np.pi)
        # Optimal control 
        mu = model.differential.m1 + model.differential.m2 * np.sin(th)**2
        Fu = np.array([0.,0.,1./mu,np.cos(th)/(model.differential.l*mu)])
        u =  np.array([-lb.T @ Fu/ model.differential.R])  # Optimal control law
        # State equations
        model.calc(data, x, u)
        model.calcDiff(data, x, u)
        xdot[:,i] = data.xnext
        # Co-state equations
        lbdot[:,i] = -(data.Lx + lb.T @ data.Fx)
    return np.vstack([xdot, lbdot])

# Boundary conditions
def boundary_conditions(ya, yb):
    # Initial state: [x0, x_dot0, theta0, theta_dot0]
    x_a, theta_a, x_dot_a, theta_dot_a, lambda1_a, lambda2_a, lambda3_a, lambda4_a = ya
    print("theta init = ", theta_a)
    # Final state: [xf, x_dotf, thetaf, theta_dotf]
    x_b, theta_b, x_dot_b, theta_dot_b, lambda1_b, lambda2_b, lambda3_b, lambda4_b = yb
    print("theta final = ", theta_b)
    
    # Initial conditions
    bc1 = x_a                  # Start at x=0
    bc2 = theta_a -np.pi       # Start with theta = np.pi/2
    bc3 = x_dot_a              # Start with zero velocity
    bc4 = theta_dot_a          # Start with zero angular velocity
    
    # Final conditions: free terminal state, hence co-states must be zero
    bc5 = lambda1_b
    bc6 = lambda2_b
    bc7 = lambda3_b
    bc8 = lambda4_b

    return np.array([bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8])

# Initial guess for the solution (linear interpolation for states and co-states)
N = 100
T = 2.
t = np.linspace(0, T, N)
y0 = np.zeros((8, N))
y0[0] = np.linspace(0, 0, N)        # x
y0[1] = np.linspace(np.pi, 0, N)  # theta
y0[2] = np.linspace(0, 0, N)        # x_dot
y0[3] = np.linspace(0, 0, N)        # theta_dot
sol = solve_bvp(dynamics, boundary_conditions, t, y0)

# Check if the solution was successful
if sol.success:
    # Extract the solution by evaluating the solution at the specified time points
    p_sol = sol.sol(t)[0]
    theta_sol = sol.sol(t)[1]
    p_dot_sol = sol.sol(t)[2]
    theta_dot_sol = sol.sol(t)[3]
    lambda1_sol = sol.sol(t)[4]
    lambda2_sol = sol.sol(t)[5]
    lambda3_sol = sol.sol(t)[6]
    lambda4_sol = sol.sol(t)[7]

    # Compute the control input from the solution
    u_sol = np.zeros(N)
    for i in range(N):
        th = theta_sol[i]
        lb = np.array([lambda1_sol[i], lambda2_sol[i], lambda3_sol[i], lambda4_sol[i]])
        mu = model.differential.m1 + model.differential.m2 * np.sin(th)**2
        Fu = np.array([0.,0.,1./mu,np.cos(th)/(model.differential.l*mu)])
        u_sol[i] =  -lb.T @ Fu / model.differential.R  # Optimal control law
        # u_sol = -lambda4_sol / R

    # Create animation
    x_sol = np.array([p_sol, theta_sol, p_dot_sol, theta_dot_sol])
    anim = animateCartpole(x_sol.T) 
    # HTML(anim.to_jshtml())
    HTML(anim.to_html5_video())

    # Plot results
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(t, p_sol, label='Cart Position (m)')
    plt.ylabel('Cart Position (m)')
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(t, theta_sol, label='Pole Angle (rad)')
    plt.ylabel('Pole Angle (rad)')
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(t, u_sol, label='Control Input (Force)')
    plt.ylabel('Control Input (N)')
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(t, p_dot_sol, label='Cart Velocity (m/s)')
    plt.ylabel('Cart Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.grid()

    plt.tight_layout()
    plt.show()
else:
    print("The solution was not successful.")
