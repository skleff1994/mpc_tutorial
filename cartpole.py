'''
This file is an adaptation of Crocoddyl's cartpole tutorial
The original code can be found here : https://github.com/loco-3d/crocoddyl/blob/devel/examples/notebooks
'''

import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
from IPython.display import HTML
from cartpole_utils import animateCartpole
import mim_solvers


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

        # Cost function parameters
        if(self.isTerminal):
            self.costWeights = [
                100,   # sin(th)
                100,   # 1-cos(th)
                0.001,   # y
                100, # ydot
                100, # thdot
                0,   # f
            ]  
        else:
            self.costWeights = [
                0.1,   # sin(th)
                0.1,   # 1-cos(th)
                0.01,   # y
                0.001, # ydot
                0.001, # thdot
                0.001,   # f
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


if __name__ == "__main__":

    # OCP parameters
    x0 = np.array([0.0, 3.14, 0.0, 0.0]).T
    u0 = np.zeros(1)
    T = 100
    dt = 2e-2

    # Create the running models
    running_DAM = DifferentialActionModelCartpole(isTerminal=False)
    running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)

    terminal_DAM = DifferentialActionModelCartpole(isTerminal=True)
    terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

    # OCP
    problem = crocoddyl.ShootingProblem(x0, [running_model] * T, terminal_model)

    # Define warm start (feasible)
    # us = [u0] * T
    # xs = problem.rollout(us)

    xs = [x0] * (T + 1)
    us = [np.zeros(1)] * T
    
    # Define solver
    solver = mim_solvers.SolverCSQP(problem)
    solver.termination_tolerance = 1e-4
    solver.with_callbacks = True 
    solver.eps_abs = 1e-10
    solver.eps_rel = 0.
    solver.use_filter_line_search = False
    # solver.extra_iteration_for_last_kkt = True
    # Solve
    max_iter = 500
    solver.setCallbacks([mim_solvers.CallbackVerbose()])
    solver.solve(xs, us, max_iter, isFeasible=False)

    x_traj = np.array(solver.xs)
    u_traj = np.array(solver.us)

    # %%capture
    # %matplotlib inline

    # Create animation
    anim = animateCartpole(solver.xs)

    # HTML(anim.to_jshtml())
    HTML(anim.to_html5_video())

    import matplotlib.pyplot as plt 

    time_lin = np.linspace(0, dt * (T + 1), T+1)

    fig, axs = plt.subplots(4)
    for i in range(4):
        axs[i].plot(time_lin, x_traj[:, i])
        axs[i].grid()
    fig.suptitle("State trajectory")

    plt.figure()
    plt.plot(time_lin[:-1], u_traj[:])
    plt.title("Control trajectory")
    plt.grid()

    plt.show()

