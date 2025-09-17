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
    def __init__(self, isTerminal=False, hasConstraints=False):
      
        if(hasConstraints and not isTerminal):
            ng = 1 # inequality constraint dimension
            nh = 0 # equality constraint dimension
        else: 
            ng = 0
            nh = 0

        nu = 1 # control vector dimension
        nr = 6 # cost residual dimension
        self.nx = 4

        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, crocoddyl.StateVector(4), nu, nr, ng, nh)  
        self.unone = np.zeros(self.nu)
        self.isTerminal = isTerminal
        self.hasConstraints = hasConstraints

        if hasConstraints and not isTerminal:
            u_lim = 100.
            self.g_lb = np.array([-u_lim])
            self.g_ub = np.array([u_lim])

        # Dynamics model parameters
        self.m1 = 1.0
        self.m2 = 0.1
        self.l = 0.5
        self.g = 9.81

        # Cost function parameters
        if(self.isTerminal):
            self.costWeights = [
                0,   # sin(th)
                0,   # 1-cos(th)
                0.,   # y
                0., # ydot
                0., # thdot
                0.,   # f
            ]  
        else:
            self.costWeights = [
                0.,   # sin(th)
                0.,   # 1-cos(th)
                0.,   # y
                0., # ydot
                0., # thdot
                0.,   # f
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

        if self.hasConstraints and not self.isTerminal:
            data.g[0] = u[0]

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

        #TODO: implement the analytical derivatives of the cost and dynamics
        # # derivative of xddot by x, theta, xdot, thetadot
        # # derivative of thddot by x, theta, xdot, thetadot
        # data.Fx[:, :] = 
        # # derivative of xddot and thddot by f
        # data.Fu[:] = 
        # # first derivative of data.cost by x, theta, xdot, thetadot
        # data.Lx[:] = 
        # # first derivative of data.cost by f
        # data.Lu[:] = 
        # # second derivative of data.cost by x, theta, xdot, thetadot
        # data.Lxx[:] = 
        # data.Luu[:] =


if __name__ == "__main__":

    # OCP parameters
    x0 = np.array([0.0, 3.14, 0.0, 0.0]).T
    u0 = np.zeros(1)
    T = 100
    dt = 5e-3

    cartpoleDAM = DifferentialActionModelCartpole(isTerminal=False, hasConstraints=True)
    cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)
    cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, dt)

    terminalCartpoleDAM = DifferentialActionModelCartpole(isTerminal=True)
    terminalCartpoleDAM.costWeights[0] = 100
    terminalCartpoleDAM.costWeights[1] = 100
    terminalCartpoleDAM.costWeights[2] = 1.0
    terminalCartpoleDAM.costWeights[3] = 0.1
    terminalCartpoleDAM.costWeights[4] = 0.01
    terminalCartpoleDAM.costWeights[5] = 0.0001
    terminalCartpoleND = crocoddyl.DifferentialActionModelNumDiff(terminalCartpoleDAM, True)
    terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleND)

    problem = crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)

    # Define warm start (feasible)
    # us = [u0] * T
    # xs = problem.rollout(us)
    problem.x0 = x0
    us = [np.zeros(1)] * T
    # xs = problem.rollout(us)
    xs = [x0] * (T + 1)
    
    # Define solver
    solver = mim_solvers.SolverCSQP(problem)

    solver.termination_tolerance = 1e-4
    solver.with_callbacks = True 
    solver.eps_abs = 1e-10
    solver.eps_rel = 0.
    solver.use_filter_line_search = True
    # Solve
    max_iter = 50
    solver.setCallbacks([mim_solvers.CallbackVerbose()])
    solver.solve(xs, us, max_iter, False)

    x_traj = np.array(solver.xs)
    u_traj = np.array(solver.us)

    # Create animation
    anim = animateCartpole(solver.xs)

    import matplotlib.pyplot as plt 

    time_lin = np.linspace(0, dt * (T + 1), T+1)

    # fig, axs = plt.subplots(4)
    # for i in range(4):
    #     axs[i].plot(time_lin, x_traj[:, i])
    #     axs[i].grid()
    # fig.suptitle("State trajectory")

    # plt.figure()
    # plt.plot(time_lin[:-1], u_traj[:])
    # plt.title("Control trajectory")
    # plt.grid()

        # fancy plot with discretization
    fig, ax1 = plt.subplots(1,1, sharex='col')
    time_discrete = range(T+1)
    # ax1.plot(time_discrete,  x_traj[:, 0], linewidth=1, color='r', marker='.', label='Cart position $y$ ($x_1$)')
    # ax1.plot(time_discrete,  x_traj[:, 1], linewidth=1, color='g', marker='.', label='Pole angular position $theta$ ($x_2$)')
    # ax1.plot(time_discrete,  x_traj[:, 2], linewidth=1, color='b', marker='.', label='Cart velocity $y_dot$ ($x_3$)')
    # ax1.plot(time_discrete,  x_traj[:, 3], linewidth=1, color='y', marker='.', label='Pole angular velocity $theta_dot$ ($x_4$)')
    ax1.plot(time_discrete,  x_traj[:, 0], linewidth=1, color='r', marker='.', label='State ($x_1$)')
    ax1.plot(time_discrete,  x_traj[:, 1], linewidth=1, color='g', marker='.', label='State ($x_2$)')
    ax1.plot(time_discrete,  x_traj[:, 2], linewidth=1, color='b', marker='.', label='State ($x_3$)')
    ax1.plot(time_discrete,  x_traj[:, 3], linewidth=1, color='y', marker='.', label='State ($x_4$)')
    ax1.grid()
    ax1.legend(fontsize=20)

    # ax2.step(time_discrete[:-1],  u_traj, where='post', linestyle=None, color='k', label='Action (u)')
    # ax2.set_xlabel("Time step", fontsize=20)
    # # plt.ylabel("$\\dot\\theta$", fontsize=18)
    # ax2.grid()
    # ax2.legend(fontsize=20)
    # ax2.locator_params(axis='x', nbins=20) 



    plt.show()

