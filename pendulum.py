'''
This file is an adaptation of Armand Jordana's pendulum example
The original code can be found here : https://github.com/ajordana/value_function/blob/main/value_iteration/pendulum.py
'''

import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
import mim_solvers



class DiffActionModelPendulum(crocoddyl.DifferentialActionModelAbstract):
    '''
    This class defines a custom Differential Action Model for the simple pendulum
    It defines the continuous-time
        - dynamics
        - cost
        - constraint
    '''
    def __init__(self, isTerminal=False, hasConstraints=False, dt=0.1):
        self.dt = dt
        self.nq = 1 
        self.nv = 1
        self.ndx = 2
        self.nx = self.nq + self.nv
        nu = 1
        nr = 1 
        if(hasConstraints and not isTerminal):
            ng = 1
            nh = 0
        else: 
            ng = 0
            nh = 0

        # create action model
        state = crocoddyl.StateVector(self.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr, ng, nh)

        if not isTerminal:
            u_lim = 2.
            lower_bound = np.array([- u_lim])
            upper_bound = np.array([u_lim])

            self.g_lb = lower_bound
            self.g_ub = upper_bound

        self.g = 9.81
        self.L = 1

        self.x_weights = [1, 1e-2]
        self.u_weight = 1e-3

        self.isTerminal = isTerminal
        self.hasConstraints = hasConstraints

    def _running_cost(self, x, u):
        cost = self.x_weights[0] * (np.cos(x[0]) + 1)
        cost += self.x_weights[1] * x[1] ** 2 + self.u_weight * u[0] ** 2
        return 0.5 * cost

    def _terminal_cost(self, x):
        cost = self.x_weights[0] * (np.cos(x[0]) + 1)  + self.x_weights[0] * x[1] ** 2
        return 0.5 * cost * self.dt

    def calc(self, data, x, u=None):

        if self.isTerminal:
            data.cost = self._terminal_cost(x)
            data.xout = np.zeros(self.state.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = - self.g * np.sin(x[0]) / self.L + u

        if not self.isTerminal:
            data.g[0] = u[0]

    def calcDiff(self, data, x, u=None):

        data.Lx[0] = 0.5 * self.x_weights[0] * ( - np.sin(x[0])  )
        data.Lx[1] = self.x_weights[1] * x[1]
        data.Lxx[0, 0] = 0.5 * self.x_weights[0]* ( - np.cos(x[0])  )
        data.Lxx[1, 1] = self.x_weights[1]

        if not self.isTerminal:
            data.Lu[0] = self.u_weight * u[0]
            data.Luu[0, 0] = self.u_weight

            data.Fx[0] = - self.g * np.cos(x[0]) / self.L
            data.Fx[1] = 0.
            data.Fu[0] = 1.
            
        data.Lxu = np.zeros([2, 1])

        if self.isTerminal:
            data.Lx   = self.dt * data.Lx
            data.Lxx =  self.dt * data.Lxx

        if not self.isTerminal:
            data.Gx = np.zeros((self.ng, self.nx)) 
            data.Gu[0] = 1.


if __name__ == "__main__":

    nx = 2
    nu = 1
    x0 = np.zeros(nx) 
    x0[0] = 1+ 0.01
    x0[1] = -3

    # Create the running models
    runningModels = []
    dt = 2e-2
    T = 1000
    running_DAM = DiffActionModelPendulum(isTerminal=False)
    running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)

    running_DAM_terminal = DiffActionModelPendulum(isTerminal=True)
    running_model_terminal = crocoddyl.IntegratedActionModelEuler(running_DAM_terminal, dt)

    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, [running_model] * T, running_model_terminal)


    # # # # # # # # # # # # #
    ###     SOLVE OCP     ###
    # # # # # # # # # # # # #

    # Define warm start
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
    max_iter = 200
    solver.solve(xs, us, max_iter)

    x_traj = np.array(solver.xs)
    u_traj = np.array(solver.us)

    import matplotlib.pyplot as plt 

    time_lin = np.linspace(0, dt * (T + 1), T+1)

    fig, axs = plt.subplots(nx)
    for i in range(nx):
        axs[i].plot(time_lin, x_traj[:, i])
        axs[i].grid()
    fig.suptitle("State trajectory")

    plt.figure()
    plt.plot(time_lin[:-1], u_traj[:])
    plt.title("Control trajectory")
    plt.grid()

    plt.show()
