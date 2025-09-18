'''
This file is an adaptation of Crocoddyl's cartpole tutorial
The original code can be found here : https://github.com/loco-3d/crocoddyl/blob/devel/examples/notebooks
'''

from math import cos, sin
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

import crocoddyl

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
        nr = 4
        if(hasConstraints and not isTerminal):
            ng = 1
            nh = 0
        else: 
            ng = 0
            nh = 0

        # Cost function parameters
        if(isTerminal):
            self.costWeights = [
                1,    # sin(th)
                1,    # 1-cos(th)
                1e-2, # thdot
                1e-3, # f
            ]  
        else:
            self.costWeights = [
                1,    # sin(th)
                1,    # 1-cos(th)
                1e-2, # thdot
                1e-3, # f
            ]  

        # create action model
        state = crocoddyl.StateVector(self.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr, ng, nh)
        self.unone = np.zeros(self.nu)

        if hasConstraints and not isTerminal:
            u_lim = 5.
            self.g_lb = np.array([- u_lim])
            self.g_ub = np.array([u_lim])

        self.g = 9.81
        self.L = 1

        self.isTerminal = isTerminal
        self.hasConstraints = hasConstraints

    def _running_cost(self, x, u):


        cost = self.x_weights[0] * (1 - np.cos(x[0])) # cos(th)
        cost += self.x_weights[1] * (np.sin(x[0]))    # sin(th)
        cost += self.x_weights[2] * x[1] ** 2         # thdot
        cost += self.u_weight * u[0] ** 2             # u
        return 0.5 * cost

    def _terminal_cost(self, x):
        cost = self.x_weights[0] * (1 - np.cos(x[0])) # cos(th)
        cost += self.x_weights[1] * (np.sin(x[0]))    # sin(th)
        cost + self.x_weights[2] * x[1] ** 2          # thdot
        return 0.5 * cost 

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone

        s, c = np.sin(x[0]), np.cos(x[0])
        if self.isTerminal:
            # Computing the cost residual and value
            data.r[:] = np.array([s, 1 - c, x[1], u[0]]).T
            data.cost = 0.5 * sum(self.costWeights * (np.asarray(data.r) ** 2)).item()
            data.xout = np.zeros(self.state.nx)
        else:
            # Computing the cost residual and value
            data.r = np.array([s, 1 - c, x[1], u[0]]).T
            data.cost = 0.5 * sum(self.costWeights * (np.asarray(data.r) ** 2)).item()
            data.xout = - self.g * np.sin(x[0]) / self.L + u

        if self.hasConstraints and not self.isTerminal:
            data.g[0] = u[0]

    def calcDiff(self, data, x, u=None):
        w = self.costWeights
        s, c = np.sin(x[0]), np.cos(x[0])
        data.Lx[0] = s * ((w[0] - w[1]) * c + w[1])
        data.Lx[1] = w[2] * x[1]
        data.Lxx[0, 0] = w[0] * (c**2 - s**2) + w[1] * (s**2 - c**2 + c)
        data.Lxx[1, 1] = w[2]

        if not self.isTerminal:
            data.Lu[0] = w[3] * u[0]
            data.Luu[0, 0] = w[3]

            data.Fx[0] = - self.g * np.cos(x[0]) / self.L
            data.Fx[1] = 0.
            data.Fu[0] = 1.
            
        data.Lxu = np.zeros([2, 1])

        if self.isTerminal:
            data.Lx   = self.dt * data.Lx
            data.Lxx =  self.dt * data.Lxx

        if self.hasConstraints and not self.isTerminal:
            data.Gx = np.zeros((self.ng, self.nx)) 
            data.Gu[0] = 1.


def createPendulumOptimalControlProblem(x0=np.zeros(2), dt=2e-2, T=100, hasConstraints=False):
    '''
    Creates the crocoddyl.ShootingProblem for pendulum
    '''
    # Create the running models
    running_DAM = DiffActionModelPendulum(isTerminal=False, hasConstraints=hasConstraints)
    running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    # Terminal model
    running_DAM_terminal = DiffActionModelPendulum(isTerminal=True, hasConstraints=hasConstraints)
    running_model_terminal = crocoddyl.IntegratedActionModelEuler(running_DAM_terminal, dt)
    # Create the shooting problem
    return crocoddyl.ShootingProblem(x0, [running_model] * T, running_model_terminal)

def plotPendulumSolution(xs, us, T=100):
    '''
    Plots the solution of the Pendulum OCP
    '''
    x_traj = np.array(xs)
    u_traj = np.array(us)
    plt.figure()
    plt.plot(x_traj[:, 0],  x_traj[:, 1], label='Pendulum')
    plt.plot(x_traj[0,0], x_traj[0,1], 'ro')
    plt.plot(0, 0, 'ro')
    # plt.plot(3 * np.pi, 0, 'ro')
    plt.legend()
    plt.title("phase portrait")   
    plt.xlabel("$\\theta$", fontsize=18)
    plt.ylabel("$\\dot\\theta$", fontsize=18)
    plt.grid()
    # fancy plot with discretization
    fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')
    time_discrete = range(T+1)
    ax1.plot(time_discrete,  x_traj[:, 0], linewidth=3, color='r', marker='.', label='Pendulum position $\\theta$ ($x_1$)')
    ax1.plot(time_discrete,  x_traj[:, 1], linewidth=3, color='g', marker='.', label='Pendulum velocity $\\omega$ ($x_2$)')
    ax1.grid()
    ax1.legend(fontsize=20)
    ax2.step(time_discrete[:-1],  u_traj, where='post', linewidth=3, linestyle=None, color='b', label='Control input (u)')
    ax2.set_xlabel("k", fontsize=20)
    # plt.ylabel("$\\dot\\theta$", fontsize=18)
    ax2.grid()
    ax2.legend(fontsize=20)
    ax2.locator_params(axis='x', nbins=20) 
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def animatePendulum(xs, sleep=50, show=False, mode="html"):
    """
    xs: trajectory (list/array of states)
    sleep: ms per frame
    show: if True, open matplotlib window (only outside Jupyter)
    mode: 'html' for inline animation, 'js' for JavaScript, 'video' for mp4
    """
    print("processing the animation ... ")
    cart_size = 0.1
    pole_length = 5.0

    fig, ax = plt.subplots()
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6, 6)

    patch = plt.Rectangle((0.0, 0.0), cart_size, cart_size, fc="b")
    (line,) = ax.plot([], [], "k-", lw=2)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        ax.add_patch(patch)
        line.set_data([], [])
        time_text.set_text("")
        return patch, line, time_text

    def animate(i):
        x_cart = 0.0
        y_cart = 0.0
        theta = xs[i][1]  # assuming state = [?, theta]
        patch.set_xy([x_cart - cart_size / 2, y_cart - cart_size / 2])
        x_pole = np.cumsum([x_cart, -pole_length * np.sin(theta)])
        y_pole = np.cumsum([y_cart, pole_length * np.cos(theta)])
        line.set_data(x_pole, y_pole)
        time = i * sleep / 1000.0
        time_text.set_text(f"time = {time:.1f} sec")
        return patch, line, time_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True
    )

    print("... processing done")

    if show:
        plt.show()
        return None
    plt.close(fig)
    if mode == "html":
        return HTML(anim.to_html5_video())
    elif mode == "js":
        return HTML(anim.to_jshtml())
    elif mode == "video":
        anim.save("pendulum.mp4", fps=1000/sleep, extra_args=["-vcodec", "libx264"])
        from IPython.display import Video
        return Video("pendulum.mp4")
    else:
        return anim
