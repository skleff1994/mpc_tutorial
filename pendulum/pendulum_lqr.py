import numpy as np
import scipy.sparse as sparse
import osqp
import matplotlib.pyplot as plt

# ---------------------------
# Problem data (same as yours)
# ---------------------------
dt = 0.02
A = np.array([[1., dt],
              [9.81*dt, 1.]])
B = np.array([[0.],
              [dt]])
nx = A.shape[0]
nu = B.shape[1]

Q = np.diag([10., 1.])   # state stage cost
Qf = Q                   # terminal cost (could be different)
R = np.array([[0.01]])    # control cost

# horizon
N = 100

# initial state
x0 = np.array([1.01, -3.0])

# choose bounds (try small to activate constraint)
u_max = 2
u_min = -u_max

# helper to build block-diagonal repeated matrices
def block_diag_repeat(mat, reps):
    mats = [mat for _ in range(reps)]
    return sparse.block_diag(mats, format='csc')

# ---------------------------
# Build QP matrices for OSQP
# Decision z = [x0, x1, ..., xN, u0, ..., u_{N-1}] length nz
# ---------------------------
nx_all = (N+1) * nx
nu_all = N * nu
nz = nx_all + nu_all

# Hessian P (H in math): block diag [Q,...,Q, Qf, R,...,R]
P_x = block_diag_repeat(Q, N)            # Q repeated for x0..x_{N-1}
P_x = sparse.block_diag([P_x, Qf], format='csc')  # add Qf for x_N
P_u = block_diag_repeat(R, N)            # R repeated for u0..u_{N-1}
P = sparse.block_diag([P_x, P_u], format='csc')

# Linear term q (zeros here)
q = np.zeros(nz)

# ---------------------------
# Equality constraints for dynamics:
# For k = 0..N-1: x_{k+1} - A x_k - B u_k = 0   (nx * N rows)
# Initial condition: x0 = x_init                 (nx rows)
# Inequalities for control bounds: u_min <= u_k <= u_max  (nu * N rows)
# We will form A_constr and l, u for OSQP: l <= A z <= u
# ---------------------------

rows = []
cols = []
data = []
row_count = 0

# (1) Initial condition rows: x0 = x_init
for i in range(nx):
    # index of x0 component in z
    col = i
    rows.append(row_count); cols.append(col); data.append(1.0)
    row_count += 1

# (2) Dynamics rows for each stage k: x_{k+1} - A x_k - B u_k = 0
# each is nx rows, run k=0..N-1
for k in range(N):
    # indices in z:
    idx_xk = k * nx           # starting index of x_k in z
    idx_xkp1 = (k+1) * nx     # starting index of x_{k+1}
    idx_uk = nx_all + k * nu  # starting index of u_k in z

    # x_{k+1} term: +1 * x_{k+1}
    for i in range(nx):
        rows.append(row_count + i)
        cols.append(idx_xkp1 + i)
        data.append(1.0)

    # -A * x_k term
    for i in range(nx):
        for j in range(nx):
            val = -A[i, j]
            if abs(val) > 0:
                rows.append(row_count + i)
                cols.append(idx_xk + j)
                data.append(val)

    # -B * u_k term
    for i in range(nx):
        for j in range(nu):
            val = -B[i, j]
            if abs(val) > 0:
                rows.append(row_count + i)
                cols.append(idx_uk + j)
                data.append(val)

    row_count += nx

# (3) Control bounds: for each u_k we add one row that picks the u_k value
#    We'll represent them as inequalities: u_min <= u_k <= u_max
for k in range(N):
    idx_uk = nx_all + k * nu
    for j in range(nu):
        rows.append(row_count); cols.append(idx_uk + j); data.append(1.0)
        row_count += 1

# Now assemble sparse A matrix
A_eq_ineq = sparse.csc_matrix((data, (rows, cols)), shape=(row_count, nz))

# Build l and u vectors:
# For initial condition rows (first nx rows), l = u = x0
l = []
u = []
# initial
for i in range(nx):
    l.append(x0[i])
    u.append(x0[i])

# dynamics rows (next nx * N rows): l=u=0
for _ in range(nx * N):
    l.append(0.0)
    u.append(0.0)

# control bounds rows: l = u_min, u = u_max (same for all)
for _ in range(nu * N):
    l.append(u_min)
    u.append(u_max)

l = np.array(l)
u = np.array(u)

# ---------------------------
# Solve with OSQP
# ---------------------------
# OSQP wants P as csc, q as vector, A as csc, l and u.
prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A_eq_ineq, l=l, u=u, verbose=True, polish=True, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
res = prob.solve()

if res.info.status != 'solved':
    raise RuntimeError("OSQP failed to solve: " + res.info.status)

z = res.x  # decision vector

# Extract state and control trajectories
xs = z[:nx_all].reshape((N+1, nx))
us = z[nx_all:].reshape((N, nu))

print("OSQP solve status:", res.info.status)
print("Max |u| (constrained):", np.max(np.abs(us)))
print("u bounds:", u_min, u_max)

# ---------------------------
# Also solve unconstrained variant (big bounds) for comparison
# ---------------------------
big = 1e6
l_un = l.copy()
u_un = u.copy()
# set control bound rows to [-big, +big] (they are last nu*N rows)
l_un[-(nu*N):] = -big
u_un[-(nu*N):] = +big

prob_un = osqp.OSQP()
prob_un.setup(P=P, q=q, A=A_eq_ineq, l=l_un, u=u_un, verbose=False, polish=True)
res_un = prob_un.solve()
z_un = res_un.x
xs_un = z_un[:nx_all].reshape((N+1, nx))
us_un = z_un[nx_all:].reshape((N, nu))

print("Max |u| (unconstrained):", np.max(np.abs(us_un)))

# Plot results
time_x = np.arange(N+1) * dt
time_u = np.arange(N) * dt

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,6), sharex=True)

# States
ax1.plot(time_x, xs[:,0], '--', linewidth=3, label='theta (constr)')
ax1.plot(time_x, xs[:,1], '--', linewidth=3, label='omega (constr)')
ax1.set_ylabel('State', fontsize=18)
ax1.grid()
ax1.legend(fontsize=18)

# Controls
ax2.step(time_u, us.flatten(), where='post', linewidth=3, label='u (constr)')
# ax2.axhline(u_max, color='r', linestyle='--', linewidth=3, label='u_max')
# ax2.axhline(u_min, color='r', linestyle='--', linewidth=3, label='u_min')
ax2.set_xlabel('Time [s]', fontsize=18)
ax2.set_ylabel('Control u', fontsize=18)
ax2.grid()
# ax2.legend(fontsize=14)
ax2.legend(fontsize=18)

plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import crocoddyl
# import mim_solvers

# from IPython.display import HTML
# from pendulum_utils import animatePendulum

# # Toggle constraints here
# CONSTRAINT = True
# dt = 0.02
# nx, nu = 2, 1
# A = np.array([[1., dt],
#               [9.81*dt, 1.]])
# B = np.array([[0.],
#               [dt]])
# Q = np.diag([10., 1.])   # state weights
# R = np.array([[0.1]])    # control weight
# N = np.matrix(np.zeros((2,1)))  # no cross term

# u_max = 0.2   ### NEW: torque limit

# if CONSTRAINT:
#     # Inequality: G * [x; u] + g <= 0
#     G = np.matrix([[0., 0., 1.],
#                    [0., 0.,-1.]])
#     g = np.matrix([[u_max],    # u <= u_max
#                    [u_max]])   # -u <= u_max → u >= -u_max

#     H = np.matrix(np.zeros((0, 3)))
#     h = np.matrix(np.zeros((0, 1)))
#     f = np.matrix(np.zeros((2, 1)))
#     q = np.matrix(np.zeros((2, 1)))
#     r = np.matrix(np.zeros((1, 1)))

#     lqr_model = crocoddyl.ActionModelLQR(A, B, Q, R, N, G, H, f, q, r, g, h)
# else:
#     lqr_model = crocoddyl.ActionModelLQR(A, B, Q, R, np.zeros((2,1)))

# problem = crocoddyl.ShootingProblem(np.array([1.0, -3.0]), [lqr_model]*100, lqr_model)

# # Initial condition
# x0 = np.zeros(nx) 
# x0[0] = 1 + 0.01
# x0[1] = -3

# # Horizon length
# T = 100
# xs = [x0] * (T + 1)
# us = [np.ones(1)] * T

# # Define solver
# solver = mim_solvers.SolverCSQP(problem)
# solver.termination_tolerance = 1e-4
# solver.with_callbacks = True 
# solver.eps_abs = 1e-10
# solver.eps_rel = 0.
# solver.use_filter_line_search = False

# solver.setCallbacks([mim_solvers.CallbackVerbose()])
# solver.solve(xs, us, 200, False)
# x_traj = np.array(solver.xs)
# u_traj = np.array(solver.us)
# print(x_traj)
# print(u_traj)

# # Animation
# anim = animatePendulum(solver.xs)

# # Time axis
# time_discrete = range(T+1)

# ### PLOTS
# fig, (ax1, ax2) = plt.subplots(2,1, sharex='col', figsize=(8,6))

# # State trajectories
# ax1.plot(time_discrete,  x_traj[:, 0], linewidth=1, color='r', marker='.', label=r'$\theta$ (position)')
# ax1.plot(time_discrete,  x_traj[:, 1], linewidth=1, color='g', marker='.', label=r'$\omega$ (velocity)')
# ax1.grid()
# ax1.legend(fontsize=14)
# ax1.set_ylabel("State", fontsize=14)

# # Control trajectory
# ax2.step(time_discrete[:-1], u_traj, where='post', color='b', label='u (torque)')
# ### NEW: show torque bounds
# if CONSTRAINT:
#     ax2.axhline(u_max, color='k', linestyle='--', linewidth=1, label='u_max')
#     ax2.axhline(-u_max, color='k', linestyle='--', linewidth=1, label='-u_max')
# ax2.set_xlabel("Time step k", fontsize=14)
# ax2.set_ylabel("Control", fontsize=14)
# ax2.grid()
# ax2.legend(fontsize=14)
# ax2.locator_params(axis='x', nbins=20) 

# plt.suptitle("Constrained vs. Unconstrained LQR", fontsize=16)
# plt.tight_layout()
# plt.show()
