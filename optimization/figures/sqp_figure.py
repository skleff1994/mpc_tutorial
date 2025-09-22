# SQP visual: Rosenbrock contours + local quadratic (Hessian-based) approximations and iteration arrows
import numpy as np
import matplotlib.pyplot as plt

a = 1.
b = 10.

# Rosenbrock function, gradient, Hessian
def f(x, y, a=a, b=b):
    return (a - x)**2 + b*(y - x**2)**2

def grad(x, y, a=a, b=b):
    dfdx = -2*(a - x) - 4*b*x*(y - x**2)
    dfdy = 2*b*(y - x**2)
    return np.array([dfdx, dfdy])

def hess(x, y, a=a, b=b):
    d2fdx2 = 2 - 4*b*(y - 3*x**2)
    d2fdy2 = 2*b
    d2fdxdy = -4*b*x
    H = np.array([[d2fdx2, d2fdxdy],[d2fdxdy, d2fdy2]])
    return H

def sqp_rosenbrock(x0, max_iter=100, tol=1e-6):
    path = [x0]
    xk = x0.copy()
    
    for _ in range(max_iter):
        g = grad(*xk)
        H = hess(*xk)

        # Regularize Hessian to avoid numerical issues (ensure positive definiteness)
        eps = 1e-6
        H = 0.5 * (H + H.T)
        eigvals = np.linalg.eigvalsh(H)
        if np.any(eigvals <= 0):
            H += np.eye(2) * (eps - np.min(eigvals))

        # Newton step: solve H p = -g
        pk = -np.linalg.solve(H, g)

        # Line search (backtracking)
        alpha = 1.0
        while f(*(xk + alpha * pk)) > f(*xk) + 1e-4 * alpha * g @ pk:
            alpha *= 0.5
            if alpha < 1e-6:
                break

        xk = xk + alpha * pk
        path.append(xk.copy())

        if np.linalg.norm(pk) < tol:
            break

    return np.array(path)


def plot_qp_step(xk, step_id, ax=None):
    g = grad(*xk)
    H = hess(*xk)
    H = 0.5 * (H + H.T)
    
    # Solve QP: min_p g^T p + 0.5 p^T H p
    pk = -np.linalg.solve(H, g)

    # Plot setup
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    # Define local grid around p = 0
    if(step_id==0):
        grid_range = 5.0
    else:
        grid_range = 1.0
    res = 100
    p1 = np.linspace(-grid_range, grid_range, res)
    p2 = np.linspace(-grid_range, grid_range, res)
    P1, P2 = np.meshgrid(p1, p2)

    Q = np.zeros_like(P1)
    for i in range(res):
        for j in range(res):
            p = np.array([P1[i,j], P2[i,j]])
            Q[i,j] = g @ p + 0.5 * p @ H @ p

    # Contour of q(p)
    cs = ax.contour(P1, P2, Q, levels=20, cmap='Greens_r')
    print(xk)
    # Plot origin (p=0) and model minimizer p_k 
    ax.plot(0, 0, marker='o', color='#3C32A0', label='Current iterate', markersize=10)
    ax.plot(pk[0], pk[1], 'g*', label='QP minimizer $p_k$', markersize=12)
    ax.arrow(0, 0, pk[0], pk[1], head_width=0.07, color='red', length_includes_head=True, linewidth=2)
    # ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(-0.5, 2.0)
    ax.set_title(f'QP Model at Step {step_id}')
    ax.set_xlabel('$p_1$')
    ax.set_ylabel('$p_2$')
    ax.set_aspect('equal')
    ax.legend()
    return ax


# Generate contour grid
x = np.linspace(-1.5, 1.5, 300)
y = np.linspace(-0.5, 2.0, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Initial point
x0 = np.array([-1., 1.0])
# Get SQP iterates
pts = sqp_rosenbrock(x0, max_iter=10)

# Create figure (single plot, no subplots)
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# Plot Rosenbrock contours
levels = np.geomspace(1e-2, 1e3, 20)
# cs = ax.contour(X, Y, Z, levels=levels, linewidths=0.8)
cs = ax.contour(X, Y, Z, levels=levels, cmap='Greens_r', linewidths=1)

# Plot iteration points and arrows
# ax.plot(pts[:,0], pts[:,1], marker='o', linestyle='-')
# Plot all intermediate points as small black dots connected by lines
ax.plot(pts[:,0], pts[:,1], marker='o', linestyle='None', markersize=4, color="#000000")

# Highlight initial point (blue dot, larger)
# ax.plot(pts[0,0], pts[0,1], marker='o', color='#3C32A0', markersize=8, label='Initial point')

# Highlight final point (red star)
ax.plot(pts[-1,0], pts[-1,1], marker='*', color='#3C32A0', markersize=20, label='Optimum')

for i in range(len(pts)-1):
    ax.annotate(
        '',
        xy=tuple(pts[i+1]),
        xytext=tuple(pts[i]),
        arrowprops=dict(arrowstyle='->', lw=1.2, alpha=1./(i+1), color='red')
    )
    ax.text(pts[i,0]+0.03, pts[i,1]+0.05, f'k={i}', fontdict={'size': 18, 'color': '#3C32A0', 'alpha': 1./(i+1)})

ax.text(pts[-1,0]+0.03, pts[-1,1]+0.05, 'Optimum', fontdict={'size': 18, 'color': '#3C32A0'})

# Draw local quadratic approximations as ellipse level-sets around first three points
def draw_quadratic_level(ax, xk, yk, level=1.0):
    g = grad(xk, yk)
    H = hess(xk, yk)
    # Ensure symmetric
    H = 0.5*(H + H.T)
    # Construct ellipse from (p - p0)^T H (p - p0) = level
    # Get eigen-decomposition
    vals, vecs = np.linalg.eigh(H)
    # Avoid negative or tiny eigenvalues for plotting; shift if necessary
    eps = 1e-6
    vals = np.clip(vals, eps, None)
    theta = np.linspace(0, 2*np.pi, 200)
    # ellipse axes lengths proportional to sqrt(level/vals)
    axes = np.sqrt(level/vals)
    E = (vecs @ np.diag(axes) @ np.vstack([np.cos(theta), np.sin(theta)])).T
    Ex = E[:,0] + xk
    Ey = E[:,1] + yk
    ax.plot(Ex, Ey, linewidth=1, color='#3C32A0')

# choose levels scaled to show different bowls
levels_q = [0.05, 0.03, 0.02]
for (xk, yk), lvl in zip(pts[:-1], levels_q):
    draw_quadratic_level(ax, xk, yk, level=lvl)

# Cosmetic labels (no specific colors set)
ax.set_title("Sequential Quadratic Programming (SQP)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 2.0)
ax.grid(False)
# # Save and also display
# out_path = "sqp_visual.png"
plt.tight_layout()
# plt.savefig(out_path)
# # out_path

# Show QP models for first 3 steps
fig, axs = plt.subplots(1, 5, figsize=(15, 4))
for k in range(5):
    plot_qp_step(pts[k], step_id=k, ax=axs[k])
plt.tight_layout()
plt.show()