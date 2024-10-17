import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the function and its derivative
def f(x):
    return x**4 - 3 * x**3 + 2

def df(x):
    return 4 * x**3 - 9 * x**2

# Newton-Raphson method
def newton_raphson(start, iterations):
    x_n = start
    x_history = [x_n]
    for _ in range(iterations):
        x_n = x_n - f(x_n) / df(x_n)
        x_history.append(x_n)
    return x_history

# Parameters
start = 2  # Initial guess near local minimum
# start = 2.5 # Initial guess near local minimum
iterations = 10

# Calculate the iterations
x_history = newton_raphson(start, iterations)

# Prepare for animation
x = np.linspace(-1, 5, 400)
y = f(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y, label='f(x) = x^4 - 3x^3 + 2')
ax.axhline(0, color='grey', lw=0.5, ls='--')
ax.axvline(0, color='grey', lw=0.5, ls='--')
points, = ax.plot([], [], 'ro', label='Iterations')
tangent_line, = ax.plot([], [], 'g--', label='Tangent Line')
ax.set_xlim(-1, 4)
ax.set_ylim(-7, 7)
ax.legend()
import time
# Animation function
def animate(i):
    # Update points
    points.set_data(x_history[:i+1], f(np.array(x_history[:i+1])))
    
    # Calculate tangent line
    x_n = x_history[i]
    slope = df(x_n)
    tangent_x = np.linspace(x_n - 1, x_n + 1, 100)
    tangent_y = f(x_n) + slope * (tangent_x - x_n)
    tangent_line.set_data(tangent_x, tangent_y)
    # tangent_line.set_data([], [])  # Clear tangent line on first frame

    return points, tangent_line

# Create animation
ani = FuncAnimation(fig, animate, frames=len(x_history), interval=1000, blit=True)

plt.show()
