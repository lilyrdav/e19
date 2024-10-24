from numpy import zeros
import numpy as np
import matplotlib.pyplot as plt

def forward_euler(func, y0, tspan, n):
    start, end = tspan
    h = (end - start) / n
    tvals = np.linspace(start, end, n)
    yvals = np.zeros(n)
    yvals[0] = y0
    for i in range(1, n):
        yvals[i] = yvals[i - 1] + h * func(tvals[i - 1], yvals[i - 1])
    return tvals, yvals

def diff_eq(t, y):
    return np.exp(-t)

# Define initial condition, time span, and number of steps
y0 = 1
tspan = [0, 5]  # Time span from 0 to 5
n = 25        # Number of steps
n1 = 7

# Solve the initial value problem
t_values, y_values = forward_euler(diff_eq, y0, tspan, n)
t_values1, y_values1 = forward_euler(diff_eq, y0, tspan, n1)

# Plot the solution
plt.figure(1)
plt.plot(t_values, y_values, 'bo-')
plt.plot(t_values1, y_values1, 'r-')
plt.grid()
plt.legend(['step size 0.2','default step size'])
plt.show()


