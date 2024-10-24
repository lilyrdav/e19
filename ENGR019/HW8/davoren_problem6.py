import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def f_prime(x):
    return np.cos(x)

def second_order_central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def fourth_order_central_diff(f, x, h):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

# Point of interest
x_point = 3.0

# Step sizes
h_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

# Errors
errors_2nd_order = []
errors_4th_order = []

# True derivative value at x_point
true_value = f_prime(x_point)

# Compute errors
for h in h_values:
    approximation_2nd_order = second_order_central_diff(f, x_point, h)
    approximation_4th_order = fourth_order_central_diff(f, x_point, h)

    error_2nd_order = np.abs(true_value - approximation_2nd_order)
    error_4th_order = np.abs(true_value - approximation_4th_order)

    errors_2nd_order.append(error_2nd_order)
    errors_4th_order.append(error_4th_order)

# Plotting the errors on a log-log scale
# Plotting the errors on a log-log scale
plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors_2nd_order, 'ro-', label='2nd order')
plt.loglog(h_values, errors_4th_order, 'bs-', label='4th order')
plt.xlabel('Step size h')
plt.ylabel('Error')
plt.title('Error in F.D. approximations to f\'(x) at x = 3.0 for f(x) = sin(x)')
plt.legend()
plt.grid()

# Set x-axis limits
plt.xlim(1.5, 0.0005)

plt.savefig('error_plot.png')
plt.show()
