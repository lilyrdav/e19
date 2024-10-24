import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sin(x)

def integ_trap(f,a,b,n):
    '''
    Integrate function f from a to b using
    the trapezoidal rule with step size h
    '''
    x_grid = np.linspace(a,b,n)
    h = (b-a)/n

    total = 0.0
    for i in range(n):
        if i == 0 or i == n-1:
            total += f(x_grid[i])/2
        else:
            total += f(x_grid[i])
    total *= h
        
    return total

# True value of the integral
true_value = 1

# Different values of n
n_values = [1, 10, 10**2, 10**3, 10**4, 10**5, 10**6]
errors = []

# Calculate integral for different n and store the errors
for n in n_values:
    approx_value = integ_trap(f1, np.pi/2, np.pi, n)
    error = np.abs(true_value - approx_value)
    errors.append(error)

# Plotting the error
plt.loglog(n_values, errors, marker='o')
plt.xlabel('Number of intervals')
plt.ylabel('Error')
plt.title('Error in integrating sin(x) from pi/2 to pi using trapezoidal rule')
plt.grid()
plt.show()