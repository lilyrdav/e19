# Assuming you have powell.py and goldSearch.py in the same directory
from powell import powell
import numpy as np

# Define the function you want to minimize
def f(x_vector):
    x = x_vector[0]
    y = x_vector[1]
    z = x_vector[2]
    return (x - 1)**2 + (y + 1)**2 + (z - 2)**2 + x * y * np.sin(z)

# Set your starting point x0 as [0, 0, 0]
x0 = np.array([0.0, 0.0, 0.0])

# Use Powell's method to find the minimum
x_min, n_cycles = powell(f, x0)

# Print out the results
print(f"After {n_cycles} cycles, the minimum point found is {x_min}.")
