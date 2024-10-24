import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def second_order_central_diff(f, x, h):
    # Handling the boundaries by returning None
    df = np.zeros_like(x)
    df[0] = None
    df[-1] = None
    for i in range(1, len(x) - 1):
        df[i] = (f(x[i + 1]) - f(x[i - 1])) / (2 * h)
    return df

def fourth_order_central_diff(f, x, h):
    # Handling the boundaries by returning None
    df = np.zeros_like(x)
    df[0] = None
    df[1] = None
    df[-2] = None
    df[-1] = None
    for i in range(2, len(x) - 2):
        df[i] = (-f(x[i + 2]) + 8 * f(x[i + 1]) - 8 * f(x[i - 1]) + f(x[i - 2])) / (12 * h)
    return df

# Initialize figure and specs
plt.figure(figsize=(14, 6))
h_list = [1.0, 0.5, 0.1]
specs = ['rx-','r+-','r.-']

# Loop over h values for plotting
for idx, h in enumerate(h_list):
    x_grid = np.arange(0, 2 * np.pi, h)
    
    # Second Order
    plt.subplot(1, 2, 1)
    df_grid_2nd = second_order_central_diff(f, x_grid, h)
    plt.plot(x_grid, df_grid_2nd, specs[idx], label=f'h = {h}')
    
    # Fourth Order
    plt.subplot(1, 2, 2)
    df_grid_4th = fourth_order_central_diff(f, x_grid, h)
    plt.plot(x_grid, df_grid_4th, specs[idx], label=f'h = {h}')

# Plot the function itself
x_fine = np.linspace(0, 2 * np.pi, 300)
plt.subplot(1, 2, 1)
plt.plot(x_fine, f(x_fine), 'b-', label='f(x)')
plt.title("f'(x) with 2nd order centered Finite Differences")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x_fine, f(x_fine), 'b-', label='f(x)')
plt.title("f'(x) with 4th order centered Finite Differences")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('finite_differences.png')
plt.show()