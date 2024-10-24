import numpy as np
import matplotlib.pyplot as plt

# Define c
c = 0.5

# Define domain
delta_x     = 0.1
delta_t     = 0.05
final_t     = 3.0
x_values    = np.arange(0,2*np.pi+delta_x,delta_x)
t_values    = np.arange(0,final_t+delta_t,delta_t)

n = len(x_values)
m = len(t_values)

# Initialize empty vector to contain solution
phi_values = np.zeros((n,m))

# set initial condition
phi_values[:,0] = np.exp(-(x_values-np.pi)**2)

# time march
for j in range(1,m):
    for i in range(1, n - 1):
       phi_values[i,j] = phi_values[i, j-1] - c * delta_t / (2 * delta_x) * (phi_values[i+1, j-1] - phi_values[i-1, j-1])

    phi_values[0, j] = phi_values[0, j-1] - c * delta_t / delta_x * (phi_values[1, j-1] - phi_values[0, j-1])  # Forward diff at x=0
    phi_values[-1, j] = phi_values[-1, j-1] - c * delta_t / delta_x * (phi_values[-1, j-1] - phi_values[-2, j-1])  # Backward diff at x=2π

# Function to plot the results
def plot_results(phi_values, x_points, t_points, delta_t):
    plt.figure(figsize=(12, 6))
    
    for t_index in [0, int(1/delta_t), int(2/delta_t), int(3/delta_t)]:
        plt.plot(x_points, phi_values[:, t_index], label=f't = {t_index * delta_t:.2f}')

    plt.legend()
    plt.title('Numerical solution of the wave equation')
    plt.xlabel('x')
    plt.ylabel('φ(t, x)')
    plt.grid(True)
    plt.show()

    plt.figure(2)
    plt.contour(t_values,x_values,phi_values)
    plt.show()

plot_results(phi_values, x_values, t_values, delta_t)