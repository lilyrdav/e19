import numpy as np
from numpy import dot
from numpy.linalg import inv as invert
import matplotlib.pyplot as plt

def get_4n4n_matrix(n):
    A = np.zeros((4*n, 4*n), dtype=np.float64)
    
    for i in range(n):
        base = 4 * i
        
        # First row. Second derivative at start of curve is zero
        if i == 0:
            A[base, base + 2] = 2
        
        A[base + 1, base] = 1
        
        # If not the last block
        if i < n - 1:
            A[base + 2, base:base + 4] = [1, 1, 1, 1]
            A[base + 3, base + 1:base + 6] = [1, 2, 3, 0, -1]
            
            # Add the second derivative continuity constraint
            A[base + 4, base + 2:base + 7] = [2, 6, 0, 0, -2]
        
        else:  # Last block
            A[base + 2, base:base + 4] = [1, 1, 1, 1]
            # Second derivative at the end of curve is zero.
            A[base + 3, base + 2:base + 4] = [2, 6]

    return A

def get_4n_rhs(y_values):
    n = len(y_values) - 1
    rhs = np.zeros((4*n, 1))
    
    for i in range(n):
        # Start of the block
        base = 4 * i
        
        # Second row
        rhs[base + 1] = y_values[i]
        
        # Third row
        rhs[base + 2] = y_values[i + 1]
        
        # The first and last rows of each block are always 0, so no need to set them.

    return rhs

def get_4n_coeffs(matrix,rhs):
    coeffs = np.linalg.solve(matrix, rhs)
    return coeffs

def get_4n4n_matrix_diff_boundary(n):
    A = np.zeros((4*n, 4*n), dtype=np.float64)
    
    for i in range(n):
        base = 4 * i
        
        # First two rows
        if i == 0:
            A[base, base + 1] = 1  # Changed for zero gradient
        
        A[base + 1, base] = 1
        
        # If not the last block
        if i < n - 1:
            A[base + 2, base:base + 4] = [1, 1, 1, 1]
            A[base + 3, base + 1:base + 6] = [1, 2, 3, 0, -1]
            
            # Add the second derivative continuity constraint
            A[base + 4, base + 2:base + 7] = [2, 6, 0, 0, -2]
        
        else:  # Last block
            A[base + 2, base:base + 4] = [1, 1, 1, 1]
            A[base + 3, base + 1:base + 4] = [1, 2, 3]  # Changed for zero gradient --> f'(x)=b+2cx+3dx^2
 
    return A

# Generate 20 y-values
x_values = np.linspace(0, 2*np.pi, 20)
y_values = np.sin(x_values) + 0.1 * np.random.randn(20)  # Sine curve but a little sillier

# ---- Make spline (shouldn't need to change this part) ---- #
n = y_values.shape[0]-1
system_matrix = get_4n4n_matrix_diff_boundary(n)
rhs_vector = get_4n_rhs(y_values)
coeffs = get_4n_coeffs(system_matrix,rhs_vector).reshape((n,4))

u_values = np.arange(n+1)
u = np.linspace(0, n, 100*n + 1)
i = np.minimum(np.floor(u).astype(np.int32), n-1)
t = u - i 
ai = coeffs[i, 0]
bi = coeffs[i, 1]
ci = coeffs[i, 2]
di = coeffs[i, 3]
y = ai + t * (bi + t * (ci + t * di))

plt.plot(u_values, y_values, 'ro', label='Original data points',zorder=2)
plt.plot(u, y, 'b-', label='Interpolation curve', zorder=1)
plt.legend()
plt.grid()
plt.show()

# ---- Set points through which to draw the spline ---- #
y_values = np.array([1.2, 1.4, -0.6, 2.3, 0.0, 1.1, -2.3])

# ---- Make spline (shouldn't need to change this part) ---- #
n = y_values.shape[0]-1
system_matrix = get_4n4n_matrix(n)
rhs_vector = get_4n_rhs(y_values)
coeffs = get_4n_coeffs(system_matrix,rhs_vector).reshape((n,4))

u_values = np.arange(n+1)
u = np.linspace(0, n, 100*n + 1)
i = np.minimum(np.floor(u).astype(np.int32), n-1)
t = u - i 
ai = coeffs[i, 0]
bi = coeffs[i, 1]
ci = coeffs[i, 2]
di = coeffs[i, 3]
y = ai + t * (bi + t * (ci + t * di))

plt.plot(u_values, y_values, 'ro', label='Original data points',zorder=2)
plt.plot(u, y, 'b-', label='Interpolation curve', zorder=1)
plt.legend()
plt.grid()
plt.show()

n = y_values.shape[0]-1
system_matrix = get_4n4n_matrix_diff_boundary(n)
rhs_vector = get_4n_rhs(y_values)
coeffs = get_4n_coeffs(system_matrix,rhs_vector).reshape((n,4))

u_values = np.arange(n+1)
u = np.linspace(0, n, 100*n + 1)
i = np.minimum(np.floor(u).astype(np.int32), n-1)
t = u - i 
ai = coeffs[i, 0]
bi = coeffs[i, 1]
ci = coeffs[i, 2]
di = coeffs[i, 3]
y = ai + t * (bi + t * (ci + t * di))

plt.plot(u_values, y_values, 'ro', label='Original data points',zorder=2)
plt.plot(u, y, 'b-', label='Interpolation curve', zorder=1)
plt.legend()
plt.grid()
plt.show()