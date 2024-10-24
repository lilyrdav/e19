from scipy.optimize import minimize_scalar as minimize
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import minimize_scalar

def naive_optimize(f, x0, m):
    n = x0.size  # Determine the dimensionality from x0
    directions = np.eye(n)  # Create n non-repeating direction vectors
    errors = np.zeros(m)  # Initialize the error array
    
    x = x0.copy()  # Start from the initial guess
    
    for i in range(m):
        # Cycle through direction vectors
        direction = directions[i % n]
        
        # Define a one-dimensional function along the direction vector
        def f_along_direction(alpha, x=x, direction=direction):
            return f(x + alpha * direction)
        
        # Perform scalar minimization starting from alpha = 0
        result = minimize_scalar(f_along_direction)
        alpha_min = result.x
        
        # Update the guess x using the minimizing alpha along the direction
        x_new = x + alpha_min * direction
        
        # Calculate the approximate relative error
        if i > 0:  # Skip the first iteration as we don't have a previous value to compare
            errors[i] = np.linalg.norm(x_new - x) / np.linalg.norm(x)
        
        # Update the current guess to the new value
        x = x_new
    
    return x, errors

# --- This part tests your function on the function from eq.(1) of HW 7.
# --- You shouldn't need to change anything here.
# --- If your function works correctly, this should calculate the correct answer
# --- for the minimization of the function from eq.(1).

def test_function(x_vector):
    ''' Implements the function f([x,y]) = (x-1)^2 + (y+1)^2 + xy '''
    x = x_vector[0]
    y = x_vector[1]
    return (x-1)**2 + (y+1)**2 + x*y

a,apprx_rel_errs = naive_optimize(test_function,np.array([-1,2]),30)
print('test_function optimized at',a)

plt.figure(1)
plt.plot(apprx_rel_errs,'ro-')
plt.title('Approx. Relative Error using naive 2-d optimization')
plt.grid()
plt.xlabel('Step number')
plt.ylabel('Error')
plt.yscale('log')
plt.savefig('sample_naive_2d.png')
plt.show()

# --- This part tests that your function works on a 3-d function. You shouldn't
# --- need to change anything here.
def f3(x_vector):
    x = x_vector[0]
    y = x_vector[1]
    z = x_vector[2]
    return (x-1)**2 + (y+1)**2 + (z-2)**2 + x*y* np.sin(z)

b,apprx_rel_errs2 = naive_optimize(f3,np.array([0,0,0]),30)
print('f3 optimized at',b)

plt.figure(2)
plt.plot(apprx_rel_errs2,'ro-')
plt.title('Approx. Relative Error using naive 3-d optimization')
plt.grid()
plt.xlabel('Step number')
plt.ylabel('Error')
plt.yscale('log')
plt.savefig('sample_naive_3d.png')
plt.show()

assert len(b) == 3, 'Your program doesn''t work properly for 3 dimensions yet!'