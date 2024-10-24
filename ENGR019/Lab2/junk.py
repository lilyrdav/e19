import numpy as np
from numpy import dot
from numpy.linalg import inv as invert
import matplotlib.pyplot as plt

def get_4n4n_matrix(n):
    ''' Constructs a 4n x 4n matrix that will solve for the equation
        Ax = b, where 'x' is a vector of the (unknown) coefficients of n
        cubic polynomials, arranged as a column vector in the form
        x = [a_0 b_0 c_0 d_0 a_1 b_1 c_1 d_1 ... a_{n-1} b_{n-1} c_{n-1} d_{n-1}].
        This function does NOT care about the right-hand side of the equation.
        A sample matrix A has been 'hard-coded' for you. Comment this out,
        and write code that returns a 4n x 4n matrix for any value of n. '''
    A_mat = np.array([[0,0,2,0,0,0,0,0,0,0,0,0],
                      [1,0,0,0,0,0,0,0,0,0,0,0],
                      [1,1,1,1,0,0,0,0,0,0,0,0],
                      [0,1,2,3,0,-1,0,0,0,0,0,0],
                      [0,0,2,6,0,0,-2,0,0,0,0,0],
                      [0,0,0,0,1,0,0,0,0,0,0,0],
                      [0,0,0,0,1,1,1,1,0,0,0,0],
                      [0,0,0,0,0,1,2,3,0,-1,0,0],
                      [0,0,0,0,0,0,2,6,0,0,-2,0],
                      [0,0,0,0,0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,0,1,1,1,1],
                      [0,0,0,0,0,0,0,0,0,0,2,6]],
                     dtype=np.float64)

    return A_mat

def get_4n_rhs(y_values):
    ''' Constructs the right-hand side vector b for the 4n x 4n system of equations
        Ax = b, where b contains either zeros or some elements from y_values. The
        input vector, y_values, is a size (n+1) column vector containing the y
        locations for n+1 points. Recall that the x locations are just assumed to be
        0,1,2,3,...,n. A sample r.h.s. has been 'hard-coded' for you. Comment this out,
        and write code that returns a size 4n column vector containing the correct
        r.h.s. for any set of points given by y_values. '''
    rhs = np.array([[0.],
                    [1.2],
                    [3.4],
                    [0.],
                    [0.],
                    [3.4],
                    [-0.6],
                    [0.],
                    [0.],
                    [-0.6],
                    [2.3],
                    [0.]])

    return rhs

def get_4n_coeffs(matrix,rhs):
    ''' This function takes as input a matrix (4n x 4n) and a r.h.s. vector (4n x 1)
        and uses them to solve the system Ax = b for the coefficients of n
        cubic polynomials, arranged as a column vector in the form
        x = [a_0 b_0 c_0 d_0 a_1 b_1 c_1 d_1 ... a_{n-1} b_{n-1} c_{n-1} d_{n-1}].
        You may use any method you like to solve the linear system; for now, the
        correct answer corresponding to the values given in the starter code
        is hard-coded in; comment this out and write code that
        solves Ax=b using 'matrix' as A and 'rhs' as b. '''

    coeffs = np.array([[ 1.2       ],
                       [ 4.31333333],
                       [ 0.        ],
                       [-2.11333333],
                       [ 3.4       ],
                       [-2.02666667],
                       [-6.34      ],
                       [ 4.36666667],
                       [-0.6       ],
                       [-1.60666667],
                       [ 6.76      ],
                       [-2.25333333]])
    return coeffs

# ---- Set points through which to draw the spline ---- #
y_values = np.array([1.2,3.4,-0.6,2.3])

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

