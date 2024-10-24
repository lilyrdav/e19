import numpy as np

def Gaussian_solve(A,b):
    # Determine the size of the system
    n = len(b)

    # Implement conditions on the size/shape of the input matrices
    if A.shape[0] != A.shape[1]:
        print('Sorry! This does not work.')
        return None
    if A.shape[0] != b.shape[0]:
        print('Sorry this does not work.')
        return None
    
    # Initialize your 'x' vector that will contain your solution
    x = np.zeros((n,1))
    
    # Write code that performs row operations on A and b
    # and eliminates some terms from A in the process.
    for first in range(n-1):
        for sec in range(first+1,n):
            if A[sec][first] != 0:
                factor = A[sec,first]/A[first,first]
                A[sec] = A[sec] - (factor*A[first])
                b[sec] = b[sec] - (factor*b[first])

    # Next, write code that performs back-substitution using
    # the formula given in class.

    for thing in range(n-1,-1,-1):
        if A[thing,thing] != 0:
            b[thing] = (b[thing] - np.dot(A[thing,thing+1:n],x[thing+1:n]))/A[thing,thing]
            x = b

    return x

# Tests your code with the matrix and r.h.s. from problem 2
test_A = np.array([[2,1,-1,0],
                  [1,1,2,0],
                  [-1,2,1,1],
                  [6,1,1,-2]], dtype = 'float')

test_b = np.array([[1],
                   [-1],
                   [0],
                   [2]], dtype = 'float')

print(Gaussian_solve(test_A,test_b))
