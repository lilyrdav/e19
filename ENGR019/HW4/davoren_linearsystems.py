import numpy as np
from numpy import transpose as tr
from numpy import dot
from numpy.linalg import inv as invert
from numpy.linalg import norm
np.set_printoptions(precision=8)

def Gauss_Siedel(A,b,x_guess,tol=1e-6):
    x = x_guess.copy()
    steps = 0
    while 2+2 == 4:
        for i in range(len(x)):
            x[i] = (b[i] - dot(A[i,:i],x[:i]) - dot(A[i,i+1:],x[i+1:]))/A[i,i]
        steps += 1
        if norm(b - np.dot(A,x)) < tol:
            print(f'GS took {steps} steps')
            break
    return x

def steepest_descent(A, b, x_guess, tol=1e-6):
    x = x_guess.copy()
    r = b - np.dot(A, x)
    steps = 0
    while 2+2 == 4:
        a = np.dot(np.transpose(r), r) / np.dot(np.dot(np.transpose(r), A), r)
        x = x + a * r
        r = b - np.dot(A, x)
        steps += 1
        if np.linalg.norm(r) < tol:
            print(f'SD took {steps} steps')
            break
    return x

#A is the coefficient matrix, b is the right-hand side vector, 
# x_guess is the initial guess for the solution,  and tol is 
# the tolerance for convergence (default is 1e-6).
def conjugate_gradient(A,b,x_guess,tol=1e-6):
    x = x_guess.copy() # Make a copy of the initial guess for the solution
    r = b - np.dot(A,x) # Initialize the residual vector as the right-hand vector minus the dot product of the coefficient matrix and the initial guess for the solution
    beta = 0 # Initialize beta
    s = r # Initialize the search direction vector as the residual vector
    steps = 0 # Initialize the number of steps taken

    while 2+2 == 4: 
        a = dot(tr(s),r)/(dot(dot(tr(s),A),s)) # Calculate the scalar using the Conjugate Gradient method
        x = x + a*s # Update the solution estimate 'x' using the scalar 'a' and the search direction vector 's'                           
        r = b - np.dot(A,x) # Calculate the new residual vector 'r'                     
        steps += 1 # Increment the count
        if norm(r) < tol: # Check if the norm of the residual vector is below the tolerance
            print(f'CG took {steps} steps') # If so, print the number of steps taken and break the loop
            break
        beta = - (dot(dot(tr(r),A),s)/
                  (dot(dot(tr(s),A),s))) # Update value of 'beta' using Conjugate Gradient    
        s = r + beta*s # Update the search direction vector                
    return x # Return the final solution estimate 'x'

def is_diagonally_dominant(A):
    for i in range(len(A)):
        if abs(A[i,i]) <= sum(abs(A[i,:])) - abs(A[i,i]):
            return False
    return True

n = 9
# Create 5 arrays containing -4's and 1's of the right length
main_diagonal = np.array([-4] * n)
upper_diagonal = np.array([1] * (n-1))
lower_diagonal = np.array([1] * (n-1))
upppr_diagonal = np.array([1] * (n-3))
lowwr_diagonal = np.array([1] * (n-3))

matrix = (np.diag(main_diagonal) +
          np.diag(upper_diagonal, k=1) +
          np.diag(lower_diagonal, k=-1) +
          np.diag(upppr_diagonal, k=3) +
          np.diag(lowwr_diagonal, k=-3)).astype(float)

rhs = tr(np.array([0,0,100,0,0,100,200,200,300]).astype(float))

x_0 = tr(np.array([1000,0,0,0,0,0,0,0,0]).astype(float))

x_gs = Gauss_Siedel(matrix,rhs,x_0)
x_sd = steepest_descent(matrix,rhs,x_0)
x_cg = conjugate_gradient(matrix,rhs,x_0)
x_tru = dot(invert(matrix),rhs)
print(x_gs)
print(x_sd)
print(x_cg)
print(x_tru)

# Create a 4x4 matrix
A1 = np.array([[42,87,15,63],
              [95,11,29,51],
              [71,6,98,33],
              [76,4,19,88]]).astype(float)

A2 = np.array([[42,18,7,9],
               [15,70,25,31],
               [8,14,82,29],
               [12,16,17,90]]).astype(float)

A3 = np.array([[10,15,20,25],
               [15,30,35,40],
               [20,35,50,55],
               [25,40,55,70]]).astype(float)

A5 = np.array([[7,5],
               [35, 24.99]])

print("A1 diagonal dominance ", is_diagonally_dominant(A1))
print("A2 diagonal dominance ", is_diagonally_dominant(A2))
print("A2 diagonal dominance ", is_diagonally_dominant(A3))

print(np.linalg.cond(A5))