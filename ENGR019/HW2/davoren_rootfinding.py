import math
from numpy import sign, cos, pi, exp, sin
import numpy

def bisection(f,x1,x2,tol):
    # Carries out the bisection algorithm for the function f, using the
    # bracket (x1,x2) with a 'tolerance' of tol.
    f1 = f(x1)
    f2 = f(x2)
    if sign(f1) == sign(f2):
        # if the sign of f(x1) and f(x2) is the same,
        # there is probably no root between x1 and x2.
        print('Root is not bracketed between x1 and x2')
        return None
    
    er = 1 # some large value
    iters = 0
    while er > tol:
        iters+=1
        # uncomment the following line for debugging purposes
        print(f'Root is between x1 = {x1:.3f} and x2 = {x2:.3f}')
        x3 = 0.5*(x1 + x2)
        f3 = f(x3)
        
        if f3 == 0.0:
            return x3
        if sign(f3) != sign(f2):
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3
            
        er = abs(x1-x2)
    return 0.5*(x1 + x2), iters

def secant(f,x1,x2,tol):
    er = 1
    iteration = 0
    while er > tol:
        iteration+=1
        x3 = x2 - (f(x2)*(x2-x1))/(f(x2)-f(x1))
        er = abs(x3-x2)
        x1 = x2
        x2 = x3
    return x3, iteration

def newton(f,df,x0,tol):
    num_iter = 0
    er = 1
    x_current = x0
    while er > tol:
        x_new = x_current - f(x_current)/df(x_current)
        er = abs(x_current-x_new)
        x_current = x_new
        num_iter+=1
    return 0.0, num_iter

def newton_n(f,df,x0,n):
    iteration = 0
    current = x0
    for i in range(0, n):
        new = current - (f(current)/df(current))
        current = new
        iteration+=1
    return current, iteration
#---- don't change below this line ----#

a = bisection(cos,0.9,pi,0.001)
print(f'bisection found the root of cos(x) at {a[0]:.6f} in {a[1]} iterations')
b = bisection(lambda x: (x-2)*(x+1),0.8,3.1,0.0001)
print(f'bisection found the root of (x-2)*(x+1) at {b[0]:.6f} in {b[1]} iterations')

c = secant(cos,0.9,pi,0.001)
print(f'secant method found the root of cos(x) at {c[0]:.6f} in {c[1]} iterations')
d = secant(lambda x: (x-2)*(x+1),0.8,3.1,0.0001)
print(f'secant method found the root of (x-2)*(x+1) at {d[0]:.6f} in {d[1]} iterations')

e = newton(lambda x: cos(x), lambda x: -sin(x),0.9,0.001)
print(f'Newton\'s method found the root of cos(x) at {a[0]:.6f} in {e[1]} iterations')
f = newton(lambda x: (x-2)*(x+1),lambda x:-1 + 2*x,0.8,0.0001)
print(f'Newton\'s method found the root of (x-2)*(x+1) at {b[0]:.6f} in {f[1]} iterations')

for i in range(1, 7):
    example2 = newton_n(lambda x: cos(x), lambda x: -sin(x), 1, i)
    print(f'Newton\'s method found the root of cos(x) at {example2[0]:.16f} in {example2[1]} iterations')
    print(f'The true relative error is {abs((example2[0] - (numpy.pi/2))/(numpy.pi/2))}')

    if i > 1:
        approx = abs((example2[0] - example1[0])/example2[0])
        print(f'The approximate error is {approx:.16f}')
    elif i == 1:
        approx = abs((example2[0] - 1)/example2[0])
        print(f'The approximate error is {approx:.16f}')
    example1 = example2