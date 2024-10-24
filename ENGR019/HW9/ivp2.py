from scipy.integrate import solve_ivp
from numpy import array,exp
import matplotlib.pyplot as plt
import numpy as np

def f2(t,x):
    x0 = x[0]
    x1 = x[1]
    dx0 = x1
    dx1 = -(np.cos(t) + (0.5*x1) + (5*x0))
    return [dx0,dx1]

y_init1     = array([0,0])
solution1   = solve_ivp(f2,[0, 20],y_init1, max_step=0.1)

plt.figure(1)
plt.plot(solution1.t,solution1.y[0],'b--', label="x(t)")  # (e)
plt.plot(solution1.t,solution1.y[1],'r-', label="x'(t)")
plt.grid()
plt.legend()
plt.show()
