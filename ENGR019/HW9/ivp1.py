from scipy.integrate import solve_ivp
from numpy import array,exp
import matplotlib.pyplot as plt

def f1(t,x):
    return exp(-t)                          # (b)

y_init1     = array([1])                    # (c)
solution1   = solve_ivp(f1,[0,5],y_init1, first_step=0.2, max_step=0.2)   # (d)
solution2 = solve_ivp(f1,[0,5],y_init1)

plt.figure(1)
plt.plot(solution1.t,solution1.y[0],'bo-')  # (e)
plt.plot(solution2.t,solution2.y[0],'r-')
plt.grid()
plt.legend(['step size 0.2','default step size'])
plt.show()
