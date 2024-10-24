from scipy.integrate import solve_ivp
from numpy import zeros,exp,linspace,array,sin,cos
from numpy.linalg import norm
import matplotlib.pyplot as plt

def adaptive_euler(func,y0,tspan,initial_dt,abstol):
    '''
    Carries out forward Euler over the
    interval tspan, solving the IVP
    y'(t) = func(y), y(0) = y0. Uses RK2
    to check accuracy and adapt the time step.

    NOTE: This function doesn't work correctly yet!
    '''

    times = []
    yvals = []

    times.append(tspan[0])
    yvals.append(y0)

    dt = initial_dt
    y = y0
    t = tspan[0]

    end = False
    for i in range(1,1000): # either choose a large range or change to while loop
        k1 = func(t,y)
        k2 = func(t+dt/2,y+k1*dt/2)

        dy1 = k1*dt
        dy2 = k2*dt
        
        error = norm(dy2-dy1)
        d_t = dt * (abstol/error) ** (1/2)

        if error <= abstol:
            y = y + dy1
            t = t + dt
            yvals.append(y)
            times.append(t)
            
            if end:
                break

        if t + d_t > tspan[1]:
            d_t = tspan[1] - t
            end = True

        dt = d_t
    
    return times,yvals

def test_f(t,x):
    r = -0.7
    a = 1.5
    omega = 1.5
    return r*x*(1-x)*(1+a*sin(omega*t))

# Numerical solutions using this solver
tsol,ysol = adaptive_euler(test_f,0.75,[0,10],0.1,0.1)
tsol2,ysol2 = adaptive_euler(test_f,0.75,[0,10],0.1,0.05)
tsol3,ysol3 = adaptive_euler(test_f,0.75,[0,10],0.1,0.01)
tsol4,ysol4 = adaptive_euler(test_f,0.75,[0,10],0.1,0.001)

# Numerical solution using scipy
scipy_sol = solve_ivp(test_f,[0,10],array([0.75]))
t_scipy,y_scipy =scipy_sol.t,scipy_sol.y[0]

# Analytical solution
def y_analytical(t):
    result = ((1.48976*2.71828**(0.7*cos(1.5*t)))/
             (2.71828**(0.7*t) + 1.48976*2.71828**(0.7*cos(1.5*t))))
    return result
ts = linspace(0,10,500)
ys = y_analytical(ts)

# Make plots
plt.figure(1)
plt.plot(tsol4,ysol4,'r--',label='Adaptive Euler $\epsilon$ = ?')
plt.plot(t_scipy,y_scipy,'bo-',label='solve_ivp')
plt.plot(ts,ys,'k-',label='analytical')
plt.title('Adaptive Euler vs solve_ivp')
plt.legend(); plt.grid(); plt.savefig('test-fig1.png',dpi=300)

plt.figure(2)
plt.plot(tsol,ysol,'ro-',label='$\epsilon$ = 0.1')
plt.plot(tsol2,ysol2,'bo-',label='$\epsilon$ = 0.05')
plt.plot(tsol3,ysol3,'go-',label='$\epsilon$ = 0.01')
plt.plot(ts,ys,'k-',label='analytical')
plt.title('Adaptive Euler with different tolerances')
plt.legend(); plt.grid(); plt.savefig('test-fig2.png',dpi=300)
