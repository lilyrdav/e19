from scipy.integrate import solve_ivp
from numpy import zeros,exp,linspace,array,sin,cos,arange,pi
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Define c values
c_values = [0.1,0.3,0.5]

# Define domain
delta_x     = 0.1
delta_t     = 0.05
final_t     = 3.0
x_values    = arange(0,2*pi+delta_x,delta_x)
t_values    = arange(0,final_t+delta_t,delta_t)

n = len(x_values)
m = len(t_values)

phi_set = [zeros((n,m)),zeros((n,m)),zeros((n,m))]
for c,phi_values in zip(c_values,phi_set):
    # This for loop loops over three values of c.
    # Copy over your code from problem (1) in the
    # appropriate place here.
    
    # set initial condition
    phi_values[:,0] = exp(-(x_values-pi)**2)

    # time march
    for j in range(1,m):
        for i in range(1, n - 1):
            phi_values[i,j] = phi_values[i, j-1] - c * delta_t / (2 * delta_x) * (phi_values[i+1, j-1] - phi_values[i-1, j-1])

        phi_values[0, j] = phi_values[0, j-1] - c * delta_t / delta_x * (phi_values[1, j-1] - phi_values[0, j-1])  # Forward diff at x=0
        phi_values[-1, j] = phi_values[-1, j-1] - c * delta_t / delta_x * (phi_values[-1, j-1] - phi_values[-2, j-1])  # Backward diff at x=2π

# Plotting
plt.figure(2)
fig,axs = plt.subplots(nrows=1,ncols=3)

for element,data,c in zip(axs,phi_set,c_values):
    element.contour(t_values,x_values,data)
    element.set_aspect('equal')
    element.set_title(f'c = {c:.1f}')
    element.set_xlabel('t')
    element.grid()
    if c == 0.1:
        element.set_ylabel('x')

plt.savefig("fig3a.png",dpi=300)
