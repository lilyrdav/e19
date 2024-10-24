import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit,polyval

def f1(x):
    return x**6 + 2*x**4 + 3*x - 1

# Define x-axis to plot over:
xs = np.linspace(-1,3,100)

# Choose x-values to start interpolating
xvals = np.array([-1,-0.25,0.5])

# Calculate the function values at those x-values:
yvals = f1(xvals)

# Interpolate a polynomial through the points (x0,y0), (x1,y1), (x2,y2)
coeffs_polynomial = polyfit(xvals,yvals,2)

# Evaluate this polynomial over the entire x-axis
ys_poly = polyval(xs,coeffs_polynomial)

plt.plot(xs,f1(xs),'r-',xvals,yvals,'ro',xs,ys_poly,'b-')
plt.grid()
plt.savefig('test1.png')
plt.show()

xnew = -coeffs_polynomial[1]/(2*coeffs_polynomial[2])
xvals = np.array([xvals[1], xnew, xvals[2]])
yvals = f1(xvals)
coeffs_polynomial = polyfit(xvals,yvals,2)

# Plot
# 1) the three points
# 2) the function to be minimized
# 3) the interpolated polynomial
plt.plot(xs,f1(xs),'r-',xvals,yvals,'ro',xs,ys_poly,'b-')
plt.grid()
plt.savefig('test2.png')
plt.show()