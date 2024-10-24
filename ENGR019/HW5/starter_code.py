# Generate data
import numpy as np
from numpy.random import rand
from newtonPoly import coeffts, evalPoly
from polyFit import polyFit, evalPolyFit
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data2.csv', delimiter='\t', header=None)
x1 = data[0]
print(x1)
y1 = data[1]
print(y1)


# Create x-axis to plot
xs = np.linspace(-5,5,100)

# Newton Polynomial (interpolation)
a = coeffts(x1,y1)

# Evaluate Newton's polynomial at values 'xs'
y_interp = evalPoly(a,x1,xs)

# Curve fit with order m = 1
b = polyFit(x1,y1,7)

# Evaluate curve-fit at values 'xs'
y_curvefit = evalPolyFit(xs,y1,b)

# Make plot
plt.plot(x1,y1,'o',label='data')
plt.plot(xs,y_interp,'-',label='Interpolation')
plt.plot(xs,y_curvefit,'--',label='Curve Fit')
plt.legend()
plt.show()
