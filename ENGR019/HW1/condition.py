import numpy as np
import matplotlib.pyplot as plt

def condition(x):
    return abs(x*np.tan(x))

#extend x from 0 to 15
x = np.arange(0, 15, 0.01)

plt.plot(x, condition(x))
plt.ylim(0, 30)
plt.xlabel('x')
plt.ylabel('Condition Number')
plt.title('Condition number of f(x)=cos(x)')
plt.show()