import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define a list of x values
x = [0, 0.01, 0.1, 0.2, math.pi/2]
x_list = []

# Define a Maclaurin series function for estimating sin(x)
def maclaurin(x, n):
    x_list = []
    for j in x:
        sum = 0.0
        for i in range(0,n):
            exp = (2*i) + 1
            num = j**exp / math.factorial(exp)
            if i % 2 == 0:
                sum += num
            elif i % 2 != 0:
                sum += -num
        x_list.append(sum)
    return x_list

# Define a function to calculate the true relative error of the Maclaurin series
def true_rel_error(x, n):
    big_list = []
    list1 = []
    for i in range(1, n):
        list1 = maclaurin(x, i)
        for j in range(0, len(list1)):
            list1[j] = list1[j] - np.sin(x[j]) / np.sin(x[j])
        big_list.append(list1)
    return big_list

# Calculate the true relative errors for different x values and orders of n
true = true_rel_error(x, 5)
true1 = true[0]
true2 = true[1]
true3 = true[2]
true4 = true[3]

names = ['x=0', 'x=0.01', 'x=0.1', 'x=0.2', 'x=pi/2']

# Create a DataFrame to display true relative errors in a table
data = {
    'n': names,
    "1": true1,
    '3': true2,
    '5': true3,
    '7': true4
}

df = pd.DataFrame(data)

df = df.T

pd.options.display.float_format = '{:.5f}'.format

print(df)

# Generate x values for plotting
x = np.linspace(0, 2*math.pi, 100)

plt.xlim(0, 2*math.pi)
plt.ylim(-3, 3)

# Calculate Maclaurin series for different orders and the true sin(x) values
n_1 = maclaurin(x, 1)
n_3 = maclaurin(x, 2)
n_5 = maclaurin(x, 3)
n_7 = maclaurin(x, 4)
control = np.sin(x)

# Plot sin(x) and Maclaurin series approximations
plt.plot(x, control, label = 'sin(x)')
plt.plot(x, n_1, label = 'O(1)')
plt.plot(x, n_3, label = 'O(3)')
plt.plot(x, n_5, label = 'O(5)')
plt.plot(x, n_7, label = 'O(7)')

plt.grid()
plt.legend()
plt.show()