import math
import numpy as np
import pandas as pd

def maclaurin(n,x):
    sum = 0
    for i in range(n+1):
        sum += x**i/math.factorial(i)
    return sum

n = 1
current = maclaurin(n, 1)
prev = 0
true = np.e

n_list = []
current_list = []
prev_list = []
true_list = []
approx_list = []

while abs(current - prev) >= 0.0001:
    true_error = abs(current - true)/true
    approx_error = abs(current - prev)/current

    n_list.append(n)
    current_list.append(current)
    prev_list.append(prev)
    true_list.append(true_error)
    approx_list.append(approx_error)

    prev = current
    n += 1
    current = maclaurin(n, 1)

data = {
    'n': n_list,
    'x^n': current_list,
    'x^(n-1)': prev_list,
    'true error': true_list,
    'approx error': approx_list
}

df = pd.DataFrame(data)

df.to_csv('maclaurin.csv', index=False, float_format='%.6f', sep='\t')