import numpy as np
import matplotlib.pyplot as plt

# Define the true value of sin(0.2)
true = np.sin(0.2)

# Create an array of n values from 1 to 20
n_values = np.arange(1, 21)

# Initialize an empty list to store the true errors
errors = []

# Calculate the true error for Maclaurin estimates of sin(0.2) up to order n
for i in n_values:
    sign = (-1) ** i
    odd = 2 * i + 1
    approx = sign * (0.2 ** odd) / np.math.factorial(odd)
    errors.append((approx - true))

# Add a small number to each error to avoid log(0) and for visual clarity
log = [err if err > 0 else 1e-16 for err in errors]

# Plot the true errors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(n_values, errors, marker='o', linestyle='-')
ax1.set_xlabel('n')
ax1.set_ylabel('True Error')
ax1.set_title('True Error vs. n (Linear)')
ax1.grid()

ax2.plot(n_values, log, marker='o', linestyle='-')
ax2.set_xlabel('n')
ax2.set_ylabel('True Error')
ax2.set_title('True Error vs. n (Log Scale)')
ax2.set_yscale('log')
ax2.set_ylim(1e-18, 1e-14)
ax2.grid()

plt.tight_layout()
plt.show()