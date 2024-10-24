###############  Part you have to modify ###############
import math

# From part 1.1:
# Function that estimates Pi using Riemann sums
def pi_riemann(n):
    deltaX = 1.0 / n
    sum = 0.0
    for i in range(0, n):
        x = i * deltaX
        fx = math.sqrt(1 - x**2)
        sum += fx * deltaX
    return 4 * sum

# From part 1.2:
# Function that calculates the Maclaurin series of arctan(x)
def arctan_maclaurin(x, n):
    sum = 0.0
    for i in range(0,n):
        exp = (2*i) + 1
        num = x**exp / exp
        if i % 2 == 0:
            sum += num
        elif i % 2 != 0:
            sum += -num
    return sum


# Function that estimates pi using 'arctan_maclaurin'
def pi_taylors(n):
    return 4 * arctan_maclaurin(1, n)


###########  Run but don't modify this part ############

test_n_values = range(1,11)
for N in test_n_values:
    # Calculate Pi using the two functions
    riemann_pi = pi_riemann(N)
    taylors_pi = pi_taylors(N)
    
    # Print out Pi's true value, and the two estimates:
    print('With n = ',N,': ')
    print(f'Actual Pi  = {math.pi:.16f}\nriemann_pi = {riemann_pi:.16f}\ntaylors_pi = {taylors_pi:.16f}')
    print('') # so we get a new line

########################################################


