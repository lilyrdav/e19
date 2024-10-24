# Generate a list of the first n prime numbers and a list of all prime numbers less than m

# Define a function to check if a number is prime
def is_prime(n):
    if n < 2:
        return False

    for i in range(2, n):
        if n % i == 0:
            return False
        return True

# Define a function to generate a list of the first n prime numbers
def first_n_primes(n):
    nprimes = []
    i = 2
    while len(nprimes) < n:
        if is_prime(i):
            nprimes.append(i)
        i += 1
    return nprimes

# Define a function to generate a list of all prime numbers less than m
def primes_upto_than(m):
    mprimes = []
    for i in range(2, m):
        if is_prime(i):
            mprimes.append(i)
    return mprimes

print(is_prime(10))
print(first_n_primes(10))
print(primes_upto_than(10))