import numpy as np
def thomas_algorithm(a, b, c, d):
    # Trigonal matrix solver
    # solves for Ax = d

    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x_sol = np.zeros(n)

    #Forward
    c_prime[0] = c[0]/b[0]
    d_prime[0] = d[0]/b[0]
    for i in range(1, n):
        if i < n-1:
            c_prime[i] = c[i]/ (b[i] - a[i] * c_prime[i-1])
        denom = (b[i] - a[i] * c_prime[i-1])
        d_prime[i] = (d[i] - a[i]* d_prime[i-1]) / denom

    #Backward
    x_sol[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x_sol[i]= d_prime[i] - c_prime[i] * x_sol[i+1]
    return x_sol