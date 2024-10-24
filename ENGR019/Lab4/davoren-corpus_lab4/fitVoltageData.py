# short program to demonstrate fitting using 

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# input data is sampled at 20 hz
DELTA_T = 0.05

######################################################################
# compute a sum of two exponential functions 
# evaluated at a set of points t

def exp_sum(q, t):

    # unpack the parameter vector
    # into two initial amplitudes/weights and two time constants
    A1, A2, s1, s2 = q

    # sum of exponential functions
    return A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)

######################################################################
# our main function

def main():

    # load sample data from text file
    v_target = np.genfromtxt('voltage.txt')

    # make linearly spaced array of sampling times
    t = np.arange(len(v_target)) * DELTA_T

    ###################################################################
    # define some helper functions for optimizing

    # reconstruct a signal from a set of parameters
    def recons(q):
        # just calls exp_sum function above using the
        # value of t we defined in main
        return exp_sum(q, t)

    # given a parameter set, return the vector of errors
    # (i.e., target sample - reconstructed sample)
    def err_vec(q):
        return v_target - recons(q)

    # the objective function computes squared norm of the error vector
    def objective(q):
        e = err_vec(q)
        return np.dot(e, e)

    ##################################################################
    # make a set of initial parameters and get the reconstructed 
    # voltage curve

    q_init = np.array([-1.0, 1.0, -4.0, -0.1])
    v_init = recons(q_init)
    f_init = objective(q_init)

    ######################################################################
    # minimize f(q) where f: R^n -> R
    
    res = scipy.optimize.minimize(objective, q_init, method='BFGS', jac='2-point')

    n_min = res.nfev

    q_min = res.x
    v_min = recons(q_min)
    f_min = objective(q_min)

    ######################################################################
    # minimize ||f(q)|| where f: R^n -> R^m

    res = scipy.optimize.least_squares(err_vec, q_init, method='trf')

    n_lsq = res.nfev

    q_lsq = res.x
    v_lsq = recons(q_lsq)
    f_lsq = objective(q_lsq)

    ######################################################################
    # now plot our outputs

    fig, ax = plt.subplots()

    ax.plot(t, v_target, 'k.', label='target')
    ax.plot(t, v_init, 'g:', label=f'initial guess (err={f_init:.3g})')
    ax.plot(t, v_min, 'c-', label=f'minimize (err={f_min:.3g})')
    ax.plot(t, v_lsq, 'b--', label=f'least_squares (err={f_lsq:.3g})')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('voltage (V)')

    # Include information in the legend about how many times the function to be minimized
    # was evaluated when the two optimizers, scipy.optimize.minimize and scipy.optimize.least_squares, 
    # were used.
    ax.legend(title='no. of evals: minimize=%d, least_squares=%d' % (n_min, n_lsq))

    plt.show()
    
######################################################################

main()    