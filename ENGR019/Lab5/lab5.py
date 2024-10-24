import numpy as np
import numpy.polynomial.polynomial as nppoly

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import sys

######################################################################

# fits to a 2D spline using methods of Lab 2
# https://mathworld.wolfram.com/CubicSpline.html
def fit_cubic_spline(xy_values):

    n = xy_values.shape[0] - 1

    assert xy_values.shape == (n+1, 2)

    A = np.zeros((n+1, n+1))
    b = np.zeros_like(xy_values)

    idx = np.arange(n+1)
    idx1  = idx[:-1]

    A[idx, idx] = 4
    A[idx1, idx1+1] = 1
    A[idx1+1, idx1] = 1
    A[0, 0] = 2
    A[n, n] = 2

    b[0] = 3*(xy_values[1] - xy_values[0])
    b[1:-1] = 3*(xy_values[2:] - xy_values[:-2])
    b[n] = 3*(xy_values[n] - xy_values[n-1])

    D_values = np.linalg.solve(A, b)

    yi = xy_values[:-1]
    yi1 = xy_values[1:]

    Di = D_values[:-1]
    Di1 = D_values[1:]

    # all of these are shape (n, 2)
    ai = yi
    bi = Di
    ci = 3*(yi1 - yi) - 2*Di - Di1
    di = 2*(yi-yi1) + Di + Di1

    coeffs = np.stack((ai, bi, ci, di), axis=0)

    assert coeffs.shape == (4, n, 2)

    return coeffs

######################################################################
# evaluates a 2d spline at one or more u values

def eval_spline(u, coeffs):

    u_is_scalar = np.isscalar(u)

    if u_is_scalar:
        u = np.array([u])

    # degree-(d-1) splines, n of them, with k dimensions
    d, n, k = coeffs.shape

    assert k == 2
    
    # which of the n splines to use?
    floor_u = np.clip(np.floor(u).astype(int), 0, n-1)

    # where in the [0,1] interval to evaluate the spline?
    fract_u = u - floor_u

    # we will do l*k polynomial evaluations where
    # l = len(u)
    #
    # j indexes dimensions in range(k)
    # i indexes entries in u
    #
    # both have size l, k
    j, i = np.meshgrid(np.arange(k), floor_u)

    assert j.shape == (len(u), k)
    assert i.shape == j.shape
    
    cij = coeffs[:, i, j]
    assert cij.shape == (d, len(u), k)

    # evaluate the polynomial at the given u points
    rval = nppoly.polyval(fract_u.reshape(-1, 1), cij, tensor=False)

    if u_is_scalar:
        rval = rval.reshape(k)

    return rval

######################################################################
# evaluate the spline and its derivatives at one more u values

def eval_spline_and_derivs(u, coeffs_and_deriv_coeffs):

    rvals = []

    for c in coeffs_and_deriv_coeffs:
        rvals.append(eval_spline(u, c))

    return rvals

######################################################################
# numerically solves an ordinary differential equation using either
# Euler's method, the midpoint method, or the 4th-order Runge-Kutta
# method. arguments:
#
#   f: ODE function to integrate: R^m -> R^m
#   q0: initial state in R^m
#   dt: timestep (scalar)
#   num: integer number of steps to take
#   method: one of 'euler', 'midpoint', or 'rk4'
#
# returns a num-by-(m+1) array of states starting with q0

def approximate_ode(f, q0, dt, num, method):

    q = q0.copy()
    q_values = [q.copy()]

    for i in range(num):
        if method == 'euler':
            dq = f(q)
            q += dq * dt
        elif method == 'midpoint':
            k1 = f(q)
            k2 = f(q + k1 * dt / 2)
            q += k2 * dt
        elif method == 'rk4':
            k1 = f(q)
            k2 = f(q + k1 * dt / 2)
            k3 = f(q + k2 * dt / 2)
            k4 = f(q + k3 * dt)
            q += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        q_values.append(q.copy())

    return np.array(q_values)

######################################################################
# plot the roller coaster with the specified coefficients

def plot_coaster(ax, coeffs):

    (d, n, k) = coeffs.shape
    
    assert d == 4
    assert k == 2

    u = np.linspace(0, n, 16*n + 1)

    xy = eval_spline(u, coeffs)

    ax.plot(xy[:,0], xy[:,1], 'b-')

    xy_points = eval_spline(np.arange(n+1), coeffs)

    ax.plot(xy_points[:,0], xy_points[:,1], 'r.')
    ax.set_aspect('equal')

######################################################################
# evenly subdivides the total duration by steps of dt.
# returns the number of subdivisions, and an array of times to 
# sample at.

def discretize_time(total_duration, dt):

    num = int(round(total_duration/dt))

    t = np.arange(num+1)*dt

    return num, t

######################################################################

def main():

    # TODO: edit me!
    xy_points = np.array([
        [0.0, 10.0], 
        [4.0, 4.0], 
        [10.0, 0.0],
        [12.5, 5.0],
        [10.0, 9.0],
        [7.5, 5.0],
        [15.0, 0.0],
        [18.0, 5.0],
        [20.0, 10.0],
        [22.0, 5.0],
        [25.0, 0.0],
        [28.0, 5.0],
        [30.0, 10.0],
        [32.0, 5.0],
        [35.0, 0.0]
    ], dtype=float)

    # TODO: edit me!
    total_duration = 10.0

    # TODO: edit me!
    plot = 2
    
    n = len(xy_points) - 1

    # fit cubic spline in x & y
    coeffs = fit_cubic_spline(xy_points)

    dcoeffs = nppoly.polyder(coeffs)
    ddcoeffs = nppoly.polyder(dcoeffs)

    coeffs_and_deriv_coeffs = (coeffs, dcoeffs, ddcoeffs)

    ######################################################################
    # set up some helper functions for simulation

    m = 1.0
    g = 9.8

    # ODE function - evaluate dq/dt at the given q
    def f(q):

        # state contains generalized position & velocity
        u, du = q

        (x, y), (xu, yu), (xuu, yuu) = eval_spline_and_derivs(
            u, coeffs_and_deriv_coeffs)

        # dynamics from Euler-Lagrange equation
        # that we derived during lab
        ddu = (-du*du*(xu*xuu + yu*yuu) - g*yu) / (xu*xu + yu*yu)

        return np.array([du, ddu])

    # return list of [total, potential, kinetic] energy
    def energy(q):

        # state contains generalized position & velocity
        u, du = q

        (x, y), (xu, yu), (xuu, yuu) = eval_spline_and_derivs(
            u, coeffs_and_deriv_coeffs)

        # potential and kinetic energy functions
        # that we derived during lab
        potential = m * g * y

        kinetic = m * (xu*xu + yu*yu) * du*du / 2

        total = potential + kinetic

        return np.array([total, potential, kinetic])

    ######################################################################
    # display some plots!

    if plot == 0:

        # plot the roller coaster

        fig, ax = plt.subplots()

        plot_coaster(ax, coeffs)

        ax.set_title("Here's your design:")
        ax.grid('on')

        #Save plot as a png file
        plt.savefig('coaster_plot0.png')

        plt.show()

    elif plot == 1:
        
        # make an animated gif 
    
        # using Euler's method
        method = 'euler'

        # simulating 100 ticks/sec
        sim_dt = 0.01

        # record gif at 40 ms/frame <=> 25 frames/sec
        frame_msec = 40
        frame_dt = frame_msec * 0.001
        fps = 1.0 / frame_dt

        # discretize time
        num_sim, t_sim = discretize_time(total_duration, sim_dt)
        
        # do simulation for time interval
        q0 = np.zeros(2)
        q_sim = approximate_ode(f, q0, sim_dt, num_sim, method)

        # get generalized position coordinate for entire sim
        u_sim = q_sim[:, 0]
        du_sim = q_sim[:, 1]

        # what's the last one? it should be close to n to
        # hit the end of the roller coaster
        last_u_sim = u_sim[-1]

        # get our relative error
        rel_err = np.abs((last_u_sim - n) / n)

        # print out some info
        print(f'final u was {last_u_sim:.2f}, should be {n:.1f}')

        if np.any(du_sim < 0.0):
            print('the roller coaster is going the WRONG WAY!')
            print('eliminate an uphill somewhere?')
        elif rel_err < 0.05:
            print('total_duration is good!')
        elif last_u_sim < n:
            print('you should INCREASE total_duration')
        else:
            print('you should DECREASE total_duration')

        ##################################################
        # everything below here for gif output

        num_frames, t_frames = discretize_time(total_duration, frame_dt)

        u_frames = np.interp(t_frames, t_sim, u_sim)
        
        xy_frames = eval_spline(u_frames, coeffs)

        fig, ax = plt.subplots()

        plot_coaster(ax, coeffs)

        handle, = ax.plot([], [], 'ko')

        ax.set_title('Roller coaster!!!')

        def init_func():
            return handle,

        def update_func(frame):
            x, y = xy_frames[frame]
            handle.set_data([x],[y])
            return handle,

        ani = FuncAnimation(fig, update_func, frames=num_frames+1,
                            init_func=init_func, blit=True,
                            interval=100)

        filename = 'coaster.gif'

        ani.save(filename, writer=PillowWriter(fps=fps))

        print('wrote', filename)

        ##################################################
        # done gif output
        
    else:

        assert plot == 2

        # compare energy conservation for each of the different
        # integration methods.

        dt_values = [0.01, 0.005]
        methods = ['euler', 'midpoint', 'rk4']

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

        for method in methods:

            for sim_dt in dt_values:

                num_sim, t_sim = discretize_time(total_duration, sim_dt)

                q0 = np.zeros(2)
                q_sim = approximate_ode(f, q0, sim_dt, num_sim, method)

                e = []

                for q in q_sim:
                    e.append(energy(q))

                e = np.array(e)
                
                total = e[:, 0]
                delta_total = np.abs(np.diff(total)) 
                log_delta_total = np.log(delta_total)

                ax1.plot(t_sim[1:], log_delta_total, label=f'{method}, dt={sim_dt}')
                ax2.plot(t_sim, e[:,1])
                ax3.plot(t_sim, e[:,2])

        ax1.set_title('Energy vs time')

        ax1.legend(loc='upper left', fontsize=6)

        ax1.set_ylabel('log(|change|)')
        ax2.set_ylabel('potential')
        ax3.set_ylabel('kinetic')

        ax3.set_xlabel('time (s)')

        plt.savefig('coaster_plot2.png')

        plt.show()

main()
