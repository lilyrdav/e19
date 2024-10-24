from datetime import datetime
import numpy as np
import numpy.polynomial.polynomial as nppoly
from PIL import Image
import sys
import matplotlib.pyplot as plt

# let's define some constants to use in our program

IMAGE_WIDTH_HEIGHT = (128, 128)

CAMERA_POS = [0, -3, 3]
CAMERA_TGT = [0, 0, 0]
CAMERA_UP = [0, 1, 0]

LIGHT_DIR = [-1, 1, 2]

CAMERA_F = 0.6

SPHERE_RAD = 2.0

TORUS_R1 = 0.6
TORUS_R2 = 1.4

MARCH_TOL = 1e-3
MARCH_MAX_ITER = 100
MARCH_MAX_DIST = 10.0

DK_TOL = 1e-9
IM_TOL = 1e-9

######################################################################

def vec3(x, y, z):
    '''Construct a length-3 array from 3 scalars.'''
    return np.array([x,y,z], dtype=float)

def length(v):
    '''Thin wrapper on numpy's norm function.'''
    return np.linalg.norm(v)

def normalize(v):
    '''Return the vector v, scaled to a unit vector.'''
    return v / length(v)

def vec2color(v):
    '''
    Convert a 3D vector representing a color in a scene to an RGB color suitable for display or storage.

    This function takes a 3D vector `v` where each component of the vector represents a color intensity
    in a floating-point range [0.0, 1.0]. It clips each component to ensure it falls within this range,
    scales it to the range [0, 255], and then converts to an 8-bit unsigned integer.

    Parameters:
    v (np.array): A numpy array of three floating-point numbers representing the color intensities of Red, Green,
                  and Blue channels respectively.

    Returns:
    np.array: A numpy array of three 8-bit unsigned integers representing the color in RGB format, suitable
              for display or storage in standard image formats.
    '''
    return (np.clip(v, 0.0, 1.0) * 255).astype(np.uint8)


######################################################################

def guess_roots(coeffs):
    '''Given polynomial coefficients in increasing order of exponent, come
    up with some guesses for polynomial roots.

    Generate initial guesses for the roots of a polynomial with given coefficients.

    The algorithm generates guesses that are distributed around a circle in the
    complex plane. The circle's radius is chosen based on the magnitude of the
    coefficients, with the idea that the size of the coefficients might give
    a rough indication of the size of the roots.

    Parameters:
    coeffs (np.array): A 1D numpy array of polynomial coefficients in increasing
                       order of exponent.

    Returns:
    np.array: A numpy array of complex numbers, each an initial guess at one of
              the polynomial's roots. There will be one fewer roots than the
              number of coefficients, since a polynomial of degree n has n roots.'''

    ncoeff = len(coeffs)
    nroots = ncoeff - 1

    cmax = np.max(np.abs(coeffs))
    cmin = np.maximum(cmax * 0.1, np.min(np.abs(coeffs)))

    mags = np.linspace(cmin, cmax, nroots)
    theta = 2*np.pi / nroots

    roots = np.empty(nroots, dtype=np.complex128)

    for i in range(nroots):
        c = np.cos(theta*i)
        s = np.sin(theta*i)
        roots[i] = (c + 1j*s)*mags[i]

    return roots

######################################################################

def smallest_positive_real_root(coeffs, dk_tol, im_tol):
    '''Given polynomial coefficients in increasing order of exponent,
    return the smallest real (or nearly real) root which is greater
    than zero, or return None if there is no such root.
    '''

    roots = durand_kerner(coeffs, dk_tol)

    best_root = None

    '''
    Identify the smallest positive real or nearly real root of a polynomial.

    This function calculates the roots of a polynomial using the Durand-Kerner method
    and identifies the smallest root that is greater than zero and considered to be 
    real or approximately real based on a given tolerance level.

    Parameters:
    - coeffs (list or np.array): Coefficients of the polynomial in increasing order 
                                 of their degree, with the first element as the 
                                 constant term.
    - dk_tol (float): Tolerance level for the Durand-Kerner method, which determines 
                      when the iterative approximation of the roots should stop.
    - im_tol (float): Tolerance level for the imaginary part of the roots to be 
                      considered negligible, relative to the root's real part.

    Returns:
    - float or None: The smallest positive real (or nearly real) root of the polynomial. 
                     Returns None if there is no root that satisfies the conditions.'''
    
    for root in roots:
        re, im = np.real(root), np.imag(root)
        if np.abs(im) < im_tol*np.abs(re) and re > 0:
            if best_root is None:
                best_root = re
            else:
                best_root = np.minimum(best_root, re)

    return best_root

######################################################################

def test_roots():

    '''Test the durand_kerner() function. This will raise an
    AssertionError if rootfinding does not work.'''
    
    #coeffs = np.array([1.0, -3.0, 3.0, 5.0])
    coeffs = np.array([5.0, 3.0, -3.0, 1.0])
    
    roots_np = np.sort( nppoly.polyroots(coeffs) )
    
    roots_dk = np.sort(durand_kerner(coeffs, DK_TOL) )

    delta = roots_np - roots_dk

    relerr = np.abs(delta) / np.abs(roots_np)

    print('np roots:', roots_np)
    print('dk roots:', roots_dk)
    print('relerr:', relerr)

    if np.max(relerr) < DK_TOL:
        print('rootfinding test passed!')
    else:
        print('rootfinding test FAILED!')

    print
        
    assert np.max(relerr) < 10.0 * DK_TOL

######################################################################

def sphere_coeffs(ro, rd):

    '''Returns the coefficients a, b, c of the quadratic polynomial
    resulting from intersecting a ray with origin ro and direction 
    rd with a sphere at the origin with radius SPHERE_RAD.'''


    a = np.dot(rd, rd)
    b = 2*np.dot(ro, rd)
    c = np.dot(ro, ro) - SPHERE_RAD**2

    coeffs = np.array([c, b, a])

    return coeffs

def sphere_normal(pos):
    '''Returns the surface normal vector to a sphere centered
    at the origin, sampled at the given input position.'''
    return normalize(pos)

def sphere_distance(pos):
    '''Return the signed distance from the point at position pos to the
    sphere centered at the origin with radius rad. The returned
    distance will be positive is pos is outside the sphere, negative
    if pos is inside, or zero if pos is on the boundary.'''
    return length(pos) - SPHERE_RAD

def torus_coeffs(ro, rd):

    '''Returns the coefficients of the quartic polynomial
    resulting from intersecting a ray with origin ro and direction 
    rd with a torus at the origin with minor and major radii 
    of TORUS_R1 and TORUS_R2, respectively.'''

    r1_sqr = TORUS_R1**2
    r2_sqr = TORUS_R2**2

    alpha = np.dot(rd,rd)
    beta = 2*np.dot(ro,rd)
    gamma = np.dot(ro, ro) - r1_sqr - r2_sqr

    pz = ro[2]
    dz = rd[2]

    coeffs = np.array([
        gamma**2 + 4*r2_sqr*(pz*pz - r1_sqr),
        2*beta*gamma + 8*r2_sqr*pz*dz,
        beta**2 + 2*alpha*gamma + 4*r2_sqr*dz*dz,
        2*alpha*beta,
        alpha**2,
    ])

    return coeffs

def torus_normal(pos):
    '''Returns the surface normal vector to a torus centered
    at the origin with radii specified by constants above, 
    sampled at the given input position.'''

    x, y, z = pos

    l = np.sqrt(x**2 + y**2)

    rdiff = l - TORUS_R2

    return normalize(vec3(x*rdiff/l, y*rdiff/l, z))



######################################################################

def durand_kerner(coeffs, tol):
    '''Given polynomial coefficients in increasing order of exponent,
    return a list of all (possibly complex) roots of the
    polynomial.'''

    # Get the number of roots (degree of polynomial)
    n = len(coeffs) - 1

    # Start with an initial guess
    roots = guess_roots(coeffs)

    # We will iterate for a max of 1000 iterations or until convergence
    for _ in range(1000):

        # Store the previous roots to check for convergence later
        original_roots = np.copy(roots)

        for i in range(n):
            product = np.prod([roots[i] - roots[j] for j in range(n) if i != j])
            roots[i] -= np.polyval(coeffs[::-1], roots[i]) / product

        # Check for convergence
        if np.all(np.abs(roots - original_roots) < tol):
            break

    return roots

def torus_distance(pos):
    '''Return the signed distance from the point at position pos to the torus 
    centered at the origin with radii r1 (tube radius) and r2 (central radius).'''
    
    # Project pos onto the xy plane by ignoring the z component
    projection = np.array([pos[0], pos[1], 0.0])
    
    # Find the closest point on the circle of radius r2 to the projection
    projection_length = length(projection)
    if projection_length > 1e-5:
        closest_point = projection * TORUS_R2 / projection_length
    else:
        # If the projected length is too small, arbitrarily choose a point on the circle
        closest_point = vec3(TORUS_R2, 0.0, 0.0)
    
    # Find the vector from this closest point to the original position
    vector = pos - closest_point
    
    # The distance to the surface of the torus is then the length of this vector minus r1
    return length(vector) - TORUS_R1


######################################################################

def render_scene(shape, method, 
                 coeffs_or_dist_func, 
                 normal_func, 
                 base_color):

    '''Renders the scene with a sphere or torus primitive,
    using either raytracing or raymarching. 

    You should not need to change anything in this function to accomplish
    the objectives of Lab 3.'''

    img_w, img_h = IMAGE_WIDTH_HEIGHT

    pos = vec3(*CAMERA_POS)
    tgt = vec3(*CAMERA_TGT)
    up = vec3(*CAMERA_UP)

    l = normalize(vec3(*LIGHT_DIR))

    view_z = normalize(tgt - pos)
    view_x = normalize(np.cross(view_z, up))
    view_y = normalize(np.cross(view_x, view_z))

    R = np.array([view_x, view_y, view_z]).transpose()

    ro = pos

    image_data = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    px_scl = 1.0 / (0.5 * img_h)

    start_time = datetime.now()

    for row in range(img_h):

        y = -(row - 0.5*img_h + 0.5) * px_scl

        for col in range(img_w):

            x = (col - 0.5*img_w + 0.5) * px_scl

            rd = normalize(vec3(x*CAMERA_F, y*CAMERA_F, 1))
            rd = np.dot(R, rd)

            hit_pos = None
            hit_normal = None

            if method == 'trace':
                
                coeffs = coeffs_or_dist_func(ro, rd)

                t = smallest_positive_real_root(coeffs, DK_TOL, IM_TOL)

                if t is not None:

                    hit_pos = ro + rd * t

                    hit_normal = normal_func(hit_pos)
                
            elif method == 'march':

                t = MARCH_TOL
                hit = False

                for i in range(MARCH_MAX_ITER):
                    pos = ro + t * rd
                    d = coeffs_or_dist_func(pos)
                    if np.abs(d) < MARCH_TOL:
                        hit_pos = pos
                        hit_normal = vec3(0, 0, 0)
                        h = 1e-4
                        for i in range(3):
                            delta = vec3(0, 0, 0)
                            delta[i] = h
                            d1 = coeffs_or_dist_func(pos + delta)
                            d0 = coeffs_or_dist_func(pos - delta)
                            hit_normal[i] = (d1 - d0) / (2.0*h)
                        hit_normal = normalize(hit_normal)
                        break

                    t += d
                    if t >= MARCH_MAX_DIST:
                        break


            else:

                raise RuntimeError('invalid method')

            if hit_pos is not None:

                # shade the image
                NdotL = np.maximum(np.dot(hit_normal, l), 0.0)

                ka = 0.2
                kd = 0.8

                H = normalize(l - rd)
                NdotH = np.maximum(np.dot(hit_normal, H), 0.0)

                diffamb = (ka + kd*NdotL) * base_color

                spec = vec3(1,1,1) * np.power(NdotH, 120.0)

                image_data[row, col] = vec2color(diffamb + spec)

    elapsed = (datetime.now() - start_time).total_seconds()

    image = Image.fromarray(image_data, mode='RGB')

    filename = shape + '_' + method + '.png'

    image.save(filename)

    print('  rendered', filename, 'in', elapsed, 'seconds')
    
    return elapsed

######################################################################
test_roots()

# The following code creates four 'cases'.
cases = [
    ('sphere', 'trace', sphere_coeffs, sphere_normal, vec3(1, 0.5, 0)),
    ('sphere', 'march', sphere_distance, None, vec3(0.5, 1, 0)),
    ('torus', 'trace', torus_coeffs, torus_normal, vec3(0, 0.5, 1)),
    ('torus', 'march', torus_distance, None, vec3(0.5, 0, 1))
]

print('rendering images...')

for shape, method, cdfunc, nfunc, color in cases:
    render_scene(shape, method, cdfunc, nfunc, color)

######################################################################

def time_size(n, shape, method, cdfunc, nfunc, color):
    global IMAGE_WIDTH_HEIGHT
    IMAGE_WIDTH_HEIGHT = (2**n, 2**n)
    time_elapsed = render_scene(shape, method, cdfunc, nfunc, color)
    return time_elapsed

# Collect data
times = []
ns = range(4, 9)
for n in ns:
    elapsed_time = time_size(n, 'sphere', 'trace', sphere_coeffs, sphere_normal, vec3(1, 0.5, 0))
    times.append(elapsed_time)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(ns, times, 'o-')
plt.xlabel('Image Size (n)')
plt.ylabel('Rendering Time (seconds)')
plt.title('Rendering Time vs Image Size')
plt.grid(True)
plt.show()