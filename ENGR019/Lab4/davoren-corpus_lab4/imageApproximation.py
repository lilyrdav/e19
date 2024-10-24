import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse
import scipy.optimize
from PIL import Image


# each gaussian function is defined by six parameters:
#
#       a: amplitude/weight
#      xc: x center point in pixels
#      yc: y center point in pixels
#      su: gaussian size in u direction
#      sv: gaussian size in v direction
#   theta: angle of u axis with respect to image x axis

PARAMS_PER_BASIS_FUNC = 6

######################################################################
# bounds for gaussian function parameters, for an image with w cols
# and h rows

def get_bounds(w, h):

    lo = np.array([ -1.0, 0.0, 0.0,   1.0,   1.0, -np.pi ])
    hi = np.array([  1.0,   w,   h, 0.2*w, 0.2*h,  np.pi ])

    return lo, hi

######################################################################
# create a set of random initial gaussian function paramters for an
# image with given dimensions, and given the desired count of gaussian
# functions
#
# note that each gaussian function has six parameters but instead
# of returning a 2D array of size count-by-6, we instead return
# flat array of size (count*6)

def rand_params(w, h, count):

    # create lower and upper bounds for random gaussian function
    # parameters
    lo, hi = get_bounds(w, h)

    # sample uniformly in between lower and upper bounds
    q = np.random.uniform(lo, hi, (count, PARAMS_PER_BASIS_FUNC))

    # convert to float32 for quicker reconstruction
    q = q.astype(np.float32)

    # take 2d array and flatten down to 1d array
    return q.flatten()

######################################################################
# create an image as a sum of gaussian functions given x and y
# coordinates of each pixel location

def make_image(q, xmesh, ymesh):

    # reshape from flat array to n-by-6 array of params
    q = q.reshape(-1, PARAMS_PER_BASIS_FUNC)

    # initialize output to zeros
    gimg = np.zeros_like(xmesh)

    # for each gaussian function
    for (a, xc, yc, su, sv, theta) in q:

        # get cosine and sine of rotation angle
        ct = np.cos(theta)
        st = np.sin(theta)

        # subtract out the x/y center of the gaussian function
        xs = xmesh - xc
        ys = ymesh - yc

        # decompose into u/v coordinate system
        u = xs*ct - ys*st
        v = xs*st + ys*ct
        
        # add contribution of gaussian function to output image
        # note scaled by amplitude of this individual function
        gimg += a * np.exp( -u**2 / (2*su**2)  - v**2 / (2*sv**2) )

    # return the summed image
    return gimg


######################################################################
# utility function to plot outlines of Gaussian ellipses on a figure

def plot_gaussian_ellipses(ax, q):

    q = q.reshape(-1, PARAMS_PER_BASIS_FUNC)

    for (a, xc, yc, su, sv, theta) in q:

        red = np.clip(0.5*a + 0.5, 0, 1)
        blue = 1-red

        ei = Ellipse(xy=(xc,yc),
                     width=3*su, height=3*sv,
                     angle=-theta*180/np.pi)
        
        ei.set_facecolor('none')
        ei.set_edgecolor([red, 0, blue])
        ax.add_artist(ei)

######################################################################
# nice imshow that uses grayscale colormap, makes pixels chunky
# squares, and turns axis ticks off.

def imshow_nice(ax, img, cmap='gray'):
    ax.imshow(img, cmap=cmap, interpolation='nearest')
    ax.set_axis_off()

######################################################################
# save parameters to a text file

def save_params(filename, q):
    np.savetxt(filename, q, fmt='%f')
    print('wrote', filename)

######################################################################
# save image data to file

def save_image(filename, img, scale, bias):

    img = img * scale + bias

    mpimg.imsave(filename, img, vmin=0.0, vmax=1.0, cmap='gray')

    print('wrote', filename)
    
######################################################################

def main():

    input_filename = 'eye.png'

    gaussian_count = 16

    # read the input image and convert it to floating point
    target = mpimg.imread(input_filename).astype(np.float32)

    if len(target.shape) == 3:
        raise RuntimeError('only grayscale images are supported right now')

    h, w = target.shape

    print(f'image {input_filename} has width {w} and height {h}')

    # rescale target to have mean zero and lie in [-1, 1] range
    #
    # this is not technically necessary but helps the optimization
    # converge a bit faster

    bias = target.mean()

    target -= bias # set mean to zero

    scale = np.abs(target).max()

    target /= scale # set to be within [-1, 1] interval

    xrng = np.arange(w, dtype=np.float32)
    yrng = np.arange(h, dtype=np.float32)

    xmesh, ymesh = np.meshgrid(xrng, yrng)

    ##################################################################
    # inner functions for optimization

    # reconstruct an image
    def recons(q):
        # calls make_image function above using
        # the xmesh, ymesh variables defined in main
        return make_image(q, xmesh, ymesh)

    # compute an "error image" showing good fit = close to zero, poor
    # fit = high magnitude pixels
    def err_img(q):
        return target - recons(q)

    # reshapes the 2D h-by-w image into a flat array of length (h*w)
    def err_vec(q):
        return err_img(q).flatten()

    # the objective function computes squared norm of the error vector
    def objective(q):
        e = err_vec(q)
        return np.dot(e, e)
    
    ##################################################################
    # Note: you can replace all of the code below this line with
    # your solution for task 2.

    results = {}  # Dictionary to store results for plotting

    for k in [8, 16, 32]:
        for method in ['Powell', 'L-BFGS-B', 'trf']:
            best_obj = float('inf')
            best_params = None
            obj_values = []

            for trial in range(10):
                q = rand_params(w, h, k)  # Random initial parameters

                if method == 'trf':
                    res = scipy.optimize.least_squares(err_vec, q, method=method, max_nfev=10000)
                else:
                    res = scipy.optimize.minimize(objective, q, method=method, jac='2-point' if method == 'L-BFGS-B' else None, options=dict(maxfev=10000) if method == 'Powell' else dict(maxfun=10000))

                current_obj = objective(res.x)
                obj_values.append(current_obj)

                if current_obj < best_obj:
                    best_obj = current_obj
                    best_params = res.x

            # Save the best parameters and image
            save_params(f'params_{k}_{method.lower()}.txt', best_params)
            save_image(f'best_image_{k}_{method.lower()}.png', recons(best_params), scale, bias)

            # Calculate and store statistics
            mean_obj = np.mean(obj_values)
            std_obj = np.std(obj_values)
            results[(k, method)] = (mean_obj, std_obj)

    # Plotting the results
    kfuncs = ("k = 8", "k = 16", "k = 32")
    method_means = {
        'Powell': ([results[(8, 'Powell')][0], results[(16, 'Powell')][0], results[(32, 'Powell')][0]]),
        'L-BFGS-B': ([results[(8, 'L-BFGS-B')][0], results[(16, 'L-BFGS-B')][0], results[(32, 'L-BFGS-B')][0]]),
        'trf': ([results[(8, 'trf')][0], results[(16, 'trf')][0], results[(32, 'trf')][0]])
    }

    x = np.arange(len(kfuncs))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in method_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Mean Objective Value')
    ax.set_title('Mean Objective Value by k and Method')
    ax.set_xticks(x + width, kfuncs)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 23)

    plt.show()

    #Save bar chart
    fig.savefig('bar_chart.png', dpi=300)

    # Define the image file names
    image_files = [f'best_image_{k}_{method}.png' for k in [8, 16, 32] for method in ['powell', 'l-bfgs-b', 'trf']]

    # Initialize a figure for the 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Iterate over the image files and add them to the grid
    for i, ax in enumerate(axes.flatten()):
        try:
            # Load the image
            img = Image.open(image_files[i])

            # Display the image
            ax.imshow(img)

            # Set the title for each subplot
            k_value = image_files[i].split('_')[2]
            method = image_files[i].split('_')[3].split('.')[0]
            ax.set_title(f'k={k_value}, method={method}')

            # Hide the axis
            ax.axis('off')
        except FileNotFoundError:
            # If the file is not found, just pass
            pass

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Save the grid of images
    fig.savefig('best_images_grid.png', dpi=300)

if __name__ == '__main__':
    main()