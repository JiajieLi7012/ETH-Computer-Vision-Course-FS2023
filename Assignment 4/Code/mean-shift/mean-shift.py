import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    # Calculate the difference between x and each point in X
    diff = X - x

    # Square the differences and sum over columns to get squared Euclidean distance
    sq_diff = np.sum(diff ** 2, axis=1)

    # Take the square root to get the Euclidean distance
    dist = np.sqrt(sq_diff)

    return dist

def gaussian(dist, bandwidth):
    # Calculate the Gaussian weights
    weights = (1 / (bandwidth * np.sqrt(2*np.pi))) * np.exp(- (dist ** 2) / (2 * bandwidth ** 2))

    return weights

def update_point(weight, X):
    # Compute the weighted sum of points
    weighted_sum = np.sum(weight.reshape(-1,1) * X, axis=0)

    # Compute the sum of weights
    total_weight = np.sum(weight)

    # Calculate the new position as the weighted mean
    new_position = weighted_sum / total_weight

    return new_position

def meanshift_step(X, bandwidth=2.5):
    # Create a copy of X to store the new positions
    new_X = np.zeros_like(X)

    # Iterate over all points in X
    for i, x in enumerate(X):
        # Calculate the distance from the current point to all points (including itself)
        dist = distance(x, X)

        # Calculate the Gaussian weights for each point
        weights = gaussian(dist, bandwidth)

        # Update the position of the current point
        new_X[i] = update_point(weights, X)

    return new_X

def meanshift(X):
    for _ in range(20):
        X = meanshift_step(X,5)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# print(f"Original Image Shape: {shape}")
# print(f"The shape of X: {X.shape}")

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

# print(f"The shape of colors: {colors.shape}")

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)
# print(f"The shape of centroids: {centroids.shape}")

# Map labels that are out of bounds to the last color
# labels[labels >= len(colors)] = len(colors) - 1

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
