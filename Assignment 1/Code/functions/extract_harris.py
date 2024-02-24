import numpy as np
import cv2 # for the cv2.GaussianBlur function
from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    kernel_x = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
    kernel_y = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])

    Ix = signal.convolve2d(img, kernel_x, mode="same")
    Iy = signal.convolve2d(img, kernel_y, mode="same")
    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    Ixx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Iyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    M = np.array([[Ixx, Ixy], [Ixy, Iyy]])

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    detM = M[0,0] * M[1,1] - M[0,1] * M[1,0]
    traceM = M[0,0] + M[1,1]
    C = detM - k * traceM * traceM

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format

    maxC = ndimage.maximum_filter(C, size=(3, 3))
    # The first componet is the y coordinates.
    corners_y, corners_x = np.where((C == maxC) & (C > thresh))
    corners = np.column_stack((corners_x, corners_y))

    return corners, C

