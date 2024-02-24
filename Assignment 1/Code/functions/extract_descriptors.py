import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    '''
    Inputs:
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    Returns:
    - keypoints:    (q', 2) numpy array of keypoint locations [x, y] that are far enough from edges
    '''
    # TODO: Filter out keypoints that are too close to the edges

    h, w = img.shape
    offset = int(np.floor(patch_size / 2.0))
    
    # Keep only keypoints that are far enough from (half of the patch size, 4 pixels in our case) the edges/boundries of the whole image
    valid_x = (keypoints[:, 0] >= offset) & (keypoints[:, 0] < w - offset)
    valid_y = (keypoints[:, 1] >= offset) & (keypoints[:, 1] < h - offset)
    
    valid_idx = np.where(valid_x & valid_y)[0] # np.where always returns a tuple, even with size 1
    valid_keypoints = keypoints[valid_idx]
    
    return valid_keypoints

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

