import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    q1, feature_dim = desc1.shape
    q2, _ = desc2.shape
    
    distances = np.zeros((q1, q2))
    
    for i in range(q1):
        for j in range(q2):
            # Difference between the two descriptors
            diff = desc1[i] - desc2[j]
            
            # Squaring the differences and summing them up
            squared_diff = np.sum(diff * diff)
            
            distances[i, j] = squared_diff
    
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        matches = np.argmin(distances, axis=1)  # size: (q1,)
        return np.column_stack((np.arange(matches.shape[0]), matches))
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        min_idx1 = np.argmin(distances, axis=1) # from image 1 to image 2, size: (q1,)
        min_idx2 = np.argmin(distances, axis=0) # from image 2 to image 1, size: (q2,)

        mutual_matches = []

        # Identify the matching pairs that are valid from both directions
        for desc1_idx, matched_desc2_idx in enumerate(min_idx1):
            if min_idx2[matched_desc2_idx] == desc1_idx:
                mutual_matches.append([desc1_idx, matched_desc2_idx])

        return np.array(mutual_matches)

        
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=1)[:,2] to find the second smallest value over a row
        sorted_distances = np.partition(distances, 2, axis=1)[:, :2] # size: (q1, 2)
        ratio = sorted_distances[:,0] / sorted_distances[:,1] # size: (q1,)
        valid_idx = np.where(ratio < ratio_thresh)[0] # np.where always returns a tuple, even with size 1
        return np.column_stack((valid_idx, np.argmin(distances[valid_idx,:],axis=1)))

    else:
        raise ValueError("Unsupported matching strategy!")

