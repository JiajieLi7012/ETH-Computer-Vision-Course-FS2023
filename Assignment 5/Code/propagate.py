import numpy as np

def propagate(particles, frame_height, frame_width, params):
    # Constant velocity motion model
    if params["model"] == 1:  
        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        for i in range(len(particles)):
            noise = np.random.normal(0, [params["sigma_position"], params["sigma_position"], 
                                         params["sigma_velocity"], params["sigma_velocity"]], 4)
            particles[i] = np.dot(A, particles[i]) + noise
            # Keep particles within frame boundaries
            particles[i][0] = np.clip(particles[i][0], 0, frame_width - 1)
            particles[i][1] = np.clip(particles[i][1], 0, frame_height - 1)
    else:  # No-motion model
        for i in range(len(particles)):
            noise = np.random.normal(0, params["sigma_position"], 2)
            particles[i][:2] += noise
            # Keep particles within frame boundaries
            particles[i][0] = np.clip(particles[i][0], 0, frame_width - 1)
            particles[i][1] = np.clip(particles[i][1], 0, frame_height - 1)

    return particles