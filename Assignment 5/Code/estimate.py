import numpy as np

def estimate(particles, particles_w):
    mean_state = np.average(particles, axis=0, weights=particles_w.flatten())
    return mean_state