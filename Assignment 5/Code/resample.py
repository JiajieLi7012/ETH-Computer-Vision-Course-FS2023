import numpy as np

def resample(particles, particles_w):
    # Normalize the weights (just in case)
    particles_w = particles_w / np.sum(particles_w)

    # Resample the particles based on their weights
    indices = np.random.choice(range(len(particles)), size=len(particles), p=particles_w)
    resampled_particles = particles[indices]
    resampled_weights = particles_w[indices]

    # Normalize the weights of the resampled particles
    resampled_weights /= np.sum(resampled_weights)

    return resampled_particles, resampled_weights
