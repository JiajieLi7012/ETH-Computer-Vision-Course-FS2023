import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist_target, sigma_observe):
    particles_w = np.zeros(len(particles))

    for i, particle in enumerate(particles):
        # Refine the bounding boxes
        xmin = int(max(particle[0] - bbox_width / 2, 0))
        ymin = int(max(particle[1] - bbox_height / 2, 0))
        xmax = int(min(particle[0] + bbox_width / 2, frame.shape[1] - 1))
        ymax = int(min(particle[1] + bbox_height / 2, frame.shape[0] - 1))

        # Calculate the color histogram
        hist_particle = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)
        # Compute chi2 distance
        chi2_distance = chi2_cost(hist_particle, hist_target)
        
        # Update the weight
        particles_w[i] = (1 / (np.sqrt(2 * np.pi) * sigma_observe)) * np.exp(-chi2_distance / (2 * sigma_observe**2))

    # Normalize the weights
    particles_w /= np.sum(particles_w)
    return particles_w