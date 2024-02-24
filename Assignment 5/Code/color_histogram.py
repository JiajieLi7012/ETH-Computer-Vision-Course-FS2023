import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    # Crop the frame to the bounding box
    cropped_frame = frame[ymin:ymax, xmin:xmax]

    # Initialize an empty array for the histogram
    hist = np.zeros(hist_bin * 3)

    # Calculate the histogram for each color channel
    for i in range(3):
        channel = cropped_frame[:, :, i]
        channel_hist, _ = np.histogram(channel, bins=hist_bin, range=(0, 256))
        hist[i * hist_bin:(i + 1) * hist_bin] = channel_hist

    # Normalize the histogram
    hist_normalized = hist / np.sum(hist)

    return hist_normalized