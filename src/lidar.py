import numpy as np


ALL_ANGLES = np.linspace(-135/180*np.pi, 135/180*np.pi, 1081)

def get_lidar_coordinates(ranges, min_range, max_range):

    # Get valid ranges
    valid_index = np.logical_and(ranges <= max_range, ranges >= min_range)
    filtered_ranges = ranges[valid_index]

    # Get valid angles
    angles = ALL_ANGLES[valid_index]

    x = filtered_ranges * np.cos(angles)
    y = filtered_ranges * np.sin(angles)

    coords_lidar_frame = np.stack((x, y))

    return coords_lidar_frame
