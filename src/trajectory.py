import numpy as np
from tqdm import tqdm

from src.lidar import get_lidar_coordinates
from src.map import MAP
from utils import get_rotation_matrix
from src.load_data import Data


def generate_closest_velocities(data: Data):
    time_diff = np.abs(data.encoder_stamps[:, None] - data.lidar_stamps[None, :])
    v_index = np.argmin(time_diff, axis=0)

    time_diff = np.abs(data.imu_stamps[:, None] - data.lidar_stamps[None, :])
    w_index = np.argmin(time_diff, axis=0)

    return v_index, w_index


def get_next_pos_ort(particles, v, w, tau, noise=False):

    if noise:
        N = particles.shape[1]
        v = v + np.random.normal(0, 0.02, size=(1,N))
        w = w + np.random.normal(0, 0.002, size=(1,N))

    theta = particles[2,:]
    net_v = np.zeros_like(particles, dtype=np.float32)
    net_v[0,:] = v * np.cos(theta)
    net_v[1,:] = v * np.sin(theta)
    net_v[2,:] = w
    particles_t = particles + tau * net_v
    return particles_t



def generate_trajectory(data: Data, v_index, w_index, noise=False, N=1):

    total_steps = data.lidar_stamps.shape[0]
    lidar_position_wrt_body = np.array([136.73/1000, 0]).reshape((2, 1))

    robot_trajectory = np.zeros((2, total_steps))
    robot_theta = np.zeros((1, total_steps))
    x = np.array([0, 0, 0])

    # The current best estimate of the particle
    x_t = np.array([0, 0, 0])

    if N == 1:
        x = x.reshape((3, 1))
        x_t = x_t.reshape((3,1))
    else:
        # Maintain n particles
        particles = np.stack([x] * N, axis=-1)
        wt = np.stack([1/N] * N, axis=-1)

    resolution = 0.01
    xmin = -15
    ymin = -15
    xmax = 30
    ymax = 30

    mp = MAP(resolution, xmin, ymin, xmax, ymax)

    pb = tqdm(range(1, total_steps))

    for tidx in pb:

        ranges = data.lidar_ranges[:, tidx]
        coords_lidar = get_lidar_coordinates(ranges, data.lidar_range_min, data.lidar_range_max)

        
        coords_body = coords_lidar - lidar_position_wrt_body

        # Get pose robot pose at current time
        tau = data.lidar_stamps[tidx] - data.lidar_stamps[tidx - 1]
        v = data.linear_velocity[v_index[tidx-1]]
        w = data.filtered_yaw_angular_velocity[w_index[tidx-1]]

        if N != 1:
            # Prediction step
            particles_t = get_next_pos_ort(particles, v, w, tau, noise)
            
            # Update step
            correlations = mp.get_correlation_with_map(particles_t, coords_body)
            wt = wt * np.max(correlations, axis=1)
            wt = wt / np.sum(wt)
            # pb.set_description(f"{wt}")
            
            # Resample
            Neff = 1 / np.sum(np.square(wt))
            if Neff < N/2:
                indexes = np.random.choice(list(range(N)), N, replace=True, p=wt.tolist())
                particles = particles_t[:, indexes]
            else:
                particles = particles_t

            # Choose the best particle
            idx = np.argmax(wt)
            x_t = particles_t[:, idx].reshape((3,1))
        else:
            x_t = get_next_pos_ort(x, v, w, tau)
            x = x_t

        robot_trajectory[:, tidx][:, None] = x_t[0:2, :]
        robot_theta[:, tidx] = x_t[2, :]

        R = get_rotation_matrix(x_t[2, :])
        p = x_t[0:2,:]

        # Get all lidar coordinates in world frame
        coords_world = R @ coords_body + p

        # Get the occupancy map
        if tidx % 1 == 0:
            mp.update_occ_map(coords_world, p + lidar_position_wrt_body)

    return mp, robot_trajectory, robot_theta