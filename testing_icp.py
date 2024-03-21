from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from src.load_data import load_all_data

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.sum(AA[..., None] @ BB[:,None, :], axis=0)
    # H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # R = U @ Vt

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    
    # R = U @ Vt

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def py_icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances

ALL_ANGLES = np.linspace(-135/180*np.pi, 135/180*np.pi, 1081)
def get_lidar_coordinates(ranges):

    # Get valid ranges
    # valid_index = np.logical_and(ranges <= data.lidar_range_max, ranges >= data.lidar_range_min)
    # filtered_ranges = ranges[valid_index]

    # # Get valid angles
    # angles = ALL_ANGLES[valid_index]
    filtered_ranges = ranges
    angles = ALL_ANGLES

    x = filtered_ranges * np.cos(angles)
    y = filtered_ranges * np.sin(angles)

    coords_lidar_frame = np.stack((x, y))

    return coords_lidar_frame


def generate_scan_match_trajectory(data):

    total_steps = data.lidar_stamps.shape[0]

    # The current best estimate of the particle
    x_t = np.array([0, 0, 0])
    T_t = np.identity(3)
    T_delta = np.identity(3)
    T_deltas = [T_delta]
    T_ts = [T_t]

    pb = tqdm(range(1, total_steps))

    for tidx in pb:

        ranges = data.lidar_ranges[:, tidx]
        coords_lidar_t = get_lidar_coordinates(ranges)

        ranges = data.lidar_ranges[:, tidx-1]
        coords_lidar_t_1 = get_lidar_coordinates(ranges)

        lidar_position_wrt_body = np.array([136.73/1000, 0]).reshape((2, 1))
        
        coords_body_t = coords_lidar_t - lidar_position_wrt_body
        coords_body_t_1 = coords_lidar_t_1 - lidar_position_wrt_body

        T_delta, _ = py_icp(coords_body_t.T, coords_body_t_1.T, init_pose=T_delta, max_iterations=20)
        T_deltas.append(T_delta)
        # T_delta, _ = hn_icp(coords_body_t.T, coords_body_t_1.T, R_initial=T_delta[:2,:2], t_initial=T_delta[:2,2])
        T_t = T_t @ T_delta
        T_ts.append(T_t)
        x_t[:2] = T_t[:2,2]
        x_t[2] = np.arccos(T_t[0,0])

    robot_trajectory = list(zip(*[(x[0,2], x[1, 2]) for x in T_ts]))
    robot_theta = [np.arctan2(x[1,0], x[0,0]) for x in T_ts]

    return robot_trajectory, robot_theta, T_ts, T_deltas

if __name__=="__main__":
    data = load_all_data(20)
    robot_trajectory_sc, robot_theta_sc, T_ts, T_deltas = generate_scan_match_trajectory(data)
    plt.plot(robot_trajectory_sc[0], robot_trajectory_sc[1], label='Scan Match Dead Reckoning')
    plt.legend()
    plt.show()

    plt.plot(robot_theta_sc, label="Scan Matching Orientation")
    plt.show()
