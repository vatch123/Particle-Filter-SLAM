import numpy as np
import scipy.spatial.transform.rotation as sciRot


# farthest point sampling
def fps(points, K):
    ''' points: N x 3
        K: number of points to sample
    '''
    fps_points = np.zeros((K, 3))
    dist = np.ones(points.shape[0]) * np.inf
    for i in range(K):
        idx = np.argmax(dist)
        fps_points[i] = points[idx]
        dist_ = ((points - fps_points[i]) ** 2).sum(-1)
        dist = np.minimum(dist, dist_)
    return fps_points

# icp
def icp(source_points, target_points, R_initial, t_initial, max_iter=10000, threshold=1e-4):
    ''' source_points: N x 3, target_points: M x 3
        R_initial: 3 x 3, t_initial: 3
        max_iter: maximum number of iterations
        threshold: threshold for convergence
    '''

    R = R_initial
    t = t_initial
    
    for _ in range(max_iter):
    # while True:
        # find correspondence
        P = (R @ source_points[..., np.newaxis]).squeeze() + t
        if P.ndim == 1:
            t = target_points.mean(0) - P
            break

        dist = P[:, np.newaxis, :] -  target_points[np.newaxis, :, :]
        dist = (dist ** 2).sum(-1)
        min_idx = np.argmin(dist, axis=1)
        Q = target_points[min_idx, :]

        M = (Q - Q.mean(0))[..., np.newaxis] @ (P - P.mean(0))[:, np.newaxis, :]
        M = M.sum(0)
        U, _, VT = np.linalg.svd(M)
        # R_tmp = U @ VT
        R_tmp = np.dot(VT.T, U.T)
        if abs(np.linalg.det(R_tmp) - 1) > 1:
            VT[2, :] = -VT[2, :]
            # R_tmp = U @ VT
            R_tmp = np.dot(VT.T, U.T)
        # print(R.shape)
        
        t_tmp = Q.mean(0) - (R_tmp @ P[..., np.newaxis]).squeeze().mean(0)

        # P = (R_tmp @ P[..., np.newaxis]).squeeze() + t_tmp
        R = R_tmp @ R
        t = R_tmp @ t + t_tmp

        # vec_tmp = sciRot.Rotation.from_matrix(R_tmp).as_rotvec()
        # if np.linalg.norm(vec_tmp) < (0.1 * np.pi / 180) and np.linalg.norm(t_tmp) < threshold:
            # break

        if np.linalg.norm(t_tmp) < threshold:
            break


    T = np.identity(3)
    T[:2, :2] = R
    T[:2, 2] = t

    return T, P
