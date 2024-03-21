import os
import cv2
import numpy as np
import transforms3d as t3d
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.map import MAP
from utils import get_rotation_matrix

def generate_texture_map(dataset, data, dir_path, robot_trajectory, robot_theta, occupancy_map, frame_path):

    camera_R = t3d.euler.euler2mat(0,0.36,0.021)
    camera_p_body_frame = np.array([0.16766, 0, 0.38])
    K = np.array([[585.05108211, 0, 242.94140713], [0, 585.05108211, 315.83800193], [0, 0, 1]])
    R_o_r = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    lidar_position_wrt_body = np.array([0.13673, 0, 0.51435])

    occupancy_map = occupancy_map.MAP["map"]
    binary_map = np.zeros_like(occupancy_map)
    binary_map[occupancy_map == 100] = 0
    binary_map[occupancy_map == -100] = 1

    num_rgb_imgs = data.rgb_stamps.shape[0]

    resolution = 0.01
    xmin = -15
    ymin = -15
    xmax = 30
    ymax = 30

    mp = MAP(resolution, xmin, ymin, xmax, ymax)
    mp.convert_to_color()

    for img_idx in tqdm(range(num_rgb_imgs)):
        rgb_path = os.path.join(dir_path, f"RGB{dataset}/rgb{dataset}_{img_idx+1}.png")
        rgb_img = cv2.imread(rgb_path)
        rgb_img = rgb_img[...,::-1]

        rgb_timestamp = data.rgb_stamps[img_idx]
        disp_idx = np.argmin(np.abs(data.disp_stamps - rgb_timestamp)) + 1

        disp_path = os.path.join(dir_path, f"Disparity{dataset}/disparity{dataset}_{disp_idx}.png")
        disp_img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)

        # convert from disparity from uint16 to double
        disparity = disp_img.astype(np.float32)

        # get depth
        dd = (-0.00304 * disparity + 3.31)
        z = 1.03 / dd


        # calculate u and v coordinates 
        v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
        #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

        # get 3D coordinates 
        fx = 585.05108211
        fy = 585.05108211
        cx = 315.83800193
        cy = 242.94140713
        x = (u-cx) / fx * z
        y = (v-cy) / fy * z

        # calculate the location of each pixel in the RGB image
        rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
        rgbv = np.round((v * 526.37 + 16662.0)/fy)
        rgbu = rgbu.astype(np.int32)
        rgbv = rgbv.astype(np.int32)

        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)

        optical_coordinates = np.stack([x,y,z])

        real_coordinates_camera = np.linalg.inv(R_o_r) @ optical_coordinates.reshape((3,-1))
        coordinates_body = camera_R @ real_coordinates_camera + camera_p_body_frame[:, None]

        r_idx = np.argmin(np.abs(data.lidar_stamps - rgb_timestamp))
        
        body_p = robot_trajectory[:, r_idx]
        body_p = np.hstack((body_p, 0))
        
        R_p = get_rotation_matrix(robot_theta[:, r_idx])
        R_p = np.pad(R_p, (0, 1), 'constant', constant_values=(0))
        R_p[-1,-1] = 1

        coordinates_world = R_p @ coordinates_body + body_p[:, None]
        xis, yis = mp.get_pixel_coordinates(coordinates_world)

        # Get only those indexes whose z-coordinate is lower than lidar
        valid = coordinates_world[2,:] <= lidar_position_wrt_body[2]

        # Filter wrt to z
        xis = xis[valid]
        yis = yis[valid]
        rgbu = rgbu.reshape(-1)[valid]
        rgbv = rgbv.reshape(-1)[valid]

        # Now filter wrt to valid rgbu, rgbv values
        valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
        xis = xis[valid]
        yis = yis[valid]
        rgbu = rgbu[valid]
        rgbv = rgbv[valid]


        mp.MAP["map"][xis, yis, :] = rgb_img[rgbv, rgbu, :]

        if img_idx % 50==0:
            color_map = mp.MAP["map"] * binary_map[:,:,None]
            plt.imsave(frame_path + f"{img_idx}.png", color_map.astype(np.uint8))

    mp.MAP["map"] = mp.MAP["map"].astype(np.int16)

    return mp
    