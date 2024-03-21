import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_filtered_angular_velocity(data, savepath):

    yaw_angular_velocity = data.imu_angular_velocity[2,:]
    plt.plot(yaw_angular_velocity, label="Unfiltered Yaw Angular Velocity")
    plt.plot(data.filtered_yaw_angular_velocity, label="Filtered Yaw Angular Velocity")
    plt.xlabel("Timestamps")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()

    fig_path = os.path.join(savepath, "Filtered yaw angular velocity.png")
    plt.savefig(fig_path, dpi=600)


def plot_map_and_trajectory(mp, robot_trajectory, robot_theta, savepath, name):
    
    occupancy_map = mp.MAP["map"]

    bmap = np.zeros_like(occupancy_map)
    bmap[occupancy_map==100] = 1
    bmap[occupancy_map==-100] = -1

    occ_path = os.path.join(savepath, f"{name}-occupancy-map.png")
    plt.figure()
    plt.title("Occupancy Map")
    plt.imshow(bmap, cmap='gray')
    plt.savefig(occ_path, dpi=600)

    traj_path = os.path.join(savepath, f"{name}-trajectory.png")
    plt.figure()
    plt.plot(robot_trajectory[0,:], robot_trajectory[1,:])
    plt.title("Robot Trajectory")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.savefig(traj_path, dpi=600)

    theta_path = os.path.join(savepath, f"{name}-theta.png")
    plt.figure()
    plt.plot(robot_theta[0,:])
    plt.title("Robot Theta")
    plt.xlabel("Timestamp")
    plt.ylabel("Angle (rad)")
    plt.savefig(theta_path, dpi=600)


def plot_texture_map(mp, savepath):

    texture_path = os.path.join(savepath, f"texture-map.png")
    plt.figure()
    plt.imshow(mp.MAP["map"])
    plt.title("Texture Map")
    plt.savefig(texture_path, dpi=600)
