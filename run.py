import os
import argparse

from src.load_data import load_all_data
from src.plots import plot_filtered_angular_velocity, plot_map_and_trajectory, plot_texture_map
from src.trajectory import generate_closest_velocities, generate_trajectory
from src.texture_map import generate_texture_map


def run_simulations(dataset):

    print(f"Loading Data for Dataset {dataset}...")
    data = load_all_data(dataset)
    savepath = os.path.join(os.getcwd(), f"results/dataset{dataset}")
    os.makedirs(savepath, exist_ok=True)

    plot_filtered_angular_velocity(data, savepath)

    v_index, w_index = generate_closest_velocities(data)

    # print("Generating Trajectory and Map for Dead Reckoning")
    # # Generate deadreckoning trajectory with N=1 and no noise
    # occupancy_map, robot_trajectory, robot_theta = generate_trajectory(
    #     data=data,
    #     v_index=v_index,
    #     w_index=w_index,
    #     noise=False,
    #     N=1
    # )

    # # Plot the map and trajectory
    # plot_map_and_trajectory(occupancy_map, robot_trajectory, robot_theta, savepath, "deadreckoning")

    N = 100
    print(f"Generating trajectory and map using SLAM with {N} particles")
    occupancy_map, robot_trajectory, robot_theta = generate_trajectory(
        data=data,
        v_index=v_index,
        w_index=w_index,
        noise=True,
        N=N
    )

    # Plot the map and trajectory
    plot_map_and_trajectory(occupancy_map, robot_trajectory, robot_theta, savepath, "slam")

    print("Generating Texture Map...")
    image_dir = os.path.join(os.getcwd(), "data/dataRGBD/")
    frame_path = os.path.join(savepath, "texture/")
    os.makedirs(frame_path, exist_ok=True)

    texture_map = generate_texture_map(
        dataset,
        data,
        image_dir,
        robot_trajectory,
        robot_theta,
        occupancy_map,
        frame_path
    )
    plot_texture_map(texture_map, savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SLAM')
    parser.add_argument('--dataset', type=int, default=20, help='The dataset on which to run tracking')
    args = parser.parse_args()
    run_simulations(args.dataset)
