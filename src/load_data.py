import os
import numpy as np
from scipy import signal

class Data:
    def __init__(
        self,
        encoder_counts,
        encoder_stamps,
        lidar_angle_min,
        lidar_angle_max,
        lidar_angle_increment,
        lidar_range_min,
        lidar_range_max,
        lidar_ranges,
        lidar_stamps,
        imu_angular_velocity,
        imu_linear_acceleration,
        imu_stamps,
        disp_stamps,
        rgb_stamps

        ) -> None:

        self.encoder_counts = encoder_counts
        self.encoder_stamps = encoder_stamps

        self.lidar_angle_min = lidar_angle_min
        self.lidar_angle_max = lidar_angle_max
        self.lidar_angle_increment = lidar_angle_increment
        self.lidar_range_min = lidar_range_min
        self.lidar_range_max = lidar_range_max
        self.lidar_ranges = lidar_ranges
        self.lidar_stamps = lidar_stamps

        self.imu_angular_velocity = imu_angular_velocity
        self.imu_linear_acceleration = imu_linear_acceleration
        self.imu_stamps = imu_stamps

        self.disp_stamps = disp_stamps
        self.rgb_stamps = rgb_stamps

        self.filtered_yaw_angular_velocity = self.filter_imu_data()
        self.linear_velocity = self.calculate_linear_velocity()

    def filter_imu_data(self):
        yaw_angular_velocity = self.imu_angular_velocity[2,:]
        sos = signal.butter(4, 0.02, output='sos')
        filtered_yaw_angular_velocity = signal.sosfilt(sos, yaw_angular_velocity)
        return filtered_yaw_angular_velocity
    
    def calculate_linear_velocity(self):
        distance_per_tick = 0.0022      # (pi*d) / 360
        left_distance = (self.encoder_counts[1,:] + self.encoder_counts[3,:]) / 2 * distance_per_tick
        right_distance = (self.encoder_counts[0,:] + self.encoder_counts[2,:]) / 2 * distance_per_tick

        distance_travelled = (left_distance + right_distance) / 2
        encoder_time_differnce = self.encoder_stamps[1:] - self.encoder_stamps[:-1]

        linear_velocity = np.zeros_like(distance_travelled)
        linear_velocity[1:] = distance_travelled[1:] / encoder_time_differnce
        return linear_velocity


def load_all_data(dataset):
  
  dir_path = os.path.join(os.getcwd(), "data")

  enc_path = os.path.join(dir_path, "Encoders%d.npz"%dataset)
  with np.load(enc_path) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  lidar_path = os.path.join(dir_path, "Hokuyo%d.npz"%dataset)
  with np.load(lidar_path) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
  imu_path = os.path.join(dir_path, "Imu%d.npz"%dataset)
  with np.load(imu_path) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  kinteic_path = os.path.join(dir_path, "Kinect%d.npz"%dataset)
  with np.load(kinteic_path) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    return Data(
        encoder_counts,
        encoder_stamps,
        lidar_angle_min,
        lidar_angle_max,
        lidar_angle_increment,
        lidar_range_min,
        lidar_range_max,
        lidar_ranges,
        lidar_stamps,
        imu_angular_velocity,
        imu_linear_acceleration,
        imu_stamps,
        disp_stamps,
        rgb_stamps
      )

if __name__ == '__main__':
    dataset = 20
    all_data = load_all_data(dataset)
