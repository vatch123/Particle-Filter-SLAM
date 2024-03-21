import gtsam
import numpy as np
import matplotlib.pyplot as plt

def create_pose(x, y, theta):
    """Create a 2D pose represented as a rigid transformation matrix."""
    return gtsam.Pose2(x, y, theta)

def add_noise_to_pose(pose, translation_noise=0.1, rotation_noise=np.deg2rad(5)):
    """Add Gaussian noise to a 2D pose."""
    noisy_pose = pose.retract(np.random.normal(0, translation_noise, 2).tolist() + [np.random.normal(0, rotation_noise)])
    return noisy_pose

# Define original poses with a loop closure (first and last pose are the same)
original_poses = [create_pose(1, 2, np.deg2rad(30)), create_pose(2, 3, np.deg2rad(45)), create_pose(3, 4, np.deg2rad(60)), create_pose(1, 2, np.deg2rad(30))]

# Add noise to these poses
noisy_poses = [add_noise_to_pose(pose) for pose in original_poses]

# Initialize factor graph
graph = gtsam.NonlinearFactorGraph()

# Noise model for the factors
noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, np.deg2rad(2.5)]))

# Add factors to the graph
for i, pose in enumerate(noisy_poses):
    graph.add(gtsam.PriorFactorPose2(gtsam.symbol('x', i), pose, noise_model))

# Adding a loop closure factor
loop_closure_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, np.deg2rad(5)]))
graph.add(gtsam.BetweenFactorPose2(gtsam.symbol('x', 0), gtsam.symbol('x', len(original_poses) - 1), gtsam.Pose2(), loop_closure_noise_model))

# Initial estimate
initial_estimate = gtsam.Values()
for i, pose in enumerate(noisy_poses):
    initial_estimate.insert(gtsam.symbol('x', i), pose)

# Optimization
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

# Extract trajectories and angles
time_steps = range(len(original_poses))
original_trajectory = [(pose.x(), pose.y()) for pose in original_poses]
noisy_trajectory = [(pose.x(), pose.y()) for pose in noisy_poses]
optimized_trajectory = [result.atPose2(gtsam.symbol('x', i)) for i in time_steps]
optimized_trajectory = [(pose.x(), pose.y()) for pose in optimized_trajectory]

original_angles = [pose.theta() for pose in original_poses]
noisy_angles = [pose.theta() for pose in noisy_poses]
optimized_angles = [result.atPose2(gtsam.symbol('x', i)).theta() for i in time_steps]

# Plotting
fig, axs = plt.subplots(2, figsize=(12, 8))

# Plot trajectories
axs[0].plot(*zip(*original_trajectory), 'go-', label='Ground Truth Trajectory')
axs[0].plot(*zip(*noisy_trajectory), 'b--', label='Noisy Trajectory')
axs[0].plot(*zip(*optimized_trajectory), 'rx-', label='Optimized Trajectory')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')
axs[0].set_title('Trajectories Comparison')
axs[0].legend()
axs[0].grid(True)

# Plot angles
axs[1].plot(time_steps, original_angles, 'go-', label='Ground Truth Angles')
axs[1].plot(time_steps, noisy_angles, 'b--', label='Noisy Angles')
axs[1].plot(time_steps, optimized_angles, 'rx-', label='Optimized Angles')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Angle (Radians)')
axs[1].set_title('Angles Over Time')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()