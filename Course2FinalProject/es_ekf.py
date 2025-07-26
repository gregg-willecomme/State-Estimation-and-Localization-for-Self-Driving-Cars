# Main code for the Coursera SDC Course 2 final project
#
# Author: Trevor Ablett
# University of Toronto Institute for Aerospace Studies

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rotations import Quaternion, skew_symmetric
sys.path.append('./data')

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For part 3, you will use p3_data.pkl.
################################################################################################
with open('data/pt3_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5) 
plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can just insert it into our solver as another position
# measurement, just as we do for GNSS. However, the LIDAR frame is not the same as the frame
# shared by the IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame
# using our known extrinsic calibration rotation matrix C_li and translation vector t_li_i.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# This is the correct calibration rotation matrix, corresponding to an euler rotation of 0.05, 0.05, .1.
C_li = np.array([
    [ 0.99376, -0.09722,  0.05466],
    [ 0.09971,  0.99401, -0.04475],
    [-0.04998,  0.04992,  0.9975 ]
])

# This is an incorrect calibration rotation matrix, corresponding to a rotation of 0.05, 0.05, 0.05
# C_li = np.array([
#     [ 0.9975 , -0.04742,  0.05235],
#     [ 0.04992,  0.99763, -0.04742],
#     [-0.04998,  0.04992,  0.9975 ]
# ])

t_li_i = np.array([0.5, 0.1, 0.5])

lidar.data = (C_li @ lidar.data.T).T + t_li_i


#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.1
var_imu_w = 1.0
var_gnss = 0.01
var_lidar = 0.25

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)     # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)     # measurement model jacobian

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values
p_est[0] = gt.p[0]      # Position of the vehicle
v_est[0] = gt.v[0]      # Velocity of the vehicle
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.eye(9)    # covariance of estimate(估计的协方差)
gnss_i = 0
lidar_i = 0

p_check = p_est[0]
v_check = v_est[0]
q_check = q_est[0]
p_cov_check = p_cov[0]

i = 0

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    """
    Perform the EKF measurement update step.

    Parameters:
    - sensor_var: 3x3 measurement noise covariance matrix (e.g. var_gnss * I)
    - p_cov_check: 9x9 predicted covariance matrix
    - y_k: 3x1 sensor measurement (GNSS or LiDAR position)
    - p_check: 3x1 predicted position
    - v_check: 3x1 predicted velocity
    - q_check: 4x1 predicted quaternion (orientation)

    Returns:
    - Updated position, velocity, orientation, and covariance
    """
    H_k = h_jac
    # 3.1 Compute Kalman Gain
    S_k = H_k @ p_cov_check @ H_k.T + sensor_var * np.eye(3)
    K_k = p_cov_check @ H_k.T @ np.linalg.inv(S_k)

    # 3.2 Compute error state
    delta_x = K_k @ (y_k - p_check)

    # 3.3 Correct predicted state
    p_check = p_check + delta_x[0:3]
    v_check = v_check + delta_x[3:6]
    delta_phi = delta_x[6:9]
    q_check = Quaternion(axis_angle=delta_phi).quat_mult(q_check, out='np')

    # 3.4 Compute corrected covariance
    p_cov_check = (np.eye(9) - K_k @ H_k) @ p_cov_check

    # 3.5 Return corrected estimates
    return p_check, v_check, q_check, p_cov_check


#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Propagation de l'état
    vfa = var_imu_f**2
    vfw = var_imu_w**2
    Q_km = delta_t**2 * np.diag([vfa, vfa, vfa, vfw, vfw, vfw])

    C_ns = Quaternion(*q_est[k - 1]).to_mat()
    f = imu_f.data[k - 1]
    w = imu_w.data[k - 1]

    p_check = p_est[k - 1] + delta_t * v_est[k - 1] + 0.5 * delta_t**2 * (C_ns @ f - g)
    v_check = v_est[k - 1] + delta_t * (C_ns @ f - g)
    q_check = Quaternion(axis_angle=w * delta_t).quat_mult(q_est[k - 1])

    F_k = np.eye(9)
    F_k[0:3, 3:6] = delta_t * np.eye(3)
    F_k[3:6, 6:9] = skew_symmetric((C_ns @ f).reshape(3, 1))

    L_k = np.zeros((9, 6))
    L_k[3:6, 0:3] = np.eye(3)
    L_k[6:9, 3:6] = np.eye(3)

    p_cov_check = F_k @ p_cov_check @ F_k.T + L_k @ Q_km @ L_k.T

    # 2. Vérification de la disponibilité des capteurs
    isGNSSAvailable = (gnss_i < len(gnss.t)) and (imu_f.t[k - 1] == gnss.t[gnss_i])
    isLIDARAvailable = (lidar_i < len(lidar.t)) and (imu_f.t[k - 1] == lidar.t[lidar_i])

    # 3. Mise à jour de l'état
    if isGNSSAvailable and not isLIDARAvailable:
        y_k = gnss.data[gnss_i]
        p_check, v_check, q_check, p_cov_check = measurement_update(
            var_gnss, p_cov_check, y_k, p_check, v_check, q_check
        )
        gnss_i += 1

    elif not isGNSSAvailable and isLIDARAvailable:
        y_k = lidar.data[lidar_i]
        p_check, v_check, q_check, p_cov_check = measurement_update(
            var_lidar, p_cov_check, y_k, p_check, v_check, q_check
        )
        lidar_i += 1

    elif isGNSSAvailable and isLIDARAvailable:
        # GNSS update
        y_k = gnss.data[gnss_i]
        p_check, v_check, q_check, p_cov_check = measurement_update(
            var_gnss, p_cov_check, y_k, p_check, v_check, q_check
        )
        gnss_i += 1

        # LIDAR update
        y_k = lidar.data[lidar_i]
        p_check, v_check, q_check, p_cov_check = measurement_update(
            var_lidar, p_cov_check, y_k, p_check, v_check, q_check
        )
        lidar_i += 1

    # 4. Stockage des valeurs estimées
    p_est[k] = p_check
    v_est[k] = v_check
    q_est[k] = q_check
    p_cov[k] = p_cov_check

    



#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Final Estimated Trajectory')
ax.legend()
ax.set_zlim(-1, 5)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty.
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error plots')
num_gt = gt.p.shape[0]
p_est_euler = []

# Convert estimated quaternions to euler angles
for q in q_est:
    p_est_euler.append(Quaternion(*q).to_euler())
p_est_euler = np.array(p_est_euler)

# Get uncertainty estimates from P matrix
p_cov_diag_std = np.sqrt(np.diagonal(p_cov, axis1=1, axis2=2))

titles = ['x', 'y', 'z', 'x rot', 'y rot', 'z rot']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])

for i in range(3):
    ax[1, i].plot(range(num_gt), gt.r[:, i] - p_est_euler[:num_gt, i])
    ax[1, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i+6], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i+6], 'r--')
    ax[1, i].set_title(titles[i+3])
plt.show()


#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
#p1_indices = [9000, 9400, 9800, 10200, 10600]
#p1_str = ''
#for val in p1_indices:
#    for i in range(3):
#        p1_str += '%.3f ' % (p_est[val, i])
#with open('pt1_submission.txt', 'w') as file:
#    file.write(p1_str)

# # Pt. 2 submission
#p2_indices = [9000, 9400, 9800, 10200, 10600]
#p2_str = ''
#for val in p2_indices:
#    for i in range(3):
#        p2_str += '%.3f ' % (p_est[val, i])
#with open('pt2_submission.txt', 'w') as file:
#    file.write(p2_str)

# Pt. 3 submission
p3_indices = [6800, 7600, 8400, 9200, 10000]
p3_str = ''
for val in p3_indices:
    for i in range(3):
        p3_str += '%.3f ' % (p_est[val, i])
with open('pt3_submission.txt', 'w') as file:
    file.write(p3_str)
