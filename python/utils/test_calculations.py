import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from utils.data_handling_functions import *

## Orientation calculations
def wrap_angle(angle_rad):
    # wraps an angle
    angle_deg = math.degrees(angle_rad)
    mod_angle = (angle_deg) % 180
    if angle_deg >= 180 : return 180-mod_angle
    if angle_deg <= -180: return -mod_angle
    else: return angle_deg

def get_orientation_error_x_y_z(q_a, q_b):
    ### Get 2 quaternions and return euler angles from q_a to q_b
    ## This works for 7 for some reason (but not 14)
     
    r_a = Rotation.from_quat(q_a).as_matrix()
    r_b = Rotation.from_quat(q_b).as_matrix()
    
    r_ab = r_a.T @ r_b
    
    euler_angles = Rotation.from_matrix(r_ab).as_euler('XYZ', degrees=True)

    # euler_angles_a = r_a @ euler_angles
    # print(r_a)
    
    return euler_angles

def get_orientation_error_manually(q_robot, q_object, iiwa_number):
    # compute error from robot EEF to Object in euler angles

    e_o = Rotation.from_quat(q_object).as_euler('xyz', degrees=True)
    e_r = Rotation.from_quat(q_robot).as_euler('XYZ', degrees=True)

    if iiwa_number == 7:
        diff_x = e_o[0] - e_r[0]
        diff_y = e_o[1] - e_r[2]
        diff_z = e_o[2] + e_r[1]
    elif iiwa_number == 14:
        # diff_x = e_o[0] + e_r[1]
        # diff_y = e_o[1] + e_r[2]
        # diff_z = e_o[2] - e_r[0]       # 1 - intrisic rotations
        diff_x = e_o[0] + e_r[2]
        diff_y = e_o[1] + e_r[0]
        diff_z = e_o[2] - e_r[1]         # 2 - extrinsic rotations

    return [-diff_x, -diff_y, -diff_z]

def get_orientation_error_in_correct_base(q_robot, q_object, iiwa_number):
    ### This works for 14 for some reason (but not 7)
    
    r_o_0 = Rotation.from_quat(q_object).as_matrix() # correct base
    r_r_r = Rotation.from_quat(q_robot).as_matrix() # incorrect base

    if iiwa_number == 7:
        r_0 = Rotation.from_quat([0.707, 0.0, 0.0, 0.707]).as_matrix() # 90 deg in x 
        r_r_0 = r_r_r @ r_0
    if iiwa_number == 14:
        r_0 = Rotation.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix() # -90 deg in Z, then -90 deg in X
        r_r_0 = r_r_r @ r_0
    
    r_o_to_r = r_o_0.T @ r_r_0
    
    euler_angles = Rotation.from_matrix(r_o_to_r).as_euler('XYZ', degrees=True)
    
    return euler_angles

def get_corrected_quat_object_2(q_object):
    # overcoming offset from object 2 (didn't set axis properly in motive)

    # Define the rotation angles
    angles = [-90, 0, 90]  # -90 degrees around x, 0 degrees around y, 90 degrees around z
    rotation = Rotation.from_euler('XYZ', angles, degrees=True) ## this hsould be dtermnied from the base orienation 

    rot = rotation * Rotation.from_quat(q_object)

    rot_euler = rot.as_euler('xyz')
    rot_euler_corrected = [rot_euler[0], -rot_euler[2], rot_euler[1]]

    rot_final = Rotation.from_euler('xyz', rot_euler_corrected)

    return rot_final.as_quat()

# FLUX TESTS            
def flux_DS_by_harshit(attractor_pos, current_inertia, current_position):
    
    # set values
    DS_attractor = np.reshape(np.array(attractor_pos), (-1,1))
    des_direction = np.array([[0.0], [-1.0], [0.0]])
    test_des_direction = np.array([[0.0], [-1.0], [0.0]])
    sigma = 0.2
    gain = -2.0 * np.identity(3)
    m_obj = 0.4
    dir_flux = 1.2

    # Finding the virtual end effector position
    relative_position = current_position - DS_attractor
    virtual_ee = DS_attractor + des_direction * (np.dot(relative_position.T, test_des_direction) / np.linalg.norm(des_direction)**2)
    
    dir_inertia = 1/np.dot(des_direction.T, np.dot(current_inertia, des_direction))
    
    exp_term = np.linalg.norm(current_position - virtual_ee)

    alpha = np.exp(-exp_term / (sigma * sigma))

    reference_vel = alpha * des_direction + (1 - alpha) * np.dot(gain, (current_position - virtual_ee))

    reference_velocity = (dir_flux / dir_inertia) * (dir_inertia + m_obj) * reference_vel / np.linalg.norm(reference_vel)

    return reference_velocity.T, reference_vel.T, virtual_ee.T

def flux_DS_by_maxime(attractor_pos, current_inertia, current_position):
    
    # set values
    DS_attractor = np.reshape(np.array(attractor_pos), (-1,1))
    des_direction = np.array([[0.0], [-1.0], [0.0]])
    sigma = 0.2
    gain = -2.0 * np.identity(3)
    m_obj = 0.4
    des_flux = 1.2

    # First fin directional inertia using DESIRED DIRECTION
    dir_inertia = 1/np.dot(des_direction.T, np.dot(current_inertia, des_direction))
    
    # define x dot start according to paper
    des_velocity = des_flux * (dir_inertia +m_obj)/dir_inertia
    des_direction_new = des_velocity*des_direction
    

    # Finding the virtual end effector position
    relative_position = current_position - DS_attractor
    virtual_ee = DS_attractor + des_direction * (np.dot(relative_position.T, des_direction) / np.linalg.norm(des_direction)**2)
    virtual_ee_new = DS_attractor + des_direction_new * (np.dot(relative_position.T, des_direction_new) / np.linalg.norm(des_direction_new)**2)
    
    exp_term = np.linalg.norm(current_position - virtual_ee)
    exp_term_new = np.linalg.norm(current_position - virtual_ee_new)

    alpha = np.exp(-exp_term / (sigma * sigma))
    alpha_new = np.exp(-exp_term_new / (sigma * sigma))

    reference_vel = alpha * des_direction + (1 - alpha) * np.dot(gain, (current_position - virtual_ee))
    reference_vel_new = alpha_new * des_direction_new + (1 - alpha_new) * np.dot(gain, (current_position - virtual_ee_new))
    

    reference_velocity = (des_flux / dir_inertia) * (dir_inertia + m_obj) * reference_vel / np.linalg.norm(reference_vel)
    reference_velocity_new = (des_flux / dir_inertia) * (dir_inertia + m_obj) * reference_vel_new / np.linalg.norm(reference_vel)
    print(reference_velocity.T, reference_velocity_new.T)

    return reference_velocity.T, reference_vel.T, virtual_ee.T

def test_flux_DS(csv_file):

    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file, skiprows=1,
                     converters={'RosTime' : parse_value, 'JointPosition': parse_list, 'JointVelocity': parse_list, 'JointEffort': parse_list, 
                                 'TorqueCmd': parse_list, 'EEF_Position': parse_list, 'EEF_Orientation': parse_list, 'EEF_Velocity': parse_list, 
                                 'EEF_DesiredVelocity': parse_list, 'Inertia': parse_list, 'HittingFlux': parse_value})
                    #  dtype={'RosTime': 'float64'})
    
    # get DS_attractor 
    df_top_row = pd.read_csv(csv_file, nrows=1, header=None)
    top_row_list = df_top_row.iloc[0].to_list()
    des_pos = parse_list(top_row_list[5])

    reshaped_inertia =  df['Inertia'].apply(lambda x : np.reshape(x, (3,3)))
    reshaped_position = df['EEF_Position'].apply(lambda x : np.reshape(x, (3,1)))
    ref_vel = np.zeros((len(df.index),3))
    ref_vel2 = np.zeros((len(df.index),3))
    virtual_ee = np.zeros((len(df.index),3))

    for i in range(len(df.index)):

          ref_vel[i], ref_vel2[i], virtual_ee[i] = (flux_DS_by_maxime(des_pos, reshaped_inertia.iloc[i], np.array(reshaped_position.iloc[i])))
        # print(ref_vel)

    # Plot the data
    plt.figure(figsize=(12, 6))
    
    # Labels for the coordinates
    coordinate_labels = ['x', 'y', 'z']

    for i in range(3):
        plt.plot(df['RosTime'], ref_vel[:,i], label=f'Python_DS {coordinate_labels[i]}')
        plt.plot(df['RosTime'], df['EEF_DesiredVelocity'].apply(lambda x: x[i]), label=f'Cpp_DS {coordinate_labels[i]}')

    # Make title string
    filename = os.path.basename(csv_file)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')
    title_str = f"Object data for hit #{parts[3]}"

    # Customize the plot
    plt.title(title_str)
    plt.xlabel('Time [s]')
    plt.ylabel('DS_vel')
    plt.legend()
    plt.grid(True)
    plt.show()
