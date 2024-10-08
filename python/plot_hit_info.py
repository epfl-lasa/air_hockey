import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.data_processing_functions import *
import math 
from scipy.interpolate import interp1d
import pybullet


def get_latest_folder(path_to_data_airhockey, folder_name): 
    directories = os.listdir(path_to_data_airhockey)
    sorted_directories = sorted(directories, key=lambda x: x, reverse=True)
    folder_name = sorted_directories[0]    
    return folder_name

## PLOT FUNCTIONS
def plot_robot_data(csv_file, show_plot=True):

    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file, skiprows=1,
                     converters={'RosTime' : parse_value, 'JointPosition': parse_list, 'JointVelocity': parse_list, 'EEF_Position': parse_list, 
                                 'EEF_Orientation': parse_list, 'EEF_Velocity': parse_list, 'Inertia': parse_list, 'HittingFlux': parse_value})
                    #  dtype={'RosTime': 'float64'})
    
    df_top_row = pd.read_csv(csv_file, nrows=1)
    print(df_top_row)

    # Get the 'Time' column as datetime
    df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')

    print(df.head())

    # Labels for the coordinates
    coordinate_labels = ['x', 'y', 'z']

    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    # Plot each element of 'JointPositions'
    for i in range(7):
        axs[0].plot(df['RosTime'], df['JointPosition'].apply(lambda x: x[i]), label=f'Joint {i+1}')

    # Plot each element of 'JointVelocities'
    for i in range(7):
        axs[1].plot(df['RosTime'], df['JointVelocity'].apply(lambda x: x[i]), label=f'Joint {i+1}')

    # Plot each element of 'EEF_Position'
    for i in range(3):
        axs[2].plot(df['RosTime'], df['EEF_Position'].apply(lambda x: x[i]), label=f'Axis {coordinate_labels[i]}')

    # Plot each element of 'EEF_Velocity'
    for i in range(3):
        axs[3].plot(df['RosTime'], df['EEF_Velocity'].apply(lambda x: x[i]), label=f'Axis {coordinate_labels[i]}')

    # Plot hitting flux
    axs[4].plot(df['RosTime'], df['HittingFlux'])

    # Make title string
    filename = os.path.basename(csv_file)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')
    title_str = f"Robot data for iiwa {parts[1]}, hit #{parts[3]}" #filename_without_extension.replace('_', ' ')

    # Customize the plots
    plt.suptitle(title_str)
    axs[0].set_title('Joint Positions Over Time')
    axs[1].set_title('Joint Velocities Over Time')
    axs[2].set_title('EEF Position Over Time')
    axs[3].set_title('EEF Velocity Over Time')
    axs[4].set_title('Hitting Flux Over Time')

    axs[4].set_xlabel('Time [s]')

    axs[0].set_ylabel('Joint angle [rad]')
    axs[1].set_ylabel('Joint velocity [rad/s]')
    axs[2].set_ylabel('Position[m]')
    axs[3].set_ylabel('Speed [m/s]')
    axs[4].set_ylabel('Hitting Flux [m/s]')
    
    for ax in axs:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
        ax.grid(True)

    if show_plot : plt.show()

def plot_object_data(csv_file, show_plot=True):

    print(f"Reading {csv_file}")

    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file, skiprows=1,
                     converters={'RosTime' : parse_value, 'TimeWriting' : parse_value, 'PositionForIiwa7': parse_list, 'PositionWorldFrame': parse_list, 'Position': parse_list})
                    #  dtype={'RosTime': 'float64'})

    df = df[df.index<1000]
    offset_x = df['PositionForIiwa7'].iloc[0][1]- df['PositionWorldFrame'].iloc[0][0]

    x_values = df['PositionForIiwa7'].apply(lambda x: x[1])
    df['Derivative'] = x_values.diff() / df['RosTime'].diff()
    df = df[df['Derivative'] != 0.0].copy()

    x_values_world =  df['PositionWorldFrame'].apply(lambda x: x[0])
    df['derivative_world'] = x_values_world.diff() / df['RosTime'].diff()

    hit_time = get_impact_time_from_object(path_to_object_hit, pos_name_str='PositionForIiwa7')
    datetime_hit_time= pd.to_datetime(hit_time, unit='s')
    # Convert the 'Time' column to datetime format
    df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')
    df['TimeWriting'] = pd.to_datetime(df['TimeWriting'], unit='s')

    # Labels for the coordinates
    coordinate_labels = ['x', 'y', 'z']

    # Plot the data
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(df['RosTime'], df['PositionForIiwa7'].apply(lambda x: x[i]), label=f'Axis {coordinate_labels[i]}')
        # plt.plot(df['TimeWriting'], df['PositionForIiwa7'].apply(lambda x: x[i]), label=f'Axis {coordinate_labels[i]} - TimeWriting')
        plt.plot(df['RosTime'], df['PositionWorldFrame'].apply(lambda x: x[i]+ offset_x), label=f'Axis {coordinate_labels[i]} - World')

    plt.vlines(datetime_hit_time, ymin =0, ymax=np.array(df['PositionForIiwa7'][0]).max(), colors = 'r')

    # Make title string
    filename = os.path.basename(csv_file)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')
    title_str = f"Object data for hit #{parts[2]}"

    # Customize the plot
    plt.title(title_str)
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)


    # Plot Object velocity
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

    # ax.plot(df['RosTime'], df['Derivative'], label=f'Axis y')
    ax.plot(df['RosTime'], df['Derivative'], label=f'Axis y')
    ax.plot(df['RosTime'], df['derivative_world'], label=f'Axis x - world')

    # ax.axvline(datetime_hit_time, color = 'r')
    # ax.axvline(recorded_hit_time, color = 'g')
    fig.suptitle(f"Object Velocity: iiwa {parts[1]}, hit #{parts[2]} ")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.grid(True)


    if show_plot : plt.show()

def plot_actual_vs_des(robot_csv, object_csv, inverse_effort=True, show_plot=True, 
                       data_to_plot=["Torque", "Pos", "Vel", "Inertia", "Flux", "Normed Vel", "Object"]):

    # Read CSV file into a Pandas DataFrame - ROBOT
    df = pd.read_csv(robot_csv, skiprows=1,
                     converters={'RosTime' : parse_value, 'JointPosition': parse_list, 'JointVelocity': parse_list, 'JointEffort': parse_list, 
                                 'TorqueCmd': parse_list, 'EEF_Position': parse_list, 'EEF_Orientation': parse_list, 'EEF_Velocity': parse_list, 
                                 'EEF_DesiredVelocity': parse_list, 'EEF_CommandedVelocity':parse_list, 'Inertia': parse_list,
                                 'DirGrad': parse_list, 'HittingFlux': parse_value})
                    #  dtype={'RosTime': 'float64'})
    
    df_obj = pd.read_csv(object_csv,
                     converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list,
                                  'OrientationForIiwa7': parse_list, 'OrientationForIiwa14': parse_list})

    # Define set values from first row
    df_top_row = pd.read_csv(robot_csv, nrows=1, header=None)
    top_row_list = df_top_row.iloc[0].to_list()
    des_flux = top_row_list[1]
    des_pos = parse_list(top_row_list[5])
    recorded_hit_time = pd.to_datetime(top_row_list[3], unit='s')
    print(f"Desired Flux: {des_flux} \n Desired Pos: [{des_pos[0]:.3f}, {des_pos[1]:.3f}, {des_pos[2]:.3f}] \n Hit Time: {pd.to_datetime(recorded_hit_time, unit='s')}")

    # Make title string
    filename = os.path.basename(robot_csv)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')
    # title_str = f"Robot data for iiwa {parts[1]}, hit #{parts[3]}" #filename_without_extension.replace('_', ' ')

    # Get actual hit time 
    hit_time, stop_time = get_impact_time_from_object(object_csv, show_print=True)
    datetime_hit_time= pd.to_datetime(hit_time, unit='s')
    datetime_stop_time= pd.to_datetime(stop_time, unit='s')

    df_during_hit = df_obj[df_obj['RosTime']>hit_time]
    df_during_hit = df_during_hit[df_during_hit['RosTime']<stop_time]

    ### FOR OBJECT - get pos relative to iiwa
    if (parts[1]== '7'):
        obj_pos = df_obj['PositionForIiwa7']
        obj_pos_during_hit = df_during_hit['PositionForIiwa7']
    elif (parts[1] == '14'):
        obj_pos = df_obj['PositionForIiwa14']
        obj_pos_during_hit = df_during_hit['PositionForIiwa14']

    x_values = obj_pos.apply(lambda x: x[1])   

    ### test out stuff for data for ekf
    # Interpolate
    f_linear = interp1d(df_obj['RosTime'][::5], x_values[::5], kind='linear')
    f_cubic = interp1d(df_obj['RosTime'][::5], x_values[::5], kind='cubic')
    new_time = np.linspace(df_obj['RosTime'].min(), df_obj['RosTime'].max()-0.05, len(df_obj['RosTime'].index)*2)
    y_linear = f_linear(new_time)
    y_cubic = f_cubic(new_time)

    window_size = 3
    y_moving_average = np.convolve(x_values, np.ones(window_size)/window_size, mode='valid')

    window_size = 5
    y_during_hit = obj_pos_during_hit.apply(lambda x: x[1])   
    y_moving_average_during_hit = np.convolve(y_during_hit, np.ones(window_size)/window_size, mode='valid')

    dy_lin = np.gradient(y_linear, new_time)
    dy_cubic = np.gradient(y_cubic, new_time)
    dy_moving_average = np.gradient(y_moving_average, df_obj['RosTime'][len(df_obj['RosTime']) - len(y_moving_average):])
    dy_moving_average_during_hit = np.gradient(y_moving_average_during_hit, df_during_hit['RosTime'][len(df_during_hit['RosTime']) - len(y_moving_average_during_hit):])
    df_obj['Derivative'] = x_values.diff() / df_obj['RosTime'].diff()
    df_obj['Acceleration'] = df_obj['Derivative'].diff() / df_obj['RosTime'].diff()

    v_moving_average = np.convolve(df_obj['Derivative'], np.ones(window_size)/window_size, mode='valid')

    # Get the 'Time' column as datetime
    df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')
    new_time = pd.to_datetime(new_time, unit='s')
    df_during_hit['RosTime'] = pd.to_datetime(df_during_hit['RosTime'], unit='s')
    df_obj['RosTime'] = pd.to_datetime(df_obj['RosTime'], unit='s')
        
    # Labels for the coordinates
    coordinate_labels = ['x', 'y', 'z']

    # Inverse Effort for easier to read plots
    if inverse_effort: effort_factor = -1
    else: effort_factor = 1

    if "Torque" in data_to_plot:
        # Plot JointEffort vs TorqueCmd
        fig, axs = plt.subplots(7, 1, figsize=(15, 12), sharex=True)
        for i in range(7):
            axs[i].plot(df['RosTime'], effort_factor*df['JointEffort'].apply(lambda x: x[i]), label=f'Effort')
            axs[i].plot(df['RosTime'], df['TorqueCmd'].apply(lambda x: x[i]), color='r', linestyle='--', label=f'Torque Cmd')
            axs[i].set_title(f'Joint{i+1}')
            axs[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
            axs[i].grid(True)
            axs[i].axvline(datetime_hit_time, color = 'r')
            # axs[i].axvline(recorded_hit_time, color = 'g')
        axs[i].set_xlabel('Time [s]')
        fig.suptitle(f"Effort vs Cmd : iiwa {parts[1]}, hit #{parts[3]}, flux {des_flux}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    if "Vel" in data_to_plot:
        # Plot EEF_Velocities vs Desired Velocities
        fig, axs = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        for i in range(3):
            axs[i].plot(df['RosTime'], df['EEF_Velocity'].apply(lambda x: x[i]), label=f'Velocity')
            axs[i].plot(df['RosTime'], df['EEF_DesiredVelocity'].apply(lambda x: x[i]), color='r',linestyle='--', label=f'Desired')
            # axs[i].plot(df['RosTime'], df['EEF_CommandedVelocity'].apply(lambda x: x[i]), color='g',linestyle='--', label=f'Commanded')
            axs[i].axvline(datetime_hit_time, color = 'r')
            # axs[i].axvline(recorded_hit_time, color = 'g')
            axs[i].set_title(f'Axis {coordinate_labels[i]}')
            axs[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
            axs[i].grid(True)
        axs[i].set_xlabel('Time [s]')
        fig.suptitle(f"EEF Velocities vs Desired : iiwa {parts[1]}, hit #{parts[3]}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    if "Pos" in data_to_plot:
        # Plot EEF_Position vs Desired Psosiont
        fig, axs = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        for i in range(3):
            axs[i].plot(df['RosTime'], df['EEF_Position'].apply(lambda x: x[i]), label=f'Position')
            axs[i].axhline(y=des_pos[i], color='r', linestyle='--', label=f'Desired')
            axs[i].axvline(datetime_hit_time, color = 'r')
            # axs[i].axvline(recorded_hit_time, color = 'g')
            axs[i].set_title(f'Axis {coordinate_labels[i]}')
            axs[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
            axs[i].grid(True)
        axs[i].set_xlabel('Time [s]')
        fig.suptitle(f"EEF Position vs Desired : iiwa {parts[1]}, hit #{parts[3]}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    if "Flux" in data_to_plot:
        # Plot Flux
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax.plot(df['RosTime'], df['HittingFlux'])
        ax.axhline(y=des_flux, color='r', linestyle='--')
        ax.axvline(datetime_hit_time, color = 'r')
        # ax.axvline(recorded_hit_time, color = 'g')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Hitting flux [m/s]')
        ax.grid(True)
        fig.suptitle(f"Hitting Flux : iiwa {parts[1]}, hit #{parts[3]}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))
        
    if "Inertia" in data_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        # First project it
        if iiwa_number == 7:
            des_direction = np.array([[0.0], [1.0], [0.0]])
        elif iiwa_number == 14:
            des_direction = np.array([[0.0], [-1.0], [0.0]])
        
        # NOTE : Inertia recorded is actually Inertia Task Position INVERSE
        projected_inertia = df['Inertia'].apply(lambda x : 1/(des_direction.T @ np.reshape(x, (3,3)) @ des_direction))

        # Then plot
        ax.plot(df['RosTime'], projected_inertia)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Inertia [kg.m^2]')
        ax.grid(True)
        fig.suptitle(f"Projected Inertia : iiwa {parts[1]}, hit #{parts[3]}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99)) 
    
    if "Joint Vel" in data_to_plot:
        fig, axs = plt.subplots(7, 1, figsize=(15, 12), sharex=True)
        for i in range(7):
            axs[i].plot(df['RosTime'], df['JointVelocity'].apply(lambda x: x[i]))
            axs[i].set_title(f'Joint{i+1}')
            axs[i].grid(True)
        axs[i].set_xlabel('Time [s]')
        fig.suptitle(f"Joint Velocity : iiwa {parts[1]}, hit #{parts[3]}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))
               
    if "Object" in data_to_plot:
        # Plot Object position  
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        for i in range(3):
            ax.plot(df_obj['RosTime'], obj_pos.apply(lambda x: x[i]), label=f'Axis {coordinate_labels[i]}')

        # ax.plot(new_time, y_linear, label=f'Linear interp y axis')
        # ax.plot(new_time, y_cubic, label=f'cubic interp y axis')
        # ax.plot(df_obj['RosTime'][len(df_obj['RosTime']) - len(y_moving_average):], dy_moving_average, label=f'cubic')
        ax.plot(df_obj['RosTime'][len(df_obj['RosTime']) - len(y_moving_average):], y_moving_average, label=f'moving average')
        ax.plot(df_during_hit['RosTime'][len(df_during_hit['RosTime']) - len(y_moving_average_during_hit):], y_moving_average_during_hit, label=f'moving average during hit')
        
        ax.axvline(datetime_hit_time, color = 'r')
        ax.axvline(datetime_stop_time, color = 'b')
        # ax.axvline(recorded_hit_time, color = 'g')
        fig.suptitle(f"Object Position: iiwa {parts[1]}, hit #{parts[3]} ")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position')
        ax.legend()
        ax.grid(True)
        # fig.tight_layout(rect=(0.01,0.01,0.99,0.99)) 

    if "ObjectVel" in data_to_plot:
        # Plot Object velocity
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        for i in range(3):
            ax.plot(df_obj['RosTime'], df_obj['Derivative'], label=f'Axis {coordinate_labels[i]}')
        
        # ax.plot(df_obj['RosTime'][len(df_obj['RosTime']) - len(v_moving_average):], v_moving_average, label='Moving average')
        # ax.plot(new_time, dy_lin, label=f'linear')
        # ax.plot(new_time, dy_cubic, label=f'cubic')
        ax.plot(df_obj['RosTime'][len(df_obj['RosTime']) - len(y_moving_average):], dy_moving_average, label=f'derivative of moving average')
        ax.plot(df_during_hit['RosTime'][len(df_during_hit['RosTime']) - len(y_moving_average_during_hit):], dy_moving_average_during_hit, label=f'derivative of moving average during hit')

        ax.axvline(datetime_hit_time, color = 'r')
        ax.axvline(datetime_stop_time, color = 'b')
        # ax.axvline(recorded_hit_time, color = 'g')
        fig.suptitle(f"Object Velocity: iiwa {parts[1]}, hit #{parts[3]} ")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity')
        ax.legend()
        ax.grid(True)
        # fig.tight_layout(rect=(0.01,0.01,0.99,0.99)) 
    
    if "ObjectAcc" in data_to_plot:
        # Plot Object velocity
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        for i in range(3):
            ax.plot(df_obj['RosTime'], df_obj['Acceleration'], label=f'Axis {coordinate_labels[i]}')

        ax.axvline(datetime_hit_time, color = 'r')
        ax.axvline(datetime_stop_time, color = 'b')
        # ax.axvline(recorded_hit_time, color = 'g')
        fig.suptitle(f"Object Accelertion: iiwa {parts[1]}, hit #{parts[3]} ")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration')
        ax.legend()
        ax.grid(True)
        # fig.tight_layout(rect=(0.01,0.01,0.99,0.99)) 
    
    if "Orient" in data_to_plot:
        
        quat_label = ["x","y","z","w"]
        
        # get pos relative to iiwa
        if (parts[1]== '7'):
            obj_pos = df_obj['PositionForIiwa7']
            missing_values = pd.Series([np.zeros(4) for _ in range((len(df_obj) - len(df["EEF_Orientation"])))])
            df_obj["RobotOrientation"] = pd.concat([df['EEF_Orientation'], missing_values], ignore_index=True)
            df_obj["OrientationError1"] = df_obj.apply(lambda row : pybullet.getEulerFromQuaternion(pybullet.getDifferenceQuaternion(row["OrientationForIiwa7"],row["RobotOrientation"])),axis=1).copy()
            df_obj["OrientationError2"] = df_obj.apply(lambda row : [pybullet.getEulerFromQuaternion(row["RobotOrientation"])[i]- pybullet.getEulerFromQuaternion(row["OrientationForIiwa7"])[i] for i in range(3)],axis=1).copy()
        elif (parts[1] == '14'):
            obj_pos = df_obj['PositionForIiwa14']
            missing_values = pd.Series([np.zeros(4) for _ in range((len(df_obj) - len(df["EEF_Orientation"])))])
            df_obj["RobotOrientation"] = pd.concat([df['EEF_Orientation'], missing_values], ignore_index=True)
            df_obj["OrientationError1"] = df_obj.apply(lambda row : pybullet.getEulerFromQuaternion(
                pybullet.getDifferenceQuaternion(row["OrientationForIiwa14"],row["RobotOrientation"])),axis=1).copy()
            df_obj["OrientationError2"] = df_obj.apply(lambda row : [pybullet.getEulerFromQuaternion(row["RobotOrientation"])[i]- pybullet.getEulerFromQuaternion(row["OrientationForIiwa14"])[i] for i in range(3)],axis=1).copy()
        
        # get wrapped orientation error
        # df_obj["OrientationError"] = df_obj["OrientationError"].apply(lambda x : [wrap_angle(i) for i in x]).copy()

        
        # Plot Orientation
        fig, axs = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        for i in range(3):
            axs[i].plot(df['RosTime'], df['EEF_Orientation'].apply(lambda x: math.degrees(pybullet.getEulerFromQuaternion(x)[i])), label=f'EEF Orientation')
            axs[i].plot(df_obj['RosTime'], df_obj['OrientationError1'].apply(lambda x: math.degrees(x[i])), label=f'Error quat')
            axs[i].plot(df_obj['RosTime'], df_obj['OrientationError2'].apply(lambda x: math.degrees(x[i])), label=f'Error euler')
            axs[i].plot(df_obj['RosTime'], df_obj['OrientationForIiwa7'].apply(lambda x: math.degrees(pybullet.getEulerFromQuaternion(x)[i])), label=f'Object')
            axs[i].axvline(datetime_hit_time, color = 'r')
            axs[i].set_ylabel(f'Orientation Error in {coordinate_labels[i]} [deg]')
            axs[i].legend()
            axs[i].grid(True)
        
        axs[i].set_xlabel('Time [s]')
        fig.suptitle(f"Orientation Error: iiwa {parts[1]}, hit #{parts[3]} ")

        # fig.tight_layout(rect=(0.01,0.01,0.99,0.99)) 

    if "Grad" in data_to_plot:
        # Plot JointEffort vs TorqueCmd
        fig, axs = plt.subplots(7, 1, figsize=(15, 12), sharex=True)
        for i in range(7):
            axs[i].plot(df['RosTime'], df['DirGrad'].apply(lambda x: x[i]), label=f'DirGrad')
            axs[i].set_title(f'Joint{i+1}')
            # axs[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
            axs[i].grid(True)
            axs[i].axvline(datetime_hit_time, color = 'r')
            # axs[i].axvline(recorded_hit_time, color = 'g')
        axs[i].set_xlabel('Time [s]')
        fig.suptitle(f"Inertia Task Pos Dir Grad : iiwa {parts[1]}, hit #{parts[3]}, flux {des_flux}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    
    # hit_time = get_impact_time_from_object(path_to_object_hit)
    flux_at_hit, inertia_at_hit, pos, orient = get_robot_data_at_hit(path_to_robot_hit, hit_time, show_print=False)
    norm_distance = get_distance_travelled(path_to_object_hit, show_print=False)
    
    print(f"Hit #{parts[3]}\n"
            f" Hitting Flux: {flux_at_hit:.4f} \n"
            f" Hitting inertia: {inertia_at_hit:.4f} \n"
            f" Distance travelled (norm): {norm_distance:.3f}")

    if show_plot : plt.show()

def plot_all_des_vs_achieved(folder_name, hit_numbers, iiwa_number, inverse_effort=True, 
                             data_to_plot=["Torque", "Pos", "Vel", "Inertia", "Flux", "Normed Vel", "Object"]):
    
    # Create figures 
    if "Torque" in data_to_plot: fig_trq, axs_trq = plt.subplots(7, 1, figsize=(15, 12), sharex=True)
    if "Vel" in data_to_plot: fig_vel, axs_vel = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    if "Pos" in data_to_plot: fig_pos, axs_pos = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    if "Flux" in data_to_plot: fig_flux, ax_flux = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
    if "Inertia" in data_to_plot: fig_inertia, ax_inertia = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
    if "Normed Vel" in data_to_plot: fig_norm_vel, ax_norm_vel = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
    if "Joint Vel" in data_to_plot: fig_jnt_vel, axs_jnt_vel = plt.subplots(7, 1, figsize=(15, 12), sharex=True)
    if "Object" in data_to_plot: fig_obj, axs_obj = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    for hit in hit_numbers:
        
        path_to_object_hit = path_to_data_airhockey + f"{folder_name}/object_{object_number}_hit_{hit}.csv"
        
        if os.path.exists(path_to_data_airhockey + f"{folder_name}/IIWA_{iiwa_number}_hit_{hit}.csv"):
            path_to_robot_hit = path_to_data_airhockey + f"{folder_name}/IIWA_{iiwa_number}_hit_{hit}.csv"
            
            # Read CSV file into a Pandas DataFrame
            df = pd.read_csv(path_to_robot_hit, skiprows=1,
                            converters={'RosTime' : parse_value, 'JointPosition': parse_list, 'JointVelocity': parse_list, 'JointEffort': parse_list, 
                                        'TorqueCmd': parse_list, 'EEF_Position': parse_list, 'EEF_Orientation': parse_list, 'EEF_Velocity': parse_list, 
                                        'EEF_DesiredVelocity': parse_list, 'Inertia': parse_list, 'HittingFlux': parse_value})
                            #  dtype={'RosTime': 'float64'})
            
            # Define set values from first row
            df_top_row = pd.read_csv(path_to_robot_hit, nrows=1, header=None)
            top_row_list = df_top_row.iloc[0].to_list()
            des_flux = top_row_list[1]
            des_pos = parse_list(top_row_list[5])
            recorded_hit_time = top_row_list[3]
            # print(f"Desired Flux: {des_flux} \n Desired Pos: [{des_pos[0]:.3f}, {des_pos[1]:.3f}, {des_pos[2]:.3f}] \n Hit Time: {pd.to_datetime(recorded_hit_time, unit='s')}")

            # Make title string
            filename = os.path.basename(path_to_robot_hit)
            filename_without_extension = os.path.splitext(filename)[0]
            parts = filename_without_extension.split('_')
            # title_str = f"Robot data for iiwa {parts[1]}, hit #{parts[3]}" #filename_without_extension.replace('_', ' ')

            # Rewrite time to be relative 
            temp_time = np.linspace(0,df['RosTime'].iloc[-1]-df['RosTime'].iloc[0], len(df['RosTime']))
            df['RosTime'] = temp_time

            # Labels for the coordinates
            coordinate_labels = ['x', 'y', 'z']

            # Inverse Effort for easier to read plots
            if inverse_effort: effort_factor = -1
            else: effort_factor = 1

            # Plot JointEffort vs TorqueCmd
            if "Torque" in data_to_plot:
                for i in range(7):
                    axs_trq[i].plot(df['RosTime'], effort_factor*df['JointEffort'].apply(lambda x: x[i]))
                    axs_trq[i].plot(df['RosTime'], df['TorqueCmd'].apply(lambda x: x[i]), linestyle='--')
                    axs_trq[i].set_title(f'Joint{i+1}')
                    # axs_trq[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
                    axs_trq[i].grid(True)
                axs_trq[i].set_xlabel('Time [s]')
                fig_trq.suptitle(f"Effort vs Cmd : iiwa {parts[1]}, hit #{hit_numbers[0]}-{hit_numbers[-1]}, flux {des_flux}")
                fig_trq.tight_layout(rect=(0.01,0.01,0.99,0.99))


            # Plot EEF_Velocities vs Desired Velocities
            if "Vel" in data_to_plot:
                for i in range(3):
                    axs_vel[i].plot(df['RosTime'], df['EEF_Velocity'].apply(lambda x: x[i]), label=f'Velocity')
                    axs_vel[i].plot(df['RosTime'], df['EEF_DesiredVelocity'].apply(lambda x: x[i]), linestyle='--', label=f'Desired')
                    axs_vel[i].set_title(f'Axis {coordinate_labels[i]}')
                    # axs_vel[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
                    axs_vel[i].grid(True)
                axs_vel[i].set_xlabel('Time [s]')
                fig_vel.suptitle(f"EEF Velocities vs Desired : iiwa {parts[1]}, hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_vel.tight_layout(rect=(0.01,0.01,0.99,0.99))


            # Plot EEF_Position vs Desired Psosiont
            if "Pos" in data_to_plot:
                for i in range(3):
                    axs_pos[i].plot(df['RosTime'], df['EEF_Position'].apply(lambda x: x[i]), label=f'Position')
                    axs_pos[i].axhline(y=des_pos[i], color='r', linestyle='--')
                    axs_pos[i].set_title(f'Axis {coordinate_labels[i]}')
                    # axs_pos[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
                    axs_pos[i].grid(True)
                axs_pos[i].set_xlabel('Time [s]')
                fig_pos.suptitle(f"EEF Position vs Desired : iiwa {parts[1]}, hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_pos.tight_layout(rect=(0.01,0.01,0.99,0.99))

            # Plot Inertia
            if "Inertia" in data_to_plot:
                # First project it
                if iiwa_number == 7:
                    des_direction = np.array([[0.0], [1.0], [0.0]])
                elif iiwa_number == 14:
                    des_direction = np.array([[0.0], [-1.0], [0.0]])
                
                # NOTE : Inertia recorded is actually Inertia Task Position INVERSE
                projected_inertia = df['Inertia'].apply(lambda x : 1/(des_direction.T @ np.reshape(x, (3,3)) @ des_direction))

                # Then plot
                ax_inertia.plot(df['RosTime'], projected_inertia)
                ax_inertia.set_xlabel('Time [s]')
                ax_inertia.set_ylabel('Inertia [kg.m^2]')
                ax_inertia.grid(True)
                fig_inertia.suptitle(f"Projected Inertia : iiwa {parts[1]}, hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_inertia.tight_layout(rect=(0.01,0.01,0.99,0.99)) 
                      
            # Plot Flux
            if "Flux" in data_to_plot:
                ax_flux.plot(df['RosTime'], df['HittingFlux'], label='recorded')
                ax_flux.axhline(y=des_flux, color='r', linestyle='--')
                ax_flux.set_xlabel('Time [s]')
                ax_flux.set_ylabel('Hitting flux [m/s]')
                ax_flux.grid(True)
                # ax_flux.legend()
                fig_flux.suptitle(f"Hitting Flux : iiwa {parts[1]}, hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_flux.tight_layout(rect=(0.01,0.01,0.99,0.99))
                 
            # Plot Normed velocity
            if "Normed Vel" in data_to_plot:
                ax_norm_vel.plot(df['RosTime'], df['EEF_Velocity'].apply(lambda x: np.linalg.norm(x)))
                ax_norm_vel.set_xlabel('Time [s]')
                ax_norm_vel.set_ylabel('Normed Velocity [kg.m^2]')
                ax_norm_vel.grid(True)
                fig_norm_vel.suptitle(f"Normed_velocity: iiwa {parts[1]}, hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_norm_vel.tight_layout(rect=(0.01,0.01,0.99,0.99))            
            
            if "Joint Vel" in data_to_plot:
                for i in range(7):
                    axs_jnt_vel[i].plot(df['RosTime'], df['JointVelocity'].apply(lambda x: x[i]))
                    axs_jnt_vel[i].set_title(f'Joint{i+1}')
                    # axs_jnt_vel[i].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
                    axs_jnt_vel[i].grid(True)
                axs_jnt_vel[i].set_xlabel('Time [s]')
                fig_jnt_vel.suptitle(f"Joint Velocity : iiwa {parts[1]}, hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_jnt_vel.tight_layout(rect=(0.01,0.01,0.99,0.99))
            
            # Plot Object position
            if "Object" in data_to_plot:
                df_obj = pd.read_csv(path_to_object_hit,
                                converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})
                
                # Rewrite time to be relative 
                temp_time = np.linspace(0,df_obj['RosTime'].iloc[-1]-df_obj['RosTime'].iloc[0], len(df_obj['RosTime']))
                df_obj['RosTime'] = temp_time
                
                # get pos relative to iiwa
                if (iiwa_number == 7):
                    obj_pos = df_obj['PositionForIiwa7']
                elif (iiwa_number == 14):
                    obj_pos = df_obj['PositionForIiwa14']

                for i in range(3):
                    axs_obj[i].plot(df_obj['RosTime'], obj_pos.apply(lambda x: x[i]))
                    axs_obj[i].set_title(f'Axis {coordinate_labels[i]}')
                    axs_obj[i].grid(True)
                    
                axs_obj[i].set_xlabel('Time [s]')
                fig_obj.suptitle(f"Object data for hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_obj.tight_layout(rect=(0.01,0.01,0.99,0.99)) 
            
            # Print info
            hit_time, stop_time = get_impact_time_from_object(path_to_object_hit)
            flux_at_hit, inertia_at_hit, pos, orient = get_robot_data_at_hit(path_to_robot_hit, hit_time, show_print=False)
            norm_distance = get_distance_travelled(path_to_object_hit, show_print=False)
            
            print(f"Hit #{parts[3]}\n"
                f" Hitting Flux: {flux_at_hit:.4f} \n"
                f" Hitting inertia: {inertia_at_hit:.4f} \n"
                f" Distance travelled (norm): {norm_distance:.3f}")
            
        else :
            print(f"No iiwa_{iiwa_number} data file for hit #{hit} \n")   
        
    plt.show()


if __name__== "__main__" :

    path_to_data_airhockey = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey/"
    # path_to_data_airhockey = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/varying_flux_datasets/D1/"
 
    # READ from file using index or enter manually
    read_hit_info_from_file = False

    ### Plots variables
    if read_hit_info_from_file:
        index_to_plot = 105 # 2168 #2176# 2267 #2299 ## FILL THIS IF ABOVE IS TRUE
        file_to_read = "all_data_march.csv" #"D1_clean.csv" #
        object_number = 1

        processed_df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey_processed/raw/"+file_to_read, index_col="Index")
        folder_name = processed_df['RecSession'].loc[index_to_plot] # "2024-03-05_14:04:43"
        hit_number = int(processed_df['HitNumber'].loc[index_to_plot]) #82 #[16,17]
        iiwa_number = processed_df['IiwaNumber'].loc[index_to_plot] #14
    
    else : ## OTHERWISE FILL THIS 
        folder_name = "latest" #"latest" # "2024-04-30_11:26:14" ##"2024-04-30_10:25:25"  # 
        hit_number = 18 # ##[2,3,4,5,6] #[16,17] #
        iiwa_number = 7
        object_number = 1

    ### DATA TO PLOT 
    plot_this_data = ["Object", "ObjectVel", "ObjectAcc"]#"Pos","Vel" "Inertia", "Object","Torque", "Grad", "Joint Vel","Orient", "Pos"[, "Inertia", "Flux", "Normed Vel"]"Torque", "Vel", , "Joint Vel"
       

    # Get the latest folder
    if(folder_name == "latest"):
        folder_name = get_latest_folder(path_to_data_airhockey, folder_name)


    # PLOT FOR SINGLE HIT 
    if isinstance(hit_number, int) :
        path_to_robot_hit = path_to_data_airhockey + f"{folder_name}/IIWA_{iiwa_number}_hit_{hit_number}.csv"
        path_to_object_hit = path_to_data_airhockey + f"{folder_name}/object_{object_number}_hit_{hit_number}.csv"
        # path_to_object_hit = path_to_data_airhockey + f"{folder_name}/object_hit_{hit_number}.csv"

        # Plot one hit info with hit time 
        # plot_object_data(path_to_object_hit)
        plot_actual_vs_des(path_to_robot_hit, path_to_object_hit, data_to_plot=plot_this_data)

    
    # PLOT SEVERAL HITS (hit_number should be a list)
    elif isinstance(hit_number, list):
        plot_all_des_vs_achieved(folder_name, hit_number, iiwa_number, object_number, data_to_plot=plot_this_data)

    