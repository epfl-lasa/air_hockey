import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
import time
from process_data import parse_list, parse_value, get_impact_time_from_object, get_robot_data_at_hit, get_distance_travelled, get_info_at_hit_time

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
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file,
                     converters={'RosTime' : parse_value, 'Position': parse_list})
                    #  dtype={'RosTime': 'float64'})

    hit_time = get_impact_time_from_object(path_to_object_hit)
    datetime_hit_time= pd.to_datetime(hit_time, unit='s')
    # Convert the 'Time' column to datetime format
    df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')

    # Labels for the coordinates
    coordinate_labels = ['x', 'y', 'z']

    # Plot the data
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.plot(df['RosTime'], df['Position'].apply(lambda x: x[i]), label=f'Axis {coordinate_labels[i]}')

    plt.vlines(datetime_hit_time, ymin =0, ymax=np.array(df['Position'][0]).max(), colors = 'r')

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

    if show_plot : plt.show()

def plot_actual_vs_des(robot_csv, object_csv, inverse_effort=True, show_plot=True, 
                       data_to_plot=["Torque", "Pos", "Vel", "Inertia", "Flux", "Normed Vel", "Object"]):

    # Read CSV file into a Pandas DataFrame - ROBOT
    df = pd.read_csv(robot_csv, skiprows=1,
                     converters={'RosTime' : parse_value, 'JointPosition': parse_list, 'JointVelocity': parse_list, 'JointEffort': parse_list, 
                                 'TorqueCmd': parse_list, 'EEF_Position': parse_list, 'EEF_Orientation': parse_list, 'EEF_Velocity': parse_list, 
                                 'EEF_DesiredVelocity': parse_list, 'Inertia': parse_list, 'HittingFlux': parse_value})
                    #  dtype={'RosTime': 'float64'})
    
    df_obj = pd.read_csv(object_csv,
                     converters={'RosTime' : parse_value, 'Position': parse_list})

    # Define set values from first row
    df_top_row = pd.read_csv(robot_csv, nrows=1, header=None)
    top_row_list = df_top_row.iloc[0].to_list()
    des_flux = top_row_list[1]
    des_pos = parse_list(top_row_list[5])
    recorded_hit_time = top_row_list[3]
    print(f"Desired Flux: {des_flux} \n Desired Pos: [{des_pos[0]:.3f}, {des_pos[1]:.3f}, {des_pos[2]:.3f}] \n Hit Time: {pd.to_datetime(recorded_hit_time, unit='s')}")

    # Make title string
    filename = os.path.basename(robot_csv)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')
    # title_str = f"Robot data for iiwa {parts[1]}, hit #{parts[3]}" #filename_without_extension.replace('_', ' ')

    # Get actual hit time 
    hit_time = get_impact_time_from_object(object_csv, show_print=True)
    datetime_hit_time= pd.to_datetime(hit_time, unit='s')

    # Get the 'Time' column as datetime
    df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')

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
        axs[i].set_xlabel('Time [s]')
        fig.suptitle(f"Effort vs Cmd : iiwa {parts[1]}, hit #{parts[3]}, flux {des_flux}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    if "Vel" in data_to_plot:
        # Plot EEF_Velocities vs Desired Velocities
        fig, axs = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        for i in range(3):
            axs[i].plot(df['RosTime'], df['EEF_Velocity'].apply(lambda x: x[i]), label=f'Velocity')
            axs[i].plot(df['RosTime'], df['EEF_DesiredVelocity'].apply(lambda x: x[i]), color='r',linestyle='--', label=f'Desired')
            axs[i].axvline(datetime_hit_time, color = 'r')
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
            axs[i].axhline(y=des_pos[i], color='r', linestyle='--')
            axs[i].axvline(datetime_hit_time, color = 'r')
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
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Hitting flux [m/s]')
        ax.grid(True)
        fig.suptitle(f"Hitting Flux : iiwa {parts[1]}, hit #{parts[3]}")
        fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    if "Object" in data_to_plot:
        # remove double values using derivative
        x_values =  df_obj['Position'].apply(lambda x: x[0])
        df_obj['derivative'] = x_values.diff() / df_obj['RosTime'].diff()

        # remove zeros
        filtered_df = df_obj[df_obj['derivative'] != 0.0].copy()

        # to date time
        filtered_df['RosTime'] = pd.to_datetime(filtered_df['RosTime'], unit='s')

        # Plot Object position  
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        for i in range(3):
            ax.plot(filtered_df['RosTime'], filtered_df['Position'].apply(lambda x: x[i]), label=f'Axis {coordinate_labels[i]}')

        ax.axvline(datetime_hit_time, color = 'r')
        fig.suptitle(f"Object Position: iiwa {parts[1]}, hit #{parts[3]} ")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position')
        ax.legend()
        ax.grid(True)
        # fig.tight_layout(rect=(0.01,0.01,0.99,0.99)) 

    hit_time = get_impact_time_from_object(path_to_object_hit)
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
        
        path_to_object_hit = path_to_data_airhockey + f"{folder_name}/object_hit_{hit}.csv"
        
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
                                converters={'RosTime' : parse_value, 'Position': parse_list})
                
                # Rewrite time to be relative 
                temp_time = np.linspace(0,df_obj['RosTime'].iloc[-1]-df_obj['RosTime'].iloc[0], len(df_obj['RosTime']))
                df_obj['RosTime'] = temp_time

                for i in range(3):
                    axs_obj[i].plot(df_obj['RosTime'], df_obj['Position'].apply(lambda x: x[i]))
                    axs_obj[i].set_title(f'Axis {coordinate_labels[i]}')
                    axs_obj[i].grid(True)
                    
                axs_obj[i].set_xlabel('Time [s]')
                fig_obj.suptitle(f"Object data for hit #{hit_numbers[0]}-{hit_numbers[-1]}")
                fig_obj.tight_layout(rect=(0.01,0.01,0.99,0.99)) 
            
        else :
            print(f"No iiwa_{iiwa_number} data file for hit #{hit} \n")
        
        hit_time = get_impact_time_from_object(path_to_object_hit)
        flux_at_hit, inertia_at_hit, pos, orient = get_robot_data_at_hit(path_to_robot_hit, hit_time, show_print=False)
        norm_distance = get_distance_travelled(path_to_object_hit, show_print=False)
        
        print(f"Hit #{parts[3]}\n"
              f" Hitting Flux: {flux_at_hit:.4f} \n"
              f" Hitting inertia: {inertia_at_hit:.4f} \n"
              f" Distance travelled (norm): {norm_distance:.3f}")
        
        
    plt.show()


# TESTING STUFF OUT             
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

def process_timestamped_folders(root_folder):    
   
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path) and len(folder) == 19 and folder[10] == '_':
            # Check if the folder name matches the timestamp format
            timestamp = folder.replace('_', ' ')
            try:
                datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            # Iterate through CSV files in the folder
            for file in os.listdir(folder_path):
                if file.startswith('iiwa'):
                    file_path = os.path.join(folder_path, file)
                    plot_robot_data(file_path)
                if file.startswith('object'):
                    file_path = os.path.join(folder_path, file)
                    plot_robot_data(file_path)


if __name__== "__main__" :

    path_to_data_airhockey = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey/"
    
    # READ from file using index or enter manually
    read_hit_info_from_file = True

    ### Plots variables
    if read_hit_info_from_file:
        index_to_plot = 789 ## FILL THIS IF ABOVE IS TRUE
        file_to_read = "all_data_march_clean.csv"

        processed_df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey_processed/"+file_to_read, index_col="Index")
        folder_name = processed_df['RecSession'].loc[index_to_plot] # "2024-03-05_14:04:43"
        hit_number = int(processed_df['HitNumber'].loc[index_to_plot]) #82 #[16,17]
        iiwa_number = processed_df['IiwaNumber'].loc[index_to_plot] #14
    
    else : ## OTHERWISE FILL THIS 
        folder_name = "2024-03-05_14:04:43"
        hit_number = 82 #[16,17]
        iiwa_number = 14
    

    ### DATA TO PLOT 
    plot_this_data = ["Flux","Vel","Torque","Object"]#["Vel", "Inertia", "Flux", "Normed Vel"]"Torque", "Vel", , "Joint Vel"


    # PLOT FOR SINGLE HIT 
    if isinstance(hit_number, int) :
        path_to_robot_hit = path_to_data_airhockey + f"{folder_name}/IIWA_{iiwa_number}_hit_{hit_number}.csv"
        path_to_object_hit = path_to_data_airhockey + f"{folder_name}/object_hit_{hit_number}.csv"

        # Plot one hit info with hit time 
        plot_actual_vs_des(path_to_robot_hit, path_to_object_hit, data_to_plot=plot_this_data)

    
    # PLOT SEVERAL HITS (hit_number should be a list)
    elif isinstance(hit_number, list):
        plot_all_des_vs_achieved(folder_name, hit_number, iiwa_number, data_to_plot=plot_this_data)

    