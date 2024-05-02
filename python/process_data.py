import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import yaml
from scipy.spatial.transform import Rotation


# PARSING FUNCTIONS
def parse_list(cell):
    # Split the space-separated values and parse them as a list of floats
    return [float(value) for value in cell.split()]

def parse_value(cell):
    # convert string to float 
    return float(cell)

def parse_strip_list_with_commas(cell):
    # Split the comma-separated values and parse them as a list of floats
    return [float(value) for value in cell.strip("[]").split(",")]

def parse_strip_list(cell):
    # Split the space-separated values and parse them as a list of floats
    return [float(value) for value in cell.strip("[]").split()]

def get_orientation_error_x_y_z(q_a, q_b):
    ### Get 2 quaternions and return orientation error
    
    r_a = Rotation.from_quat(q_a).as_matrix()
    r_b = Rotation.from_quat(q_b).as_matrix()
    
    r_ab = r_a.T @ r_b
    
    euler_angles = Rotation.from_matrix(r_ab).as_euler('xyz', degrees=True)
    
    return euler_angles

def get_robot_data_at_hit(csv_file, hit_time, show_print=False, get_max_values=False):
    ### Returns robot data info at hit time : Flux, inertia, EFF pose

    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file, skiprows=1,
                     converters={'RosTime' : parse_value, 'JointPosition': parse_list, 'JointVelocity': parse_list, 'JointEffort': parse_list, 
                                 'TorqueCmd': parse_list, 'EEF_Position': parse_list, 'EEF_Orientation': parse_list, 'EEF_Velocity': parse_list, 
                                 'EEF_DesiredVelocity': parse_list, 'Inertia': parse_list, 'HittingFlux': parse_value})
                    #  dtype={'RosTime': 'float64'})
    
    # Get DF of values after hit
    post_hit_df = df[(df['RosTime']-hit_time) >= 0]

    # Check that we recorded at this time
    if post_hit_df.empty:
        print(f"ERROR Robot data stops before Hit Time for {csv_file}")
        return 0,0,0,0

    # Get flux at closest hit time 
    flux_at_hit = post_hit_df.iloc[0]['HittingFlux']

    # Get directional inertia at hit time
    # NOTE : Inertia recorded is actually Inertia Task Position INVERSE
    inertia_at_hit = post_hit_df.iloc[0]['Inertia']

    des_direction = np.array([[0.0], [1.0], [0.0]])
    dir_inertia_at_hit = 1/(des_direction.T @ np.reshape(inertia_at_hit, (3,3)) @ des_direction)[0,0]

    # Get EEF Position at closest hit time 
    pos_at_hit = post_hit_df.iloc[0]['EEF_Position']
    orient_at_hit = post_hit_df.iloc[0]['EEF_Orientation']

    ### DONE LIVE 
    # convert quaternion from W-xyz to xyz-W
    # new_orient_at_hit = orient_at_hit[1:] + [orient_at_hit[0]]
    # r = Rotation.from_quat(new_orient_at_hit) ## normalizes quaternion
    # new_orient_at_hit = r.as_quat()

    if get_max_values : 
        # Get max normed vel
        normed_vel = df['EEF_Velocity'].apply(lambda x: np.linalg.norm(x))
        max_vel = normed_vel.max()
        
        # Get max flux
        max_flux = df['HittingFlux'].abs().max()
    
    if show_print: 
        # Define set values from first row
        df_top_row = pd.read_csv(csv_file, nrows=1, header=None)
        top_row_list = df_top_row.iloc[0].to_list()
        des_flux = top_row_list[1]
        des_pos = parse_list(top_row_list[5])
        recorded_hit_time = top_row_list[3]
        # print(f"Desired Flux: {des_flux} \n Desired Pos: [{des_pos[0]:.3f}, {des_pos[1]:.3f}, {des_pos[2]:.3f}] \n Hit Time: {pd.to_datetime(recorded_hit_time, unit='s')}")

        # Make title string
        filename = os.path.basename(csv_file)
        filename_without_extension = os.path.splitext(filename)[0]
        parts = filename_without_extension.split('_')

        print(f"Hit #{parts[3]}, IIWA_{parts[1]} \n Desired Flux: {des_flux} \n Hitting Flux: {flux_at_hit:.4f}")
        print(f" Real Hit time : {hit_time} \n Recorded Hit Time : {pd.to_datetime(recorded_hit_time, unit='s')}")
    
    if get_max_values : 
        return max_flux, max_vel

    else: 
        return flux_at_hit, dir_inertia_at_hit, pos_at_hit, orient_at_hit
    
def get_impact_time_from_object(csv_file, show_print=False, return_indexes=False):    
    # Reads object csv file and returns impact time OR indexes for before_impact, after_impact, stop moving
    
    pos_name_str = 'PositionForIiwa7'

    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file,
                     converters={'RosTime' : parse_value, pos_name_str: parse_list}) # 'PositionForIiwa7'
                
    ### SOLUTION TO DEAL WITH RECORDING OF MANUAL MOVEMENT 
    # Use derivative to find changes in speed 
    derivative_threshold = 0.3 #0.15 #0.05

    # find start and end index by using derivative in x axis -- NOTE : ASSUME MOVEMENT IN X AXIS
    x_values =  df[pos_name_str].apply(lambda x: x[1])
    df['derivative'] = x_values.diff() / df['RosTime'].diff()

    # print(df['derivative'].tail(40))
    
    # remove zeros
    filtered_df = df[df['derivative'] != 0.0].copy()

    # get start and end index
    idx_start_moving =  (filtered_df['derivative'].abs() > derivative_threshold).idxmax() # detect 1st time derivative is non-zero
    idx_stop_moving = (filtered_df['derivative'].loc[idx_start_moving:].abs() < derivative_threshold).idxmax() # detect 1st time derivative comes back to zero
    idx_before_impact = idx_start_moving-1 # time just before impact - 10ms error due to Motive streaming at 120Hz 

    hit_time = df['RosTime'].iloc[idx_before_impact] # HIT TIME as float 

    if show_print: 
        df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')
        print(f"Start moving from {df[pos_name_str].iloc[idx_before_impact]} at {df['RosTime'].iloc[idx_before_impact]}")
        print(f"Stop moving from {df[pos_name_str].iloc[idx_stop_moving]} at {df['RosTime'].iloc[idx_stop_moving]}")

    if not return_indexes: 
        return hit_time
    elif return_indexes:
        if show_print : print("Return object movement indexes")
        return idx_before_impact, idx_start_moving, idx_stop_moving

def get_distance_travelled(csv_file, return_distance_in_y=False, show_print=False, show_hit=True):
   ### Returns distance travelled  
   
   # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file,
                     converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list})

    # # Make title string
    filename = os.path.basename(csv_file)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')

    idx_before_impact, idx_start_moving, idx_stop_moving = get_impact_time_from_object(csv_file, return_indexes=True)

    # print( df['PositionForIiwa7'].iloc[idx_stop_moving][1],  df['RosTime'].iloc[idx_stop_moving])
    
    ### Get distance in X = axis of hit in optitrack frame
    distance_in_y = df['PositionForIiwa7'].iloc[idx_before_impact][1]- df['PositionForIiwa7'].iloc[idx_stop_moving][1]
    #### Get distance in norm 
    norm_distance = np.linalg.norm(np.array(df['PositionForIiwa7'].iloc[idx_before_impact])-np.array(df['PositionForIiwa7'].iloc[idx_stop_moving]))
    
    if show_print :
        if show_hit:
            print(f"Hit #{parts[2]}, Object \n")
        print(f" Distance in Y-axis : {distance_in_y:.3f} \n Normed Distance : {norm_distance:.3f}")

    if return_distance_in_y : 
        return distance_in_y
    else : 
        return norm_distance

def get_object_orientation_at_hit(object_csv, hit_time, iiwa_number):

    # get hitting params file 
    # parameter_filepath = os.path.join(os.path.dirname(object_csv),"hitting_params.yaml")
    # with open(parameter_filepath, "r") as yaml_file:
    #     hitting_data = yaml.safe_load(yaml_file)
    # # read object offset
    # iiwa_nb_str = f"iiwa{iiwa_number}"
    # object_offset = [hitting_data[iiwa_nb_str]["object_offset"]["x"],hitting_data[iiwa_nb_str]["object_offset"]["y"],hitting_data[iiwa_nb_str]["object_offset"]["z"]]

    # get orientation
    df = pd.read_csv(object_csv, converters={'RosTime' : parse_value, 'OrientationForIiwa7': parse_list, 'OrientationForIiwa14': parse_list})
    
    if(iiwa_number == '7' ) :
        object_orient_at_hit =  df[(df['RosTime']-hit_time) >= 0].iloc[0]['OrientationForIiwa7']
    elif(iiwa_number == '14'):
        object_orient_at_hit =  df[(df['RosTime']-hit_time) >= 0].iloc[0]['OrientationForIiwa14']

    
    ## This is all now done live
    # Define the rotation matrix from camera frame to robot frame
    # rotation_mat_optitrack_to_robot = np.array([[0.0, -1.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 1.0]])

    # # convert quaternion from W-xyz to xyz-W
    # new_object_orient_at_hit = object_orient_at_hit[1:] + [object_orient_at_hit[0]]

    # # Convert the quaternion to a rotation matrix
    # r = Rotation.from_quat(new_object_orient_at_hit)
    # rotation_mat_optitrack = r.as_matrix()

    # # Multiply the rotation matrix representing the camera-to-robot transformation with the rotation matrix from the quaternion
    # rotation_mat_robot = np.dot(rotation_mat_optitrack_to_robot, rotation_mat_optitrack)

    # # Convert the resulting rotation matrix back to a quaternion
    # r_robot = Rotation.from_matrix(rotation_mat_robot)
    # q_robot_frame = r_robot.as_quat()

    return np.array(object_orient_at_hit)

def get_info_at_hit_time(robot_csv, object_csv):

    # get hit time 
    hit_time = get_impact_time_from_object(object_csv)

    # get flux, inertia on hit, EEF Pose
    hitting_flux, hitting_dir_inertia, hitting_pos, hitting_orientation = get_robot_data_at_hit(robot_csv, hit_time)

    # get distance travelled
    distance_travelled = get_distance_travelled(object_csv, show_print=False)

    # get recording session, iiwa number and hit number from robot_csv name 
    recording_session = os.path.basename(os.path.dirname(robot_csv))
    filename = os.path.basename(robot_csv)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')
    iiwa_number = parts[1]
    hit_number = parts[3]
    
    # get object orientation at hit
    object_orient = get_object_orientation_at_hit(object_csv, hit_time , iiwa_number)

    # get desired flux from top row 
    des_flux = pd.read_csv(robot_csv, nrows=1, header=None).iloc[0].to_list()[1]
    des_pos = parse_list(pd.read_csv(robot_csv, nrows=1, header=None).iloc[0].to_list()[5])

    # Should be ordered in the same way as output file columns
    return [recording_session, hit_number, iiwa_number, des_flux, hitting_flux,  distance_travelled,  hitting_dir_inertia, des_pos, hitting_pos, object_orient, hitting_orientation ]

def process_data_to_one_file(data_folder, recording_sessions, output_filename="test.csv"):
    
    path_to_data_airhockey = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/"+ data_folder +"/"
    output_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey_processed/" + output_filename

    output_df = pd.DataFrame(columns=["RecSession","HitNumber","IiwaNumber","DesiredFlux","HittingFlux","DistanceTraveled","HittingInertia","ObjectPos", "HittingPos", "ObjectOrientation", "HittingOrientation"])

    start_time= time.time()
    # process each rec_sess folder 
    for rec_sess in recording_sessions: 
        
        folder_name = os.path.join(path_to_data_airhockey,rec_sess)

        # Iterate over files in the folder and write filenames to dictionary
        hit_files = {}
        for filename in os.listdir(folder_name):
            # Check if the file matches the pattern
            match = re.match(r'(?:IIWA_\d+_hit_|object_\d+_hit_)(\d+)\.csv', filename)
            if match:
                hit_number = int(match.group(1))
                # Append the file to the corresponding hit_number key in the dictionary
                hit_files.setdefault(hit_number, []).append(os.path.join(folder_name, filename))

        # Iterate over pairs of files with the same hit number
        for hit_number, files in hit_files.items():
            # Check if there are at least two files with the same hit number
            if len(files) == 2:
                
                files.sort() # Sort the files to process them in order -> robot_csv, then object_csv
                
                ## DEBUG print(files[0])

                # process each "hit_number" set 
                output_df.loc[len(output_df)]= get_info_at_hit_time(files[0], files[1])


            else :
                print(f"ERROR : Wrong number of files, discarding the following : \n {files}")

        print(f"FINISHED {rec_sess}")

    # HACK - remove lines where flux AND inertia are 0 -> lines where we didn't record robot data at hit time
    clean_output_df = output_df[output_df['HittingFlux'] != 0.0].copy()
    print(f"Removing {len(output_df[output_df['HittingFlux'] == 0.0].index)} datapoints that were not recorded correctly")

    # Save output df as .csv
    clean_output_df.to_csv(output_path, index_label="Index")
    
    print(f"Took {time.time()-start_time} seconds \nProcessed impact info for folders : {recording_sessions}")

    return

def process_one_file_for_ekf(robot_csv, object_csv, output_folder):
    
    #read files
    df_obj = pd.read_csv(object_csv, converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list}) #'Position': parse_list, 'PositionForIiwa14': parse_list})
    df_robot = pd.read_csv(robot_csv, skiprows=1, converters={'RosTime' : parse_value, 'EEF_Position': parse_list, 'HittingFlux': parse_value})
    
    # Get different position depending on iiwa !
    filename = os.path.basename(robot_csv)
    filename_without_extension = os.path.splitext(filename)[0]
    parts = filename_without_extension.split('_')
    iiwa_number = parts[1]
    hit_number = parts[3]
    
    # newdf with correct shape - [time, z,y,x_o, z,y,x_ee]          
    # Merge the dataframes on the 'RosTime' column
    df_obj['RosTime'] = df_obj['RosTime'].apply(lambda x : round(x,3))
    df_robot['RosTime'] = df_robot['RosTime'].apply(lambda x : round(x,3))
    
    merged_df = pd.merge(df_obj, df_robot, on='RosTime', how='outer')
    merged_df = merged_df[['RosTime', 'PositionForIiwa7', 'EEF_Position', 'HittingFlux']]
    
    # Complete EEF_Position with last know EEF_value 
    last_eef_value = df_robot['EEF_Position'].iloc[-1]
    start_eef = merged_df['EEF_Position'].dropna()
    end_eef = pd.Series([last_eef_value]*(len(df_obj.index)-len(df_robot.index)))
    merged_df['EEF_Position'] = pd.concat([start_eef, end_eef], ignore_index=True)
    
    # Complete HittingFlux with last know  flux value 
    last_flux_value = df_robot['HittingFlux'].iloc[-1]
    start_flux = merged_df['HittingFlux'].dropna()
    end_flux = pd.Series([last_flux_value]*(len(df_obj.index)-len(df_robot.index)))
    merged_df['HittingFlux'] = pd.concat([start_flux, end_flux], ignore_index=True)
    merged_df.dropna(inplace=True)
    
    # Reformat for James' EKF -> WATCH OUT INVERTING X AND Y HERE (due to optitrack frame)
    formatted_data = merged_df.apply(lambda row : [row['RosTime'], row['PositionForIiwa7'][2],row['PositionForIiwa7'][0],row['PositionForIiwa7'][1], 
                                                    row['EEF_Position'][2],row['EEF_Position'][0],row['EEF_Position'][1], row['HittingFlux']], axis=1)

    formatted_df = pd.DataFrame(formatted_data.tolist(), columns=['time', 'z_o', 'y_o', 'x_o', 'z_eef', 'y_eef', 'x_eef', 'flux'])

    # offset x due to wrong frame ! 
    # df_top_row = pd.read_csv(robot_csv, nrows=1, header=None)
    # top_row_list = df_top_row.iloc[0].to_list()
    # des_pos = parse_list(top_row_list[5])
    # x_offset = float(des_pos[1]-formatted_df['x_o'].iloc[0])
    # formatted_df['x_o'] = formatted_df['x_o'].apply(lambda x: x+x_offset)
    
    # write to file with same name
    formatted_df.to_csv(os.path.join(output_folder,f"hit_{hit_number}-IIWA_{iiwa_number}.csv"), index=False)

def process_one_file_to_truncate_object_data(object_csv, output_folder):
    
    #read files
    df_obj = pd.read_csv(object_csv) #'Position': parse_list, 'PositionForIiwa14': parse_list})

    filename = os.path.basename(object_csv)
    
    cutoff_index = 355
    formatted_df = df_obj.iloc[:cutoff_index]

    # write to file with same name
    formatted_df.to_csv(os.path.join(output_folder,filename), index=False)

def process_all_data_for_ekf(recording_sessions):
    
    path_to_data_airhockey = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey/"
    path_to_data_ekf = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey_ekf/" #"/data/airhockey_consistency/"
    
    start_time= time.time()
    count_removed = 0 
    
    # process each rec_sess folder 
    for rec_sess in recording_sessions: 
        
        folder_name = os.path.join(path_to_data_airhockey,rec_sess)
        
        # create folder with same name 
        new_folder = os.path.join(path_to_data_ekf,rec_sess)
        os.makedirs(new_folder, exist_ok=True)

        # Iterate over files in the folder and write filenames to dictionary
        hit_files = {}
        for filename in os.listdir(folder_name):
            # Check if the file matches the pattern
            match = re.match(r'(?:IIWA_\d+_hit_|object_\d+_hit_)(\d+)\.csv', filename)
            if match:
                hit_number = int(match.group(1))
                # Append the file to the corresponding hit_number key in the dictionary
                hit_files.setdefault(hit_number, []).append(os.path.join(folder_name, filename))

        # Iterate over pairs of files with the same hit number
        for hit_number, files in hit_files.items():
            # Check if there are at least two files with the same hit number
            if len(files) == 2:
                
                files.sort() # Sort the files to process them in order -> robot_csv, then object_csv
                
                ## DEBUG print(files[0])

                # Process each "hit_number" set 
                robot_csv = files[0]
                object_csv = files[1]
                     
                # check if hitting flux is 0 at hit time -> badly recorded
                hit_time = get_impact_time_from_object(object_csv)
                hitting_flux, hitting_dir_inertia, hitting_pos, hitting_orientation = get_robot_data_at_hit(robot_csv, hit_time)
                
                if hitting_flux == 0 : 
                    count_removed +=1
                else :                
                    process_one_file_for_ekf(robot_csv, object_csv, new_folder)
                    # process_one_file_to_truncate_object_data(object_csv, new_folder)
                   
            else :
                print(f"ERROR : Wrong number of files, discarding the following : \n {files}")

        print(f"FINISHED {rec_sess}")

    print(f"Removing {count_removed} datapoints that were not recorded correctly")

    
    print(f"Took {time.time()-start_time} seconds \nProcessed impact info for folders : {recording_sessions}")

    return
    

if __name__== "__main__" :
   
    ### Processing variables 
    ### UBUNTU
    folders_to_process = ["2024-05-01_14:09:10"] #"2024-04-30_11:26:14", "2024-04-30_13:16:40" ] #["2024-03-05_12:20:48","2024-03-05_12:28:21","2024-03-05_14:04:43","2024-03-05_14:45:46","2024-03-05_15:19:15"] #,"2024-03-05_15:58:41",
                            # "2024-03-06_12:30:55", "2024-03-06_13:40:26","2024-03-06_13:52:53","2024-03-06_15:03:42"] 

    ### WINDOWS
    # folders_to_process = ["2024-03-05_12_20_48","2024-03-05_12_28_21","2024-03-05_14_04_43","2024-03-05_14_45_46","2024-03-05_15_19_15"]#,"2024-03-05_15_58_41"]#,
    #                     #    "2024-03-06_12_30_55", "2024-03-06_13_40_26","2024-03-06_13_52_53","2024-03-06_15_03_42" ]

    data_folder = "airhockey"
    process_data_to_one_file(data_folder, folders_to_process, output_filename="100_hits-object_1-config_1-fixed_start-random_flux-IIWA_7-reduced_inertia.csv")
    # process_all_data_for_ekf(folders_to_process)
    

    # path_to_data_airhockey = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey/"
    # path_to_data_ekf = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey_ekf/"
    
    # robot_csv = os.path.join(path_to_data_airhockey,"2024-04-26_15:29:59", "IIWA_7_hit_1.csv" )
    # object_csv = os.path.join(path_to_data_airhockey,"2024-04-26_15:29:59", "object_hit_1.csv" )
    # process_one_file_for_ekf(robot_csv, object_csv, path_to_data_ekf)