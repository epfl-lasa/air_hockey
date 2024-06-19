import os
import pandas as pd
import numpy as np

from utils.data_handling_functions import parse_list, parse_value

### Processing variables to detect object motion
DERIVATIVE_THRESHOLD_START = 0.4 # 0.4 # 0.15 ## high to avoid noise at start
DERIVATIVE_THRESHOLD_STOP = 0.01                ## low to get full distance at end


## Processing functions to extract HIT INFO
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
        return flux_at_hit, dir_inertia_at_hit, np.array(pos_at_hit), np.array(orient_at_hit)
    
def get_impact_time_from_object(csv_file, pos_name_str = 'PositionForIiwa7',show_print=False, return_indexes=False):    
    # Reads object csv file and returns impact time OR indexes for before_impact, after_impact, stop moving

    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file, skiprows=1,
                     converters={'RosTime' : parse_value, pos_name_str: parse_list}) # 'PositionForIiwa7'
                
    ### SOLUTION TO DEAL WITH RECORDING OF MANUAL MOVEMENT 
    # Use derivative to find changes in speed 
    # find start and end index by using derivative in x axis -- NOTE : ASSUME MOVEMENT IN Y AXIS
    x_values =  df[pos_name_str].apply(lambda x: x[1])
    df['derivative'] = x_values.diff() / df['RosTime'].diff()


    # print(df['derivative'].tail(40))
    
    # remove zeros
    filtered_df = df[df['derivative'] != 0.0].copy()
    

    # get start and end index
    idx_start_moving =  (filtered_df['derivative'].abs() > DERIVATIVE_THRESHOLD_START).idxmax() # detect 1st time derivative is non-zero
    idx_stop_moving = (filtered_df['derivative'].loc[idx_start_moving:].abs() < DERIVATIVE_THRESHOLD_STOP).idxmax() # detect 1st time derivative comes back to zero
    idx_before_impact = idx_start_moving-1 # time just before impact - 5ms error due to Recorder at 200Hz 

    hit_time = df['RosTime'].iloc[idx_before_impact] # HIT TIME as float 
    stop_time = df['RosTime'].iloc[idx_stop_moving] # HIT TIME as float 

    if show_print: 
        df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')
        # print(idx_stop_moving, filtered_df['derivative'].loc[idx_start_moving:idx_stop_moving].abs())
        print(f"Start moving from {df[pos_name_str].iloc[idx_before_impact]} at {df['RosTime'].iloc[idx_before_impact]}")
        print(f"Stop moving from {df[pos_name_str].iloc[idx_stop_moving]} at {df['RosTime'].iloc[idx_stop_moving]}")

    if not return_indexes: 
        return hit_time, stop_time
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

def get_object_info_during_hit(object_csv, hit_time, stop_time, iiwa_number):

    # get hitting params file 
    # parameter_filepath = os.path.join(os.path.dirname(object_csv),"hitting_params.yaml")
    # with open(parameter_filepath, "r") as yaml_file:
    #     hitting_data = yaml.safe_load(yaml_file)
    # # read object offset
    # iiwa_nb_str = f"iiwa{iiwa_number}"
    # object_offset = [hitting_data[iiwa_nb_str]["object_offset"]["x"],hitting_data[iiwa_nb_str]["object_offset"]["y"],hitting_data[iiwa_nb_str]["object_offset"]["z"]]

    # Get orientation
    df = pd.read_csv(object_csv, converters={'RosTime' : parse_value, 'OrientationForIiwa7': parse_list, 'OrientationForIiwa14': parse_list,  
                                             'PositionForIiwa7': parse_list,  'PositionForIiwa14': parse_list})
    
    if(iiwa_number == '7' ) :
        object_orient_at_hit =  df[(df['RosTime']-hit_time) >= 0].iloc[0]['OrientationForIiwa7']
        object_pos_at_hit = df[(df['RosTime']-hit_time) >= 0].iloc[0]['PositionForIiwa7']
        object_pos_final = df[(df['RosTime']-stop_time) >= 0].iloc[0]['PositionForIiwa7']
    elif(iiwa_number == '14'):
        object_orient_at_hit =  df[(df['RosTime']-hit_time) >= 0].iloc[0]['OrientationForIiwa14']
        object_pos_at_hit = df[(df['RosTime']-hit_time) >= 0].iloc[0]['PositionForIiwa14']
        object_pos_final = df[(df['RosTime']-stop_time) >= 0].iloc[0]['PositionForIiwa14']

    
   
    return np.array(object_orient_at_hit), np.array(object_pos_at_hit), np.array(object_pos_final) 

