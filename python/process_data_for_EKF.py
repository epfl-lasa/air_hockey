import os
import pandas as pd
import time
import re
from python.utils.data_handling_functions import PATH_TO_DATA_FOLDER, parse_list, parse_value
from python.utils.data_processing_functions import get_impact_time_from_object, get_robot_data_at_hit

## EKF processing
def process_one_file_for_ekf(robot_csv, object_csv, output_folder, pos_name_str = 'PositionWorldFrame'):
    
    #read files
    df_obj = pd.read_csv(object_csv, skiprows=1, converters={'RosTime' : parse_value, pos_name_str: parse_list, 'PositionForIiwa7': parse_list}) #'Position': parse_list, 'PositionForIiwa14': parse_list})
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
    merged_df = merged_df[['RosTime', pos_name_str, 'EEF_Position', 'HittingFlux']]
    
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
    
    # Reformat for James' EKF -> WATCH OUT INVERTING X AND Y HERE (due to x-axis in optitrack frame == y-axis in robot frame)
    formatted_data = merged_df.apply(lambda row : [row['RosTime'], row[pos_name_str][2],row[pos_name_str][1],row[pos_name_str][0], 
                                                    row['EEF_Position'][2],row['EEF_Position'][0],row['EEF_Position'][1], row['HittingFlux']], axis=1)
    
    ## FOR MARCH DATA 
    # formatted_data = merged_df.apply(lambda row : [row['RosTime'], row[pos_name_str][2],row[pos_name_str][1],row[pos_name_str][0], 
    #                                                 row['EEF_Position'][2],row['EEF_Position'][0],row['EEF_Position'][1], row['HittingFlux']], axis=1)

    formatted_df = pd.DataFrame(formatted_data.tolist(), columns=['time', 'z_o', 'y_o', 'x_o', 'z_eef', 'y_eef', 'x_eef', 'flux'])

    ## OFFSET x due to world frame ! --> removes jumps in data
    x_offset = df_obj['PositionForIiwa7'].iloc[0][1]- df_obj['PositionWorldFrame'].iloc[0][0] # x offset from world frame to robot frame
    formatted_df['x_o'] = formatted_df['x_o'].apply(lambda x: x+x_offset)

    ## FOR MARCH DATA 
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
                hit_time, stop_time = get_impact_time_from_object(object_csv)
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
   
    # Setting up datasets
    data_folder = "varying_flux_datasets/D1_clean"

    # Process all folders in the desired data_folder
    folders_to_process = os.listdir(PATH_TO_DATA_FOLDER + data_folder)
    
    ### EKF Processing
    ## Process a few files 
    path_to_data_airhockey = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey/"
    path_to_data_ekf = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey_ekf/june/clean"

    for hit_number in range (3,9):
        robot_csv = os.path.join(path_to_data_airhockey,"2024-06-11_14:58:06", f"IIWA_7_hit_{hit_number}.csv" )
        object_csv = os.path.join(path_to_data_airhockey,"2024-06-11_14:58:06", f"object_1_hit_{hit_number}.csv" )
        process_one_file_for_ekf(robot_csv, object_csv, path_to_data_ekf, pos_name_str='PositionWorldFrame')    
    
    ## Process everything
    # process_all_data_for_ekf(folders_to_process)