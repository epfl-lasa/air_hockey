import os
import pandas as pd
import numpy as np
import time
import re
from python.utils.data_handling_functions import PATH_TO_DATA_FOLDER, parse_list
from python.utils.data_processing_functions import *

## Get hit info and format it for csv
def get_info_at_hit_time(robot_csv, object_csv):

    # get hit time 
    hit_time, stop_time = get_impact_time_from_object(object_csv)

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
    
    # get object info at hit
    object_orient, object_pos_start, object_pos_end = get_object_info_during_hit(object_csv, hit_time, stop_time, iiwa_number)

    # get desired flux from top row 
    des_flux = pd.read_csv(robot_csv, nrows=1, header=None).iloc[0].to_list()[1]
    des_pos = np.array(parse_list(pd.read_csv(robot_csv, nrows=1, header=None).iloc[0].to_list()[5]))

    # Should be ordered in the same way as output file columns
    return [recording_session, hit_number, iiwa_number, des_flux, hitting_flux,  distance_travelled,  hitting_dir_inertia, des_pos, hitting_pos, object_orient, hitting_orientation, object_pos_start, object_pos_end]

## For each hit, get hit info and write to a csv file
def process_data_to_one_file(data_folder, recording_sessions, output_filename="test.csv"):
    
    path_to_data_airhockey = PATH_TO_DATA_FOLDER + data_folder +"/"
    output_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/airhockey_processed/raw/" + output_filename

    output_df = pd.DataFrame(columns=["RecSession","HitNumber","IiwaNumber","DesiredFlux","HittingFlux","DistanceTraveled","HittingInertia","AttractorPos", "HittingPos", "ObjectOrientation", "HittingOrientation","ObjectPosStart","ObjectPosEnd"])

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
    

if __name__== "__main__" :
    
    ### VARIABLES FOR DATA PROCESING 
    data_folder = "varying_flux_datasets/D3"  ## where to get raw data
    output_dataset_fn = "D3_clean"  ## name of output csv file
    surface = "clean"   ## keyword to chose data subfoldes

    ## Grab the correct subfolders in dataset
    folders_to_process = os.listdir(PATH_TO_DATA_FOLDER + data_folder)
    to_process = []
    for folder in folders_to_process :
        if surface in folder: 
            to_process.append(folder)

    ## Process all folders in the desired data_folder
    process_data_to_one_file(data_folder, to_process, output_filename=f"{output_dataset_fn}.csv")
