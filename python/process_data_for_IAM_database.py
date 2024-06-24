
# importing required modules
from zipfile import ZipFile
import pandas as pd
import os
import glob
from utils.data_handling_functions import parse_value
import shutil
from ruamel.yaml import YAML

data_folder_name = "data/varying_flux_datasets"
datasets_list = ['D1', 'D2', 'D3', 'D4']

object_1_idx = 40
object_2_idx = 41

output_dir = "data/IAM_database"

# Specifying the yml filetemplate paths
Rec_yaml_file_template = f"{output_dir}/Session_template/Rec_template/MetaDataRec.yml"
Root_yaml_file_template = f"{output_dir}/Session_template/MetaDataRoot.yml"

def get_new_rec_folder_name(rec_folder):
    name = "Rec_"

    # extract date
    date = rec_folder.split('_',1)[0]
    date = date.replace("-","")

    # extract time
    time = rec_folder.split('_',1)[1]
    time = time.split('_',1)[0]
    time = time.replace(":", "")

    name = name + date + "T" + time + "Z"
    return name

def get_start_rec_time(csv_file):
    # Grab the recorded data
    if "object" in csv_file:
        df = pd.read_csv(csv_file, converters={'RosTime' : parse_value} )
    elif "IIWA" in csv_file:
        df = pd.read_csv(csv_file, skiprows=1, converters={'RosTime' : parse_value})

    # Get the 'Time' column as datetime
    df['RosTime'] = pd.to_datetime(df['RosTime'], unit='s')

    # Ge tthe first timestep in correct format
    start_rec_time = df['RosTime'][0]
    return start_rec_time.strftime('%H%M%S')


def write_Rec_yaml_file(dataset, destination_folder):
    yaml = YAML()
    # Read template file
    with open(Rec_yaml_file_template, 'r') as file:
        yaml_file = yaml.load(file)

    # get date-time
    datetime = destination_folder.split("Rec_")[1]

    # Change params to match recording
    if dataset == "D1":
        yaml_file['attr']['note'] = f"Back and forth hitting of an object with dual-arm setup (D1:box_1-config_1-Date_{datetime}))"
        yaml_file['object']['obj_1']['name'] = f'Box0{object_1_idx}'

    elif dataset == "D2":
        yaml_file['attr']['note'] = f"Back and forth hitting of an object with dual-arm setup (D2:box_2-config_1-Date_{datetime}))"
        yaml_file['object']['obj_1']['name'] = f'Box0{object_2_idx}'

    elif dataset == "D3":
        yaml_file['attr']['note'] = f"Back and forth hitting of an object with dual-arm setup (D3:box_1-config_2-Date_{datetime}))"
        yaml_file['object']['obj_1']['name'] = f'Box0{object_1_idx}'

    elif dataset == "D4":
        yaml_file['attr']['note'] = f"Back and forth hitting of an object with dual-arm setup (D4:box_2-config_2-Date_{datetime}))"
        yaml_file['object']['obj_1']['name'] = f'Box0{object_2_idx}'

    # Write modifications to corect folder
    destination_fn = os.path.join(destination_folder, "MetaDataRec.yml")
    with open(destination_fn, 'w') as file:
        yaml.dump(yaml_file, file)

    print(f"Updated YAML file written to: {destination_fn}")

def write_Root_yaml_file(destination_path):
    yaml = YAML()
    # Read template file
    with open(Root_yaml_file_template, 'r') as file:
        yaml_file = yaml.load(file)

    # Write modifications to corect folder
    with open(destination_path, 'w') as file:
        yaml.dump(yaml_file, file)

    print(f"Updated YAML file written to: {destination_path}")

# Process : 
# take each rec folder, move to i_am_database/session, rename to rec format 
# add .yml file to each session and rec folder


if __name__ == '__main__':
    print("START")

    # Merge folder should exist
    if not os.path.exists(output_dir):
        raise NameError("Please create a folder for IAM database")
    
    for dataset in datasets_list:
        for rec_folder in glob.iglob(f'{data_folder_name}/{dataset}/*'):
            print("Processing : ", rec_folder) ## rec_folder is complete path 
            rec_folder_dir_name = rec_folder.split("/")[3] ## get only directory name 

            # Create session folder
            session_date = rec_folder_dir_name.split("_", 1)[0]
            session_folder_name = "Session_"+session_date
            session_folder = f'{output_dir}/{session_folder_name}'
            if not os.path.exists(session_folder):
                os.mkdir(session_folder)

            # Copy Root yaml file there
            root_yaml_path = os.path.join(session_folder, "MetaDataRoot.yml")
            if not os.path.exists(root_yaml_path):
                write_Root_yaml_file(destination_path=root_yaml_path)

            # Create new rec folder
            new_rec_folder_dir_name = get_new_rec_folder_name(rec_folder_dir_name)
            new_rec_folder = os.path.join(output_dir, session_folder_name, new_rec_folder_dir_name)
            if not os.path.exists(new_rec_folder):
                os.mkdir(new_rec_folder)
            
            # Copy Rec yaml file
            write_Rec_yaml_file(dataset=dataset, destination_folder=new_rec_folder)

            # For each file in rec Session, copy and rename by adding 
            for filename in os.listdir(rec_folder):
                # Check if the file is a CSV file
                if filename.endswith(".csv") and "manually" not in filename:
                    # Construct the full file path
                    source_file = os.path.join(f'{rec_folder}', filename)
                    
                    # Construct the new file name
                    base_name = os.path.splitext(filename)[0]
                    time_rec_start = get_start_rec_time(source_file)
                    date_rec_start = session_date.replace("-", "")

                    new_filename = f"{base_name}_{date_rec_start}T{time_rec_start}Z.csv"
                    dest_file = os.path.join(new_rec_folder, new_filename)
                    
                    # Copy the file to the destination directory with the new name
                    shutil.copy(source_file, dest_file)

            # break
    print("DONE!")

