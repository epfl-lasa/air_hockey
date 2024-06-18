
# importing required modules
from zipfile import ZipFile
import pandas as pd
import os
import ruamel.yaml
import glob

data_folder_name = "CSV"

# Specifying the yml file name
# yml_file_name = "MetaDataRec.yml"
yml_file_name = "MetaDataRecIiwa.yml"

merged_dir = "merged_iiwa"
exp_name = []

# HEADERS IN CSV
header_version = '  '
header_origin_time = 'Origin_time 2022-04-21_15-28-55'
robot = "Panda" # if UR10: Robot_data, if franka panda: Panda, if iiwa: IIWA

def get_new_folder_name(experience_name):
    name = "Rec_"
    date = experience_name.split('Date_')[1]
    date = date.split('.csv')[0]
    name = name + date
    return name


def get_csv_name(experience_name):
    date = experience_name.split('Date_')[1]
    date = date.split('.csv')[0]
    name = f'{robot}_{date}Z.csv'
    return name

def get_box_idx(experience_name):
    box_idx = experience_name.split('box_')[1]
    box_idx = box_idx.split('-task_')[0]
    box_idx = str(int(box_idx) + 30)
    return box_idx

def is_tossing_impact_rs(experience_name):
    task = int((experience_name.split('task_')[1]).split('-speed_')[0])
    is_tossing = task in [1, 2, 3]
    is_impact = task in [1, 2, 4]
    is_rs = task in [1, 4]
    return is_tossing, is_impact, is_rs

def write_yml_file(box_index, final_folder, experience_name):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.explicit_start = False
    yaml.width = 100000

    with open(yml_file_name, 'r') as stream:
        yml_file = yaml.load(stream)
    yml_file['object']['obj_1']['name'] = f'Box0{box_index}'

    is_tossing, is_impact, is_rs = is_tossing_impact_rs(experience_name)
    yml_file['attr']['note'] = f"Coordinated control of dual-arm for object {'swiftly grabbing with impact' if is_impact else 'slowly picking with quasi-static contact'} {'using reference spreading' if is_rs else ''} and {'tossing' if is_tossing else 'placing'} onto a plastic pallet ({experience_name.split('.csv')[0]})"

    yml_file['attr']['transition'] = f"{'impact' if is_impact else 'no-impact'}; {'reference spreading' if is_rs else ''}; {'tossing' if is_tossing else 'placing'}; object; environment"

    new_yml=open(final_folder + "/" + yml_file_name,"w")
    yaml.dump(yml_file, new_yml) #, Dumper=ruamel.yaml.RoundTripDumper, width=10000)
    new_yml.close()

if __name__ == '__main__':
    print("START")

    # Merge folder should exist
    if not os.path.exists(merged_dir):
        raise NameError("Please create a folder named 'merged'")

    # create sessions folder
    session_folder_name = "Session__2024-04-31"
    if not os.path.exists(f'{merged_dir}/{session_folder_name}'):
        os.mkdir(merged_dir + "/"  + session_folder_name)

    file_merged = []

    for ds_filename in glob.iglob(f'{data_folder_name}/DS_*'):
        print("Processing : ", ds_filename)
        experience_name = ds_filename.split("DS_",1)[-1]

        # Read data
        df_ds = pd.read_csv(ds_filename, sep=';')
        print(len(df_ds))
        df_rs = pd.read_csv(f'{data_folder_name}/RS_{experience_name}', sep=';')
        print(len(df_rs))
        
        # Merge RS and DS files
        df_ds_diff = df_ds[list(set(df_ds.columns).difference(set(df_rs.columns)))]
        output1 = pd.merge(df_ds_diff, df_rs, left_index=True, right_index=True, how='outer')

        # session_folder_name = getSessionFolderName(csv_name)
        recording_name = get_new_folder_name(experience_name)
        if not os.path.exists(f'{merged_dir}/{session_folder_name}/{recording_name}'):
            os.mkdir(f'{merged_dir}/{session_folder_name}/{recording_name}')

        # Add CSV
        csv_name = get_csv_name(experience_name)
        output1.to_csv(f'{merged_dir}/{session_folder_name}/{recording_name}/{csv_name}', index=False, sep =' ')
        
        # Add yml file
        box_idx = get_box_idx(experience_name)
        write_yml_file(box_idx, f'{merged_dir}/{session_folder_name}/{recording_name}', experience_name)

        # break
    print("DONE!")

