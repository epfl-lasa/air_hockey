import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PATH_TO_DATA_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/data/"
PATH_TO_PROCESSED_RAW_FOLDER = PATH_TO_DATA_FOLDER + "airhockey_processed/raw/"
    

## PARSING FUNCTIONS
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


## RESAMPLING FUNCTIONS
def resample_uniformally(df, n_samples=300):
    ## resample uniform along HittingFlux 
    df_sorted = df.sort_values(by='HittingFlux')
    interval_edges = np.linspace(df_sorted['HittingFlux'].min(), df_sorted['HittingFlux'].max(), n_samples + 1)
    sampled_indices = []

    # Sample one point from each interval
    for start, end in zip(interval_edges[:-1], interval_edges[1:]):
        # Find the rows that fall within the current interval
        interval_df = df_sorted[(df_sorted['HittingFlux'] >= start) & (df_sorted['HittingFlux'] < end)]
        # Randomly select one row from the interval if it is not empty
        if not interval_df.empty:
            sampled_indices.append(interval_df.sample(n=1).index[0])

    df_sampled = df_sorted.loc[sampled_indices]

    print(f"Resampled df to {len(df_sampled.index)} values.")

    return df_sampled

def restructure_for_agnostic_plots(df1, df2, resample=False, parameter=None, dataset_name=None, save_folder="/", save_new_df=False):
    ## get 300 samples from each df and conglomerate into one

    #### Grab n_samples values from df1 
    n_samples = len(df2.index)
    if resample : 
        df_sampled = resample_uniformally(df1, n_samples=n_samples)
    else:
        df_sampled = df1

    ## Add config to df
    # Add a new column to each DataFrame to indicate the source
    if parameter is not None :
        df_sampled[parameter] = 1
        df2[parameter] = 2

    ## Combine both df 
    df_combined = pd.concat([df_sampled, df2], ignore_index=True)
    df_combined.reset_index(drop=True, inplace=True)

    # Saving clean df
    if save_new_df :

        if "/" not in save_folder:
            save_folder = "/" + save_folder + "/"

        if dataset_name is None:
            dataset_name = parameter+"_agnostic_combined"

        processed_clean_folder = PATH_TO_DATA_FOLDER+"/airhockey_processed/clean" + save_folder
        if not os.path.exists(processed_clean_folder):
            os.makedirs(processed_clean_folder)

        df_combined.to_csv(processed_clean_folder+dataset_name+".csv",index_label="Index")

    return df_combined


## READING AND CLEANING FUNCTIONS
def read_airhockey_csv(fn, folder=PATH_TO_PROCESSED_RAW_FOLDER):
    ## Read airhockey_processed files, by default in folder 'raw/'
    
    if "raw" in folder:
        df = pd.read_csv(folder+fn+".csv", index_col="Index", converters={'ObjectPos' : parse_strip_list, 
            'HittingPos': parse_strip_list, 'ObjectOrientation' : parse_strip_list,'AttractorPos' : parse_strip_list,
            'HittingOrientation': parse_strip_list, 'ObjectPosStart' : parse_strip_list,'ObjectPosEnd' : parse_strip_list})
    
    elif "clean" in folder:
                df = pd.read_csv(folder+fn+".csv", index_col="Index", converters={'ObjectPos' : parse_strip_list_with_commas, 
            'HittingPos': parse_strip_list_with_commas, 'ObjectOrientation' : parse_strip_list_with_commas,
            'AttractorPos' : parse_strip_list_with_commas,'HittingOrientation': parse_strip_list_with_commas, 
            'ObjectPosStart' : parse_strip_list_with_commas,'ObjectPosEnd' : parse_strip_list_with_commas})
    
    print(f"Reading {fn} with {len(df.index)} samples.")

    return df

def read_and_clean_data(csv_fn, dataset_name=None, resample=False, n_samples=2000, only_7=False, distance_threshold=0.1, max_flux=0.8, min_flux=0.5, save_folder="/", save_clean_df=False):
    
    ## Reading data 
    df = read_airhockey_csv(csv_fn)

    ### Remove low outliers -> due to way of recording and processing 
    # Distance
    clean_df = df[df['DistanceTraveled']>distance_threshold]  

    # Robot 
    if only_7 : clean_df = clean_df[clean_df['IiwaNumber']==7]

    # Flux
    clean_df = clean_df[clean_df['HittingFlux']>min_flux]
    clean_df = clean_df[clean_df['HittingFlux']<max_flux]

    # Desired Flux 
    clean_df = clean_df[clean_df['DesiredFlux']>0.5]

    # Resample
    if resample : clean_df = resample_uniformally(clean_df, n_samples=n_samples)

    # Reset index
    clean_df.reset_index(drop=True, inplace=True)       

    print(f"Removed {len(df.index)-len(clean_df.index)} outlier datapoints.")
   
    # Saving clean df
    if save_clean_df :

        if "/" not in save_folder:
            save_folder = "/" + save_folder + "/"
        
        if dataset_name is None:
            dataset_name = csv_fn + "_clean"

        processed_clean_folder = PATH_TO_DATA_FOLDER+"/airhockey_processed/clean" + save_folder
        if not os.path.exists(processed_clean_folder):
            os.makedirs(processed_clean_folder)
            
        clean_df.to_csv(processed_clean_folder+dataset_name+".csv",index_label="Index")
 
    return clean_df


## UTILITY FUNCTIONS
def get_object_based_on_dataset(dataset):
    # get object number based on dataset 
    dataset = dataset.split('_')[0]
    if dataset == 'D1' or dataset == 'D3':
        object_number = 1
    if dataset == 'D2' or dataset == 'D4':
        object_number = 2
    if dataset == 'D5' or dataset == 'D6':
        object_number = 3

    return object_number


## SAVE FUNCTIONS
def save_one_figure(folder_name, title):
    # Specify the directory where you want to save the figures
    save_dir = PATH_TO_DATA_FOLDER + "figures/" + folder_name

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save figure without window borders
    file_name = f"{title}.png"
    plt.gca().set_frame_on(True)  # Turn on frame
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight", pad_inches=0.1)

    print("Figure saved successfully!")

def save_all_figures(folder_name): 

    # Specify the directory where you want to save the figures
    save_dir = PATH_TO_DATA_FOLDER + "figures/" + folder_name

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all figures without window borders
    for fig in plt.get_fignums():
        plt.figure(fig)
        title = plt.gca().get_title()  # Get the title of the figure
        file_name = f"{title}.png" if title else f"figure_{fig}.png"
        plt.gca().set_frame_on(True)  # Turn on frame
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight", pad_inches=0.1)

    print("All figures saved successfully!")
