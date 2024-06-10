import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
import mplcursors
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import pybullet 
import math
from scipy.spatial.transform import Rotation

import sys
sys.path.append('/home/maxime/Workspace/air_hockey/python_data_processing/gmm_torch')
# from gmm_torch.gmm import GaussianMixture
# from gmm_torch.example import plot
# import torch

from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses

from process_data import (parse_value, parse_list, parse_strip_list, parse_strip_list_with_commas, get_impact_time_from_object, 
                          get_corrected_quat_object_2, get_orientation_error_x_y_z, get_orientation_error_manually, 
                          get_orientation_error_in_correct_base, PATH_TO_DATA_FOLDER)

# Fontsize for axes and titles 
GLOBAL_FONTSIZE = 30
AXIS_TICK_FONTSIZE = 20

PROCESSED_RAW_FOLDER = PATH_TO_DATA_FOLDER + "airhockey_processed/raw/"
    

# PROCESSING
def wrap_angle(angle_rad):
    # wraps an angle
    angle_deg = math.degrees(angle_rad)
    mod_angle = (angle_deg) % 180
    if angle_deg >= 180 : return 180-mod_angle
    if angle_deg <= -180: return -mod_angle
    else: return angle_deg

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

def read_airhockey_csv(fn, folder=PROCESSED_RAW_FOLDER):
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

# CLEANING FUNCTION
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

# PLOTTING FUNCTIONS
## GMM ANALYISIS -- DEPRECATED
def test_gmm_torch(df): 

    temp_array = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    data = torch.tensor(temp_array, dtype=torch.float32) 

    # Next, the Gaussian mixture is instantiated and ..
    model = GaussianMixture(n_components=3, n_features=2, covariance_type="full")
    model.fit(data,delta=1e-7, n_iter=1000)
    # .. used to predict the data points as they where shifted
    # y = model.predict(data)

    # flux_test = torch.tensor(np.linspace(0.4,1.2,100).reshape(-1,1), dtype=torch.float32)
    # TODO : do bic test on predicted results ???
    # print(model.bic(flux_test))

    plot_distance_vs_flux(df, with_linear_regression=True, gmm_model=model, use_mplcursors=False)
    # plot(data, y)

## DATA PLOTS
def plot_distance_vs_flux(df, colors="iiwa", with_linear_regression=True, gmm_model=None, use_mplcursors=True, show_plot=False):
    ## use colors input to dtermine color of datapoints

    # Plot Flux
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), sharex=True)

    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    ## Plot with different colors 
    if colors == "iiwa":
        ax.scatter(df_iiwa7['HittingFlux'], df_iiwa7['DistanceTraveled'], color='red', alpha=0.5, label='IIWA 7')
        ax.scatter(df_iiwa14['HittingFlux'], df_iiwa14['DistanceTraveled'], color='blue', alpha=0.5, label='IIWA 14')

    elif colors == "orientation":
        ###TODO : check this is correct ?? 
        df["OrientationError"] = df.apply(lambda row : np.linalg.norm(np.array(row["HittingOrientation"])-np.array(row["ObjectOrientation"])),axis=1)
        scatter = ax.scatter(df['HittingFlux'], df['DistanceTraveled'], c=df['OrientationError'], cmap="viridis")
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Orientation error')

    elif colors == "config":
        df_cfg1 = df[df['config']==1].copy()
        df_cfg2 = df[df['config']==2].copy()
        # df_cfg1 = df[df['RecSession']=="2024-05-08_16:27:34"].copy()
        # df_cfg2 = df[df['RecSession']=="2024-05-16_13:49:02"].copy()
        ax.scatter(df_cfg1['HittingFlux'], df_cfg1['DistanceTraveled'], color='green', alpha=0.5, label='Config 1')
        ax.scatter(df_cfg2['HittingFlux'], df_cfg2['DistanceTraveled'], color='orange', alpha=0.5, label='Config 2')

    elif colors == "object":
        df_obj1 = df[df['object']==1].copy()
        df_obj2 = df[df['object']==2].copy()
        ax.scatter(df_obj1['HittingFlux'], df_obj1['DistanceTraveled'], color='purple', alpha=0.5, label='Object 1')
        ax.scatter(df_obj2['HittingFlux'], df_obj2['DistanceTraveled'], color='orange', alpha=0.5, label='Object 2')

    ## Add linear regression
    if with_linear_regression: 
        if colors == "iiwa":
            lin_model = LinearRegression()
            lin_model.fit(df['HittingFlux'].values.reshape(-1,1), df['DistanceTraveled'].values)

            flux_test = np.linspace(df['HittingFlux'].min(), df['HittingFlux'].max(),100).reshape(-1,1)
            distance_pred = lin_model.predict(flux_test)
            ax.plot(flux_test,distance_pred,color='black', label='Linear Regression')
        
        elif colors =="config":
            lin_model = LinearRegression()
            lin_model.fit(df_cfg1['HittingFlux'].values.reshape(-1,1), df_cfg1['DistanceTraveled'].values)

            flux_test = np.linspace(df_cfg1['HittingFlux'].min(), df_cfg1['HittingFlux'].max(),100).reshape(-1,1)
            distance_pred = lin_model.predict(flux_test)
            ax.plot(flux_test,distance_pred,color='green', label='Linear Regression Config 1')

            lin_model2 = LinearRegression()
            lin_model2.fit(df_cfg2['HittingFlux'].values.reshape(-1,1), df_cfg2['DistanceTraveled'].values)

            flux_test2 = np.linspace(df_cfg2['HittingFlux'].min(), df_cfg2['HittingFlux'].max(),100).reshape(-1,1)
            distance_pred2 = lin_model2.predict(flux_test2)
            ax.plot(flux_test2,distance_pred2,color='orange', label='Linear Regression Config 2')

        elif colors =="object":
            lin_model = LinearRegression()
            lin_model.fit(df_obj1['HittingFlux'].values.reshape(-1,1), df_obj1['DistanceTraveled'].values)

            flux_test = np.linspace(df_obj1['HittingFlux'].min(), df_obj1['HittingFlux'].max(),100).reshape(-1,1)
            distance_pred = lin_model.predict(flux_test)
            ax.plot(flux_test,distance_pred,color='purple', label='Linear Regression Object 1')

            lin_model2 = LinearRegression()
            lin_model2.fit(df_obj2['HittingFlux'].values.reshape(-1,1), df_obj2['DistanceTraveled'].values)

            flux_test2 = np.linspace(df_obj2['HittingFlux'].min(), df_obj2['HittingFlux'].max(),100).reshape(-1,1)
            distance_pred2 = lin_model2.predict(flux_test2)
            ax.plot(flux_test2,distance_pred2,color='orange', label='Linear Regression Object 2')


    ## Add GMM model
    if gmm_model is not None:
        for i in range(gmm_model.n_components):
            # Plot center of each component
            center = gmm_model.mu[0,i,:]
            ax.plot(center[0], center[1], 'ro', label=f'Center {i+1}')

            # Plot ellipse representing covariance matrix
            cov_matrix = gmm_model.var[0,i,:,:] 
            lambda_, v = np.linalg.eig(cov_matrix)
            lambda_ = np.sqrt(lambda_)
            ellipse = plt.matplotlib.patches.Ellipse(xy=center, width=lambda_[0]*2, height=lambda_[1]*2,
                                                    angle=np.rad2deg(np.arccos(v[0, 0])), color='grey',alpha=0.4)
            plt.gca().add_patch(ellipse)

    # Print some infos
    print(f"Dataset info : \n"
          f" Iiwa 7 points : {len(df_iiwa7.index)} \n"
          f" Iiwa 14 points : {len(df_iiwa14.index)} \n"
          f" Total points : {len(df.index)}")
    
    if colors == "object":
        print(f"Dataset info : \n"
          f" Object 1 points : {len(df_obj1.index)} \n"
          f" Object 2 points : {len(df_obj2.index)}")
    if colors == "config":
        print(f"Dataset info : \n"
          f" Config 1 points : {len(df_cfg1.index)} \n"
          f" config 2 points : {len(df_cfg2.index)}")
        
    # Adding info when hovering cursor
    if use_mplcursors:
        mplcursors.cursor(hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df['RecSession'][sel.index]}, hit #{df['HitNumber'][sel.index]}, iiwa{df['IiwaNumber'][sel.index]}"))   

    ax.set_xlabel('Hitting flux [m/s]',fontsize=GLOBAL_FONTSIZE)
    ax.set_ylabel('Distance Traveled [m]',fontsize=GLOBAL_FONTSIZE)
    ax.grid(True, alpha=0.5)
    plt.legend(fontsize=15)
    plt.title(f"Distance over Flux",fontsize=GLOBAL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE) 
    fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    if show_plot : plt.show()

def flux_hashtable(df, use_mplcursors=True, show_plot = False):
    # Plot Flux
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), sharex=True)

    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()
    ax.scatter(df_iiwa7['DesiredFlux'], df_iiwa7['HittingFlux'], color='red', alpha=0.5, label='Iiwa 7')
    ax.scatter(df_iiwa14['DesiredFlux'], df_iiwa14['HittingFlux'], color='blue', alpha=0.5, label='Iiwa 14')

    # add line
    diagonal = np.linspace(df_iiwa7['DesiredFlux'].min(), df_iiwa7['DesiredFlux'].max(), 100)
    ax.plot(diagonal, diagonal, color='black')

    # Adding info when hovering cursor
    if use_mplcursors:
        mplcursors.cursor(hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df['RecSession'][sel.index]}, hit #{df['HitNumber'][sel.index]}, iiwa{df['IiwaNumber'][sel.index]}"))   

    ax.set_xlabel('Desired flux [m/s]',fontsize=GLOBAL_FONTSIZE)
    ax.set_ylabel('Hitting Flux [m/s]',fontsize=GLOBAL_FONTSIZE)
    ax.grid(True)
    plt.legend()
    plt.title(f"Flux Hashtable",fontsize=GLOBAL_FONTSIZE)
    fig.tight_layout(rect=(0.01,0.01,0.99,0.99))
    
    if show_plot : plt.show()


def plot_hit_position(df, plot="on object", use_mplcursors=True, show_plot = False):
    # plot choices : "on object" , "flux", "distance"

    # for each hit, get relative error in x,y,z 
    df["RelPosError"]=df.apply(lambda row : [a-b for a,b in zip(row["HittingPos"],row["ObjectPos"])], axis=1)
    df["RelPosError"] = df["RelPosError"].apply(lambda list : [x*100 for x in list]) # Read as cm

    # remove datapoint soutside of box 
    df = df[~df['RelPosError'].apply(lambda list : any(abs(x)>10 for x in [list[0],list[2]]))]
    
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    # Show on object 
    if plot == "on object":
        # plot x,z with y as color 
        
        fig_iiwa7 = plt.figure()
        # scatter = plt.scatter(df_iiwa7["RelPosError"].apply(lambda x: x[0]),df_iiwa7["RelPosError"].apply(lambda x: x[2]), c=df_iiwa7["RelPosError"].apply(lambda x: abs(x[1])), cmap='viridis')
        scatter = plt.scatter(df_iiwa7["RelPosError"].apply(lambda x: x[0]),df_iiwa7["RelPosError"].apply(lambda x: x[2]), c=df_iiwa7["HittingFlux"], cmap='viridis')
        plt.scatter(0,0, c="red", marker="x")
        
        # Add box - TODO : define it better
        rect = Rectangle((-12.5, -11.5), 25, 23, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        cbar = plt.colorbar(scatter)
        # cbar.set_label('Y-axis [cm]')
        cbar.set_label('Flux')
        plt.xlabel('X-axis [cm]')
        plt.ylabel('Z-Axis[cm]')
        plt.title('Hitting Point shown on object - IIWA 7')

        ###SECOND FIG FOR IIWA 14
        fig_iiwa14 = plt.figure()
        # plot x,z with y as color 
        # scatter = plt.scatter(df_iiwa14["RelPosError"].apply(lambda x: x[0]),df_iiwa14["RelPosError"].apply(lambda x: x[2]), c=df_iiwa14["RelPosError"].apply(lambda x: abs(x[1])), cmap='viridis')
        scatter = plt.scatter(df_iiwa14["RelPosError"].apply(lambda x: x[0]),df_iiwa14["RelPosError"].apply(lambda x: x[2]), c=df_iiwa14["HittingFlux"], cmap='viridis')
        plt.scatter(0,0, c="red", marker="x")
        
        # Add box - TODO : define it better
        rect = Rectangle((-12.5, -11.5), 25, 23, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        cbar = plt.colorbar(scatter)
        # cbar.set_label('Y-axis [cm]')
        cbar.set_label('Flux')
        plt.xlabel('X-axis [cm]')
        plt.ylabel('Z-Axis[cm]')
        plt.title('Hitting Point shown on object - IIWA 14')

    # Show over flux 
    elif plot == "flux":
        plt.scatter(df_iiwa7['HittingFlux'],df_iiwa7["RelPosError"].apply(lambda x: np.linalg.norm(x)), color="red", alpha=0.5, label='Iiwa 7')
        plt.scatter(df_iiwa14['HittingFlux'],df_iiwa14["RelPosError"].apply(lambda x: np.linalg.norm(x)),color="blue", alpha=0.5, label='Iiwa 14')
        plt.legend()
        plt.xlabel('Hitting Flux [m/s]')
        plt.ylabel('Normed Position error [cm]')
        plt.title('Hitting Position error over Flux')

    elif plot == "distance":
        plt.scatter(df_iiwa7['DistanceTraveled'],df_iiwa7["RelPosError"].apply(lambda x: np.linalg.norm(x)),color="red", alpha=0.5, label='Iiwa 7')
        plt.scatter(df_iiwa14['DistanceTraveled'],df_iiwa14["RelPosError"].apply(lambda x: np.linalg.norm(x)), color="blue", alpha=0.5, label='Iiwa 14')
        plt.legend()
        plt.ylabel('Normed Position error [cm]')
        plt.xlabel('Distance travelled [m]')
        plt.title('Position error at impact over distance traveled')

    # Adding info when hovering cursor
    if use_mplcursors:
        if plot == "on object":
            mplcursors.cursor([fig_iiwa7],hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df_iiwa7['RecSession'][sel.index]}, hit #{df_iiwa7['HitNumber'][sel.index]}, iiwa{df_iiwa7['IiwaNumber'][sel.index]}")) 
            mplcursors.cursor([fig_iiwa14],hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df_iiwa14['RecSession'][sel.index]}, hit #{df_iiwa14['HitNumber'][sel.index]}, iiwa{df_iiwa14['IiwaNumber'][sel.index]}")) 
        else:
            mplcursors.cursor(hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df['RecSession'][sel.index]}, hit #{df['HitNumber'][sel.index]}, iiwa{df['IiwaNumber'][sel.index]}"))   

    if show_plot : plt.show()

def plot_orientation_vs_distance(df, axis="z", use_mplcursors=True, show_plot = False):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    ## calculate quaternion diff
    
    df["OrientationError"] = df.apply(lambda row : pybullet.getEulerFromQuaternion(pybullet.getDifferenceQuaternion(row["ObjectOrientation"],row["HittingOrientation"])),axis=1)
    df["OrientationError"] = df["OrientationError"].apply(lambda x : [wrap_angle(i) for i in x]).copy()

    df["OrientationError2"] = df.apply(lambda row : [pybullet.getEulerFromQuaternion(row["HittingOrientation"])[i]- pybullet.getEulerFromQuaternion(row["ObjectOrientation"])[i] for i in range(3)],axis=1).copy()
    df["OrientationError3"] = df.apply(lambda row : get_orientation_error_x_y_z(row["ObjectOrientation"],row["HittingOrientation"]),axis=1).copy()
    
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    if axis == "x":
        # scatter = axes.scatter(df['DistanceTraveled'], df['OrientationError'].apply(lambda x : x[0]), c=df['HittingFlux'], cmap="viridis")
        # axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa7['DistanceTraveled'], cmap="viridis")
        # scatter = axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa14['DistanceTraveled'], cmap="viridis")
        
        scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError'].apply(lambda x : x[0]), color="red", label="Quat error")
        axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError'].apply(lambda x : x[0]), color="red", label="Quat error")
        
        scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError2'].apply(lambda x : x[0]), color="blue", label="Euler error")
        axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError2'].apply(lambda x : x[0]), color="blue", label="Euler error")
        
        # cbar = plt.colorbar(scatter)
        # cbar.set_label('Hitting Flux [m/s]')
        axes[0].set_ylabel('Orientation in X-axis [rad]')

    if axis == "y":
        scatter = axes.scatter(df['DistanceTraveled'], df['OrientationError'].apply(lambda x : x[1]), c=df['HittingFlux'], cmap="viridis")
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Hitting Flux [m/s]')
        axes.set_ylabel('Orientation Error in Y-axis [rad]')

    if axis == "z":
        # scatter = ax.scatter(df['DistanceTraveled'], df['OrientationError'].apply(lambda x : x[2]), c=df['HittingFlux'], cmap="viridis")
        # scatter = axes[0].scatter(df_iiwa7['DistanceTraveled'], df_iiwa7['OrientationError'].apply(lambda x : x[2]), c=df_iiwa7['HittingFlux'], cmap="viridis")
        # axes[1].scatter(df_iiwa14['DistanceTraveled'], df_iiwa14['OrientationError'].apply(lambda x : x[2]), c=df_iiwa14['HittingFlux'], cmap="viridis")

        # scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError'].apply(lambda x : x[2]), c=df_iiwa7['DistanceTraveled'], cmap="viridis")
        # axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError'].apply(lambda x : x[2]), c=df_iiwa14['DistanceTraveled'], cmap="viridis")

        scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError'].apply(lambda x : x[2]), color="red", label="Quat error")
        axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError'].apply(lambda x : x[2]), color="red", label="Quat error")
        
        scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError2'].apply(lambda x : x[2]), color="blue", label="Euler error")
        axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError2'].apply(lambda x : x[2]), color="blue", label="Euler error")

        scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError3'].apply(lambda x : x[2]), color="green", label="Rot error")
        axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError3'].apply(lambda x : x[2]), color="green", label="Rot error")

        # scatter = axes[0].scatter(df_iiwa7['DistanceTraveled'], df_iiwa7['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa7['HittingFlux'], cmap="viridis")
        # axes[1].scatter(df_iiwa14['DistanceTraveled'], df_iiwa14['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa14['HittingFlux'], cmap="viridis")

        # scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['ObjectOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[1]), c=df_iiwa7['DistanceTraveled'], cmap="viridis")
        # axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['ObjectOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[1]), c=df_iiwa14['DistanceTraveled'], cmap="viridis")


        # cbar = plt.colorbar(scatter, ax=axes.ravel().tolist())
        # # cbar.set_label('Hitting Flux [m/s]')
        # cbar.set_label('Distance traveled [m]')

        axes[0].set_ylabel('Orientation Error in Z-axis [rad]')
    


    # Adding info when hovering cursor
    if use_mplcursors:
        mplcursors.cursor(hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df['RecSession'][sel.index]}, hit #{df['HitNumber'][sel.index]}, iiwa{df['IiwaNumber'][sel.index]}"))   

    axes[0].set_title("IIWA 7")
    axes[1].set_title("IIWA 14")

    for ax in axes : 
        # ax.set_xlabel('Distance Travelled [m]')
        ax.set_xlabel('Hitting Flux [m/s]')
        ax.grid(True)

    plt.legend()
    plt.title(f"Orientation over Flux")
    # fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    if show_plot : plt.show()

def plot_object_trajectory(df, use_mplcursors=True, selection="all", show_plot = False):
   
    fig_iiwa7 ,ax_iiwa7 = plt.subplots()
    fig_iiwa14, ax_iiwa14 = plt.subplots()

    # Random selection
    if selection == "all":
        pass
    elif selection == "random":
        nb_samples = 5

        temp_df = pd.DataFrame()

        for iiwa in [7,14]:
            high_flux_df = df[(df['HittingFlux'] >= 0.90) & (df['IiwaNumber'] == iiwa)].sample(n=nb_samples)
            med_flux_df = df[(df['HittingFlux'] >= 0.8) & (df['HittingFlux'] <=0.9) & (df['IiwaNumber'] == iiwa)].sample(n=nb_samples)
            low_flux_df = df[(df['HittingFlux'] <=0.8) & (df['IiwaNumber'] == iiwa)].sample(n=nb_samples)

            temp_df = pd.concat([temp_df, high_flux_df, med_flux_df, low_flux_df])

        df = temp_df

    elif selection == "selected" :
        # hand selected trajectories to plot 
        idx_to_plot = [4,21,33,67,69,81,105,115,122,130,169,171,306,357,649,652,655,669,701,755,788,837,845,861,870]
        df = df[df.index.isin(idx_to_plot)]

    # get wrapped orientation error
    df["OrientationError"] = df.apply(lambda row : pybullet.getEulerFromQuaternion(pybullet.getDifferenceQuaternion(row["ObjectOrientation"],row["HittingOrientation"])),axis=1).copy()
    df["OrientationError"] = df["OrientationError"].apply(lambda x : [wrap_angle(i) for i in x]).copy()
    
    df["OrientationError2"] = df.apply(lambda row : [pybullet.getEulerFromQuaternion(row["HittingOrientation"])[i]- pybullet.getEulerFromQuaternion(row["ObjectOrientation"])[i] for i in range(3)],axis=1).copy()
    df["OrientationError3"] = df.apply(lambda row : [get_orientation_error_x_y_z(row["HittingOrientation"],row["ObjectOrientation"])],axis=1).copy()

    
    start_pos0 = [] # list for color
    start_pos1 = []

    for index,row in df.iterrows():

        # get object trajectory from file name
        data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ "/data/airhockey/"
        if os.name == "nt": # Windows OS
            rec_sess = row["RecSession"].replace(":","_")
        else : rec_sess = row["RecSession"]
        
        obj_fn = data_folder + rec_sess + f"/object_2_hit_{row['HitNumber']}.csv"
        
        df_obj = pd.read_csv(obj_fn, converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})
        # df_obj = pd.read_csv(obj_fn, converters={'RosTime' : parse_value, 'Position': parse_list})

        if row['IiwaNumber']==7 :
    
            ax_iiwa7.plot(df_obj['Position'].apply(lambda x: -x[0]), df_obj['Position'].apply(lambda x: x[1]), alpha=0.8)
            scatter_iiwa7 = ax_iiwa7.scatter(-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1], label=f"IDX:{index}")
            start_pos0.append([-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1]])
            
            ax_iiwa7.set_xlabel('Y Axis [m]')
            ax_iiwa7.set_ylabel('X Axis [m]')
            ax_iiwa7.grid(True)
            ax_iiwa7.set_aspect('equal')
            ax_iiwa7.set_title("Object trajectories for IIWA 7")
            
            # Adding info when clicking cursor
            if use_mplcursors:
                mplcursors.cursor(scatter_iiwa7, hover=False).connect('add', lambda sel: sel.annotation.set_text(sel.artist.get_label()))   

        if row['IiwaNumber']==14 :
            
            # inverted to adpat to optitrack reference frame
            ax_iiwa14.plot(df_obj['Position'].apply(lambda x: -x[0]), df_obj['Position'].apply(lambda x: x[1]), alpha=0.8)
            scatter_iiwa14 = ax_iiwa14.scatter(-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1], label=f"IDX:{index}")
            start_pos1.append([-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1]])

            
            ax_iiwa14.set_xlabel('Y Axis [m]')
            ax_iiwa14.set_ylabel('X Axis [m]')
            ax_iiwa14.grid(True)
            ax_iiwa14.set_aspect('equal')
            ax_iiwa14.set_title("Object trajectories for IIWA 14")
            
            # Adding info when clicking cursor
            if use_mplcursors:
                mplcursors.cursor(scatter_iiwa14, hover=False).connect('add', lambda sel: sel.annotation.set_text(sel.artist.get_label()))   

    # add colors
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()
    scatter_iiwa7 = ax_iiwa7.scatter(np.array(start_pos0)[:,0],np.array(start_pos0)[:,1], c=df_iiwa7['OrientationError2'].apply(lambda x : x[2]), cmap="viridis")
    scatter_iiwa14 = ax_iiwa14.scatter(np.array(start_pos1)[:,0],np.array(start_pos1)[:,1], c=df_iiwa14['OrientationError2'].apply(lambda x : x[2]), cmap="viridis")
    cbar = plt.colorbar(scatter_iiwa7)
    cbar.set_label('Orientation Error in Z-axis [deg]')
    
    cbar2 = plt.colorbar(scatter_iiwa14)
    cbar2.set_label('Orientation Error in Z-axis [deg]')
            
    # move bottom ax for pretty -> DOESN'T WORK
    # ax_pos = axes[1].get_position()
    # print(ax_pos)
    # fig_width, fig_height = fig.get_size_inches()
    # new_left = ax_pos.x0 - 0.29
    # ax_pos.x0 = new_left
    # print(ax_pos)
    # axes[1].set_position(ax_pos)
    
    # leg = fig.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))
    # leg.set_draggable(state=True)
    plt.title(f"Object trajectories")
    # fig.tight_layout(rect=(0.01,0.01,0.99,0.99))
    
    if show_plot : plt.show()


def plot_object_trajectory_onefig(df, dataset_path="varying_flux_datasets/D1/", use_mplcursors=False, selection="all", show_plot = False):

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Random selection
    if selection == "all":
        pass
    elif selection == "random":
        nb_samples = 5

        temp_df = pd.DataFrame()

        for iiwa in [7,14]:
            high_flux_df = df[(df['HittingFlux'] >= 0.90) & (df['IiwaNumber'] == iiwa)].sample(n=nb_samples)
            med_flux_df = df[(df['HittingFlux'] >= 0.8) & (df['HittingFlux'] <=0.9) & (df['IiwaNumber'] == iiwa)].sample(n=nb_samples)
            low_flux_df = df[(df['HittingFlux'] <=0.8) & (df['IiwaNumber'] == iiwa)].sample(n=nb_samples)

            temp_df = pd.concat([temp_df, high_flux_df, med_flux_df, low_flux_df])

        df = temp_df

    elif selection == "selected" :
        # hand selected trajectories to plot 
        idx_to_plot = [4,7,33,63,67,69,84,97,115,130,131,132,138,649,701,712,870,891,899,900,1019,1174]
        # idx_to_plot = [4,21,33,67,69,81,105,115,122,130,169,171,306,357,649,652,655,669,701,755,788,837,845,861,870]
        df = df[df.index.isin(idx_to_plot)]

    start_pos0 = [] # list for color
    start_pos1 = []

    # get object number based on dataset path
    dataset = dataset_path.split('/')[1]
    object_number = get_object_based_on_dataset(dataset)

    for index,row in df.iterrows():

        # get object trajectory from file name
        data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ "/data/airhockey/"
        if os.name == "nt": # Windows OS
            rec_sess = row["RecSession"].replace(":","_")
        else : rec_sess = row["RecSession"]
        
        obj_fn = PATH_TO_DATA_FOLDER + dataset_path + rec_sess + f"/object_{object_number}_hit_{row['HitNumber']}.csv"
        
        df_obj = pd.read_csv(obj_fn, converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})

        if row['IiwaNumber']==7 :
            
            axes[0].plot(df_obj['PositionForIiwa7'].apply(lambda x: x[1]), df_obj['PositionForIiwa7'].apply(lambda x: x[0]), alpha=0.8)
            scatter_iiwa7 = axes[0].scatter(df_obj['PositionForIiwa7'].iloc[0][1], df_obj['PositionForIiwa7'].iloc[0][0], label=f"idx:{index}")
            start_pos0.append([df_obj['PositionForIiwa7'].iloc[0][1], df_obj['PositionForIiwa7'].iloc[0][0]])

            axes[0].set_title("IIWA 7", fontsize=GLOBAL_FONTSIZE)
            
            # Adding info when clicking cursor
            if use_mplcursors:
                mplcursors.cursor(scatter_iiwa7, hover=False).connect('add', lambda sel: sel.annotation.set_text(sel.artist.get_label()))   

        if row['IiwaNumber']==14 :
            
            axes[1].plot(df_obj['PositionForIiwa14'].apply(lambda x: x[1]), df_obj['PositionForIiwa14'].apply(lambda x: x[0]))
            scatter_iiwa14 = axes[1].scatter(df_obj['PositionForIiwa14'].iloc[0][1], df_obj['PositionForIiwa14'].iloc[0][0], label=f"idx:{index}")
            start_pos1.append([df_obj['PositionForIiwa14'].iloc[0][1], df_obj['PositionForIiwa14'].iloc[0][0]])
            
            axes[1].set_title("IIWA 14", fontsize=GLOBAL_FONTSIZE)
            
            # Adding info when clicking cursor
            if use_mplcursors:
                mplcursors.cursor(scatter_iiwa14, hover=False).connect('add', lambda sel: sel.annotation.set_text(sel.artist.get_label()))   

    # add colors
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()
    axes[0].scatter(np.array(start_pos0)[:,0],np.array(start_pos0)[:,1], c=df_iiwa7['HittingFlux'], cmap="viridis")
    axes[1].scatter(np.array(start_pos1)[:,0],np.array(start_pos1)[:,1], c=df_iiwa14['HittingFlux'], cmap="viridis")
    norm = plt.Normalize(vmin=min(df_iiwa7['HittingFlux'].min(), df_iiwa14['HittingFlux'].min()), vmax=max(df_iiwa7['HittingFlux'].max(), df_iiwa14['HittingFlux'].max()))
    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist())
    cbar.set_label('Hitting Flux [m/s]', fontsize=GLOBAL_FONTSIZE)

    # Set axes settings
    for ax in axes:
        ax.set_ylabel('X-axis [m]', fontsize=GLOBAL_FONTSIZE)
        ax.grid(True, alpha=0.5)
        ax.set_ylim(0.39,0.62)
        

    axes[1].set_xlabel('Y Axis [m]', fontsize=GLOBAL_FONTSIZE)

    # leg = fig.legend(loc='center left', bbox_to_anchor=(0.9, 0.9))
    # leg.set_draggable(state=True)

    fig.suptitle(f"Object trajectories seen from above", fontsize=GLOBAL_FONTSIZE)
    # plt.title(f"Object trajectories seen from above", fontsize=GLOBAL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE) 
    
    if show_plot : plt.show()

def plot_object_start_end(df, dataset_path="varying_flux_datasets/D1/", relative=True, use_mplcursors=True, selection="all", show_plot = False):
    # Plot object start and end view form above 
    # One figure for each iiwa
    # If relative, plot relative to start position
    
    df_iiwa7 = df[df['IiwaNumber'] == 7]
    df_iiwa14 = df[df['IiwaNumber'] == 14]
    
    # SELECTION - taking only high fluxes
    max_flux_7 = 0.80
    max_flux_14 = 0.80
    high_flux_iiwa_7_df = df_iiwa7[(df['HittingFlux'] >= max_flux_7)]
    high_flux_iiwa_14_df = df_iiwa14[(df['HittingFlux'] >= max_flux_14)]

    high_flux_iiwa_7_df.reset_index(drop=True, inplace=True)
    high_flux_iiwa_14_df.reset_index(drop=True, inplace=True)
    
    print(f"Plotting {len(high_flux_iiwa_7_df.index)} points for iiwa 7")
    
    # get object number based on dataset path
    dataset = dataset_path.split('/')[1]
    object_number = get_object_based_on_dataset(dataset)
    
    ## PLOT FOR IIWA 7
    plt.figure(figsize=(18, 10))
    
    highest_angle = -10
    lowest_angle = 0
    
    for index,row in high_flux_iiwa_7_df.iterrows():

        # Get object trajectory from file name
        if os.name == "nt": # Windows OS
            rec_sess = row["RecSession"].replace(":","_")
        else : rec_sess = row["RecSession"]
        obj_fn = PATH_TO_DATA_FOLDER + dataset_path + rec_sess + f"/object_{object_number}_hit_{row['HitNumber']}.csv"
        
        if relative :
            # Plot start
            plt.scatter(0, 0, color='b', marker = 'o')
            # Plot end
            plt.scatter(row['ObjectPosEnd'][1]-row['ObjectPosStart'][1], row['ObjectPosEnd'][0]-row['ObjectPosStart'][0], alpha=0.6, color='r', marker = 'x')
            
        else :
            # Plot start
            plt.scatter(row['ObjectPosStart'][1], row['ObjectPosStart'][0], alpha=0.6, color='b', marker = 'o')
            # Plot end
            plt.scatter(row['ObjectPosEnd'][1], row['ObjectPosEnd'][0], alpha=0.6, color='r', marker = 'x')
        
        # Get highest and lowest traj fn - according to angle
        angle = np.degrees(np.arctan2(row['ObjectPosEnd'][0]-row['ObjectPosStart'][0], abs(row['ObjectPosEnd'][1]-row['ObjectPosStart'][1])))
        print(angle)
        if  angle <= lowest_angle : 
            lowest_x_fn = obj_fn
            lowest_angle = angle
            idx_lowest = index
        if  angle >= highest_angle : 
            highest_x_fn = obj_fn
            highest_angle = angle
            idx_highest = index

    # Add trajectory for highest x value
    df_obj = pd.read_csv(highest_x_fn , converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})
    start_time, end_time = get_impact_time_from_object(highest_x_fn)
    start_pos =  high_flux_iiwa_7_df['ObjectPosStart'].iloc[idx_highest]
    df_obj_moving = df_obj[(df_obj['RosTime']-end_time) <= 0 ] # get object traj while it's movign due to robot 
    if relative :
        plt.plot(df_obj_moving['PositionForIiwa7'].apply(lambda x: x[1]-start_pos[1]), df_obj_moving['PositionForIiwa7'].apply(lambda x: x[0]-start_pos[0]), color='g')
    else :
        plt.plot(df_obj_moving['PositionForIiwa7'].apply(lambda x: x[1]), df_obj_moving['PositionForIiwa7'].apply(lambda x: x[0]), color='g')
        
    # Add trajectory for lowest x value
    df_obj = pd.read_csv(lowest_x_fn , converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})
    start_time, end_time = get_impact_time_from_object(lowest_x_fn)
    start_pos =  high_flux_iiwa_7_df['ObjectPosStart'].iloc[idx_lowest]
    df_obj_moving = df_obj[(df_obj['RosTime']-end_time) <= 0 ] # get object traj while it's movign due to robot 
    if relative :
        plt.plot(df_obj_moving['PositionForIiwa7'].apply(lambda x: x[1]-start_pos[1]), df_obj_moving['PositionForIiwa7'].apply(lambda x: x[0]-start_pos[0]), color='g')
    else :
        plt.plot(df_obj_moving['PositionForIiwa7'].apply(lambda x: x[1]), df_obj_moving['PositionForIiwa7'].apply(lambda x: x[0]), color='g')
        
    # Get the angle 
    directional_error_angle = abs(highest_angle) + abs(lowest_angle)
    print("IIWA 7 - Angle lowest and highest trajectories:", directional_error_angle)
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='b', label='Start', markersize=10),
        Line2D([0], [0], marker='x', color='r', label='End', markersize=10),
        Line2D([0], [0], marker='', color='k', label=f'Angular Error: {directional_error_angle:.2f} degrees')
        ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=15)
    
    # Set plot variables 
    plt.axis('equal')
    plt.xlabel('Y Axis [m]',fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('X Axis [m]',fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)
    
    if relative:
        plt.title(f"Relative Object trajectories for IIWA 7 for fluxes over {max_flux_7}",fontsize=GLOBAL_FONTSIZE)
    else:
        plt.title(f"Absolute Object trajectories for IIWA 7 for fluxes over {max_flux_7}",fontsize=GLOBAL_FONTSIZE)
    
    ### PLOT FOR IIWA 14
    plt.figure(figsize=(18, 10))
    
    highest_angle = 0
    lowest_angle = 0
    
    for index,row in high_flux_iiwa_14_df.iterrows():

        # Get object trajectory from file name
        if os.name == "nt": # Windows OS
            rec_sess = row["RecSession"].replace(":","_")
        else : rec_sess = row["RecSession"]
        obj_fn = PATH_TO_DATA_FOLDER + dataset_path + rec_sess + f"/object_{object_number}_hit_{row['HitNumber']}.csv"
              
        if relative :
            # Plot start
            plt.scatter(0, 0, color='b', marker = 'o')
            # Plot end
            plt.scatter(row['ObjectPosEnd'][1]-row['ObjectPosStart'][1], row['ObjectPosEnd'][0]-row['ObjectPosStart'][0], alpha=0.6, color='r', marker = 'x')
            
        else :
            # Plot start
            plt.scatter(row['ObjectPosStart'][1], row['ObjectPosStart'][0], alpha=0.6, color='b', marker = 'o')
            # Plot end
            plt.scatter(row['ObjectPosEnd'][1], row['ObjectPosEnd'][0], alpha=0.6, color='r', marker = 'x')
            
        # Get highest and lowest traj fn - according to angle
        angle = np.degrees(np.arctan2(row['ObjectPosEnd'][0]-row['ObjectPosStart'][0], abs(row['ObjectPosEnd'][1]-row['ObjectPosStart'][1])))
        if  angle < lowest_angle : 
            lowest_x_fn = obj_fn
            lowest_angle = angle
            idx_lowest = index
        if  angle > highest_angle : 
            highest_x_fn = obj_fn
            highest_angle = angle
            idx_highest = index
    
    # Add trajectory for highest x value
    df_obj = pd.read_csv(highest_x_fn , converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})
    start_time, end_time = get_impact_time_from_object(highest_x_fn)
    start_pos =  high_flux_iiwa_14_df['ObjectPosStart'].iloc[idx_highest] # get this for offset
    df_obj_moving = df_obj[(df_obj['RosTime']-end_time) <= 0 ] # get object traj while it's movign due to robot 
    if relative :
        plt.plot(df_obj_moving['PositionForIiwa14'].apply(lambda x: x[1]-start_pos[1]), df_obj_moving['PositionForIiwa14'].apply(lambda x: x[0]-start_pos[0]), color='g')
    else :
        plt.plot(df_obj_moving['PositionForIiwa14'].apply(lambda x: x[1]), df_obj_moving['PositionForIiwa14'].apply(lambda x: x[0]), color='g')
        
    # Add trajectory for lowest x value
    df_obj = pd.read_csv(lowest_x_fn , converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})
    start_time, end_time = get_impact_time_from_object(lowest_x_fn)
    start_pos =  high_flux_iiwa_14_df['ObjectPosStart'].iloc[idx_lowest]
    df_obj_moving = df_obj[(df_obj['RosTime']-end_time) <= 0 ] # get object traj while it's movign due to robot 
    if relative :
        plt.plot(df_obj_moving['PositionForIiwa14'].apply(lambda x: x[1]-start_pos[1]), df_obj_moving['PositionForIiwa14'].apply(lambda x: x[0]-start_pos[0]), color='g')
    else :
        plt.plot(df_obj_moving['PositionForIiwa14'].apply(lambda x: x[1]), df_obj_moving['PositionForIiwa14'].apply(lambda x: x[0]), color='g')
        
    # Get the angle for highest traj
    directional_error_angle = abs(highest_angle) + abs(lowest_angle)
    print("Angle lowest and highest trajectories:", directional_error_angle)
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='b', label='Start', markersize=10),
        Line2D([0], [0], marker='x', color='r', label='End', markersize=10),
        Line2D([0], [0], marker='', color='k', label=f'Directional error angle: {directional_error_angle:.2f} degrees')
        ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=15)
    
    # Set plot variables 
    plt.axis('equal')
    plt.xlabel('Y Axis [m]', fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('X Axis [m]', fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)
    if relative :
        plt.title(f"Relative Object trajectories for IIWA 14 for fluxes over {max_flux_14}",fontsize=GLOBAL_FONTSIZE)
    else :
        plt.title(f"Absolute Object trajectories for IIWA 14 for fluxes over {max_flux_14}",fontsize=GLOBAL_FONTSIZE)
        
    if show_plot : plt.show()

def plot_orientation_vs_displacement(df, orientation ='error', sanity_check=False, show_plot = False, only_7=False, object_number=1):

    ## calculate quaternion diff
    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_x_y_z(row["ObjectOrientation"],row["HittingOrientation"]),axis=1).copy()
    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_x_y_z(row["HittingOrientation"],row["ObjectOrientation"]),axis=1).copy()
    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_in_correct_base(row["HittingOrientation"],row["ObjectOrientation"], row['IiwaNumber']),axis=1).copy()
    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_manually(row["HittingOrientation"],row["ObjectOrientation"], row['IiwaNumber']),axis=1).copy()

    if object_number == 1 :
        df["OrientationError"] = df.apply(lambda row : get_orientation_error_manually(row["HittingOrientation"],row["ObjectOrientation"], row['IiwaNumber']),axis=1).copy()
    if object_number == 2:
        # df["OrientationError"] = df.apply(lambda row : get_orientation_error_x_y_z(get_corrected_quat_object_2(row["ObjectOrientation"]),row["HittingOrientation"]),axis=1).copy()
        df["OrientationError"] = df.apply(lambda row : get_orientation_error_manually(row["HittingOrientation"], get_corrected_quat_object_2(row["ObjectOrientation"]), row['IiwaNumber']),axis=1).copy()


    # Remove outliers
    df = df[df['OrientationError'].apply(lambda x: abs(x[2]) < 20 )]

    ## calculate object displacement
    df["ObjectDisplacement"] = df.apply(lambda row : [a-b for a,b in zip(row["ObjectPosEnd"],row["ObjectPosStart"])],axis=1).copy()
        
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    ## PLOT FOR IIWA 7
    plt.figure(figsize=(18, 10))
    # Add XY lines 
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Customize the line as needed
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)  # Customize the line as needed

    # Defien x vlaues depending on orientation parameter
    if orientation == 'error':
        x_values = df_iiwa7['OrientationError'].apply(lambda x : x[2])
        plt.xlabel('Orientation Error in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
        plt.title(f"Object Displacement depending on Orientation Error at Hit Time for IIWA 7",fontsize=GLOBAL_FONTSIZE)
    if orientation == 'object':
        x_values = df_iiwa7['ObjectOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True)[2])
        plt.xlabel('Object Orientation in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
        plt.title(f"Object Displacement depending on Object Orientation at Hit Time for IIWA 7",fontsize=GLOBAL_FONTSIZE)
    if orientation == 'eef':
        x_values = df_iiwa7['HittingOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True)[2]) 
        plt.xlabel('EEF Orientation in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
        plt.title(f"Object Displacement depending on EEF Orientation at Hit Time for IIWA 7",fontsize=GLOBAL_FONTSIZE)
    
    # Orientation in z vs displacement in x 
    plt.scatter(x_values,df_iiwa7['ObjectDisplacement'].apply(lambda x : x[0]))
    # plt.scatter(df_iiwa7['OrientationError2'].apply(lambda x : x[2]),df_iiwa7['ObjectDisplacement'].apply(lambda x : x[0]))

    # Sanity check
    if sanity_check:
        errors = df_iiwa7['OrientationError'].apply(lambda x : x)
        objects = df_iiwa7['ObjectOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True))
        eefs = df_iiwa7['HittingOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True)) 
        
        for i in range(len(df_iiwa7.index)):
            print(f"Object: [{objects.iloc[i][0]:.2f},{objects.iloc[i][1]:.2f},{objects.iloc[i][2]:.2f}], " 
                  f"EEF: [{eefs.iloc[i][0]:.2f},{eefs.iloc[i][1]:.2f},{eefs.iloc[i][2]:.2f}], "
                  f"Error:[{errors.iloc[i][0]:.2f}, {errors.iloc[i][1]:.2f}, {errors.iloc[i][2]:.2f}]")
        
    # Centering around 0
    max_limit_x = max(np.abs(x_values))
    max_limit_y = max(np.abs(df_iiwa7['ObjectDisplacement'].apply(lambda x : x[0])))
    plt.xlim(-max_limit_x-0.05*max_limit_x, max_limit_x+0.05*max_limit_x)
    plt.ylim(-max_limit_y-0.05*max_limit_y, max_limit_y+0.05*max_limit_y)

    # Set plot variables 
    plt.ylabel('Displacement in X Axis [m]',fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)

    # TODO :
    # add flux as color to show no correlation
    # plot only high or low fluxes

    if(not only_7):
        ## PLOT FOR IIWA 14
        plt.figure(figsize=(18, 10))
        # Add XY lines 
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Customize the line as needed
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)  # Customize the line as needed

        # Defien x vlaues depending on orientation parameter
        if orientation == 'error':
            x_values = df_iiwa14['OrientationError'].apply(lambda x : x[2])
            plt.xlabel('Orientation Error in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
            plt.title(f"Object Displacement depending on Orientation Error at Hit Time for IIWA 14",fontsize=GLOBAL_FONTSIZE)
        if orientation == 'object':
            x_values = df_iiwa14['ObjectOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True)[2])
            plt.xlabel('Object Orientation in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
            plt.title(f"Object Displacement depending on Object Orientation at Hit Time for IIWA 14",fontsize=GLOBAL_FONTSIZE)
        if orientation == 'eef':
            x_values = df_iiwa14['HittingOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True)[2]) 
            plt.xlabel('EEF Orientation in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
            plt.title(f"Object Displacement depending on EEF Orientation at Hit Time for IIWA 14",fontsize=GLOBAL_FONTSIZE)
        
        # Orientation in z vs displacement in x 
        plt.scatter(x_values,df_iiwa14['ObjectDisplacement'].apply(lambda x : x[0]))
        # plt.scatter(df_iiwa14['OrientationError2'].apply(lambda x : x[2]),df_iiwa14['ObjectDisplacement'].apply(lambda x : x[0]))

        # Sanity check
        if sanity_check:
            errors = df_iiwa14['OrientationError'].apply(lambda x : x)
            objects = df_iiwa14['ObjectOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True))
            eefs = df_iiwa14['HittingOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True)) 
            
            for i in range(len(df_iiwa14.index)):
                print(f"Object: [{objects.iloc[i][0]:.2f},{objects.iloc[i][1]:.2f},{objects.iloc[i][2]:.2f}], " 
                    f"EEF: [{eefs.iloc[i][0]:.2f},{eefs.iloc[i][1]:.2f},{eefs.iloc[i][2]:.2f}], "
                    f"Error:[{errors.iloc[i][0]:.2f}, {errors.iloc[i][1]:.2f}, {errors.iloc[i][2]:.2f}]")
            
        # Centering around 0
        max_limit_x = max(np.abs(x_values))
        max_limit_y = max(np.abs(df_iiwa14['ObjectDisplacement'].apply(lambda x : x[0])))
        plt.xlim(-max_limit_x-0.05*max_limit_x, max_limit_x+0.05*max_limit_x)
        plt.ylim(-max_limit_y-0.05*max_limit_y, max_limit_y+0.05*max_limit_y)

    # Set plot variables 
    plt.ylabel('Displacement in X Axis [m]',fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)

def plot_orientation_vs_flux(df, sanity_check=False, show_plot = False, object_number=1):


    ### Correct object 2 orientation
    df["CorrectedOrientation"] = df.apply(lambda row : get_corrected_quat_object_2(row["ObjectOrientation"]),axis=1).copy()
    
    ## calculate quaternion diff
    if object_number == 1 :
        df["OrientationError"] = df.apply(lambda row : get_orientation_error_manually(row["HittingOrientation"],row["ObjectOrientation"], row['IiwaNumber']),axis=1).copy()
    if object_number == 2:
        df["OrientationError"] = df.apply(lambda row : get_orientation_error_x_y_z(get_corrected_quat_object_2(row["ObjectOrientation"]),row["HittingOrientation"]),axis=1).copy()

    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_x_y_z(row["CorrectedOrientation"],row["HittingOrientation"]),axis=1).copy()
    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_x_y_z(get_corrected_quat_object_2(row["ObjectOrientation"]),row["HittingOrientation"]),axis=1).copy()
    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_in_correct_base(row["HittingOrientation"],row["CorrectedOrientation"], iiwa_number=row['IiwaNumber']),axis=1).copy()
    # df["OrientationError"] = df.apply(lambda row : get_orientation_error_manually(row["HittingOrientation"],row["ObjectOrientation"], row['IiwaNumber']),axis=1).copy()
   
    
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    # Sanity check
    if sanity_check:
        errors = df_iiwa7['OrientationError'].apply(lambda x : x)
        objects = df_iiwa7['ObjectOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('xyz', degrees=True))
        eefs = df_iiwa7['HittingOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('XYZ', degrees=True)) 
        corrected = df_iiwa7['CorrectedOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('XYZ', degrees=True))
        
        for i in range(len(df_iiwa7.index)):
            print(f"Object: [{objects.iloc[i][0]:.2f},{objects.iloc[i][1]:.2f},{objects.iloc[i][2]:.2f}], " 
                  f"Corrected :[{corrected.iloc[i][0]:.2f}, {corrected.iloc[i][1]:.2f}, {corrected.iloc[i][2]:.2f}], "
                  f"EEF: [{eefs.iloc[i][0]:.2f},{eefs.iloc[i][1]:.2f},{eefs.iloc[i][2]:.2f}], "
                  f"Error:[{errors.iloc[i][0]:.2f}, {errors.iloc[i][1]:.2f}, {errors.iloc[i][2]:.2f}]")
 

    ## PLOT FOR IIWA 7
    plt.figure(figsize=(18, 10))

    # Orientation in z vs displacement in x 
    plt.scatter(df_iiwa7['HittingFlux'], df_iiwa7['HittingOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('XYZ', degrees=True)[2])  )
    plt.scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError'].apply(lambda x : x[2]))

    # Set plot variables 
    plt.xlabel('Hitting Flux [m/s]',fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('EEF Orientation in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
    plt.title(f"Hitting Flux vs EEF Orientation at Hit Time for IIWA 7",fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)

    ## PLOT FOR IIWA 14
    plt.figure(figsize=(18, 10))

    # Orientation in z vs displacement in x 
    plt.scatter(df_iiwa14['HittingFlux'], df_iiwa14['HittingOrientation'].apply(lambda x : Rotation.from_quat(x).as_euler('XYZ', degrees=True)[2]-90.0)  )
    plt.scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError'].apply(lambda x : x[2]))

    # Set plot variables 
    plt.xlabel('Hitting Flux [m/s]',fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('EEF Orientation in Z Axis [deg]',fontsize=GLOBAL_FONTSIZE)
    plt.title(f"Hitting Flux vs EEF Orientation at Hit Time for IIWA 14",fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)

    if show_plot:
        plt.show()

def plot_displacement_vs_flux(df, show_plot = False):

    ## calculate object displacement
    df["ObjectDisplacement"] = df.apply(lambda row : [a-b for a,b in zip(row["ObjectPosEnd"],row["ObjectPosStart"])],axis=1).copy()
    
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    
    ## PLOT FOR IIWA 7
    plt.figure(figsize=(18, 10))

    # Orientation in z vs displacement in x 
    plt.scatter(df_iiwa7['HittingFlux'], df_iiwa7['ObjectDisplacement'].apply(lambda x : x[0]))

    # Set plot variables 
    plt.xlabel('Hitting Flux [m/s]',fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('Object Displacement in X Axis [m]',fontsize=GLOBAL_FONTSIZE)
    plt.title(f"Hitting Flux vs Object Displacement at Hit Time for IIWA 7",fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)

    ## PLOT FOR IIWA 14
    plt.figure(figsize=(18, 10))

    # Orientation in z vs displacement in x 
    plt.scatter(df_iiwa14['HittingFlux'], df_iiwa14['ObjectDisplacement'].apply(lambda x : x[0]))

    # Set plot variables 
    plt.xlabel('Hitting Flux [m/s]',fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('Object Displacement in X Axis [m]',fontsize=GLOBAL_FONTSIZE)
    plt.title(f"Hitting Flux vs Object Displacement at Hit Time for IIWA 14",fontsize=GLOBAL_FONTSIZE)
    plt.grid(True)

    if show_plot:
        plt.show()


## TEST TO GET PRECISE HIT POSITION 
def compute_line_params_centered(point, angle_deg, length):
    # Angle is now measured from the vertical (y-axis)
    angle_rad = np.radians(angle_deg)
    half_length = length / 2
    dx = half_length * np.sin(angle_rad)
    dy = half_length * np.cos(angle_rad)
    return (point[0] - dx, point[1] - dy, point[0] + dx, point[1] + dy)

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel or coincident
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

def get_precise_hit_position(df):
    # Ignoring z axis for now, we compute the orientation of both EEF an dobject as seen from above
    # Then compute the intersection 

    hit = df.iloc[100]
    print(hit)

    point_eef = [hit["HittingPos"][1]*100,hit["HittingPos"][0]*100]
    point_object = [(hit["ObjectPosStart"][1]-0.12) *100,hit["ObjectPosStart"][0]*100]
    # # TODO fix this line, becuase its not grabbing a list
    # point_object = [hit["AttractorPos"][1]*100,hit["AttractorPos"][0]*100]

    angle_eef = Rotation.from_quat(hit["HittingOrientation"]).as_euler('XYZ', degrees=True)[2]
    angle_object= Rotation.from_quat(hit["ObjectOrientation"]).as_euler('xyz', degrees=True)[2]

    print(point_eef, point_object)

    line1 = compute_line_params_centered(point_eef,angle_eef, 17)
    line2 = compute_line_params_centered(point_object, angle_object, 26)

    # Find the intersection point
    intersection = line_intersection(line1, line2)

    print("Line 1:", line1)
    print("Line 2:", line2)
    if intersection:
        print("Intersection point:", intersection)
    else:
        print("The lines do not intersect or are parallel.")

    # Plot the lines and intersection point
    fig, ax = plt.subplots()
    ax.plot([line1[0], line1[2]], [line1[1], line1[3]], label='EEF')
    ax.plot([line2[0], line2[2]], [line2[1], line2[3]], label='Object')

    # Plot the intersection point
    if intersection:
        ax.plot(intersection[0], intersection[1], 'ro', label='Intersection')

    # Plot the center points
    ax.plot(point_eef[0], point_eef[1], 'bo', label='EEF Position')
    ax.plot(point_object[0], point_object[1], 'go', label='Objet Position')

    # Set plot limits
    ax.set_xlim(28, 32)
    ax.set_ylim(45,65)

    # Add labels and legend
    ax.set_xlabel('Y [cm]')
    ax.set_ylabel('X [cm]')
    ax.legend()
    ax.grid(True)

    # plt.show()

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


def save_all_figures(folder_name, title=None): 

    # Specify the directory where you want to save the figures
    save_dir = PATH_TO_DATA_FOLDER + "figures/" + folder_name

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all figures without window borders
    for fig in plt.get_fignums():
        plt.figure(fig)
        if title is None :
            title = plt.gca().get_title()  # Get the title of the figure
        file_name = f"{title}.png" if title else f"figure_{fig}.png"
        plt.gca().set_frame_on(True)  # Turn on frame
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight", pad_inches=0.1)

    print("All figures saved successfully!")


if __name__== "__main__" :
    
    #### SAVE CLEAN DATASET TO THIS FOLDER
    # clean_dataset_folder = "KL_div"

    # ### Datasets to use
    # csv_fn = "D1_clean" #"D1_clean" #"data_test_april"#  #"data_consistent_march"

    # csv_fn2 = "D2_clean"

    # ### Read and clean datasets
    # clean_df = read_and_clean_data(csv_fn, resample=True, n_samples=2000, only_7=False, save_folder=clean_dataset_folder, save_clean_df=True)
    # clean_df2 = read_and_clean_data(csv_fn2, resample=True, n_samples=900, only_7=False, save_folder=clean_dataset_folder, save_clean_df=True)

    # #### AGNOSTIC PARAM
    # agnostic_param = "config"

    # ## for special cases
    # # clean_df2 = clean_df2[clean_df2['RecSession']=="2024-05-08_16:27:34"]

    # df_combined = restructure_for_agnostic_plots(clean_df, clean_df2, resample=False, parameter=agnostic_param)
    
    # ### Values used for plots 
    # object_number = get_object_based_on_dataset(csv_fn)

    ### Plot functions
    # plot_distance_vs_flux(clean_df, colors="iiwa", with_linear_regression=True, use_mplcursors=False)
    # flux_hashtable(clean_df)
    # plot_object_start_end(clean_df, dataset_path="varying_flux_datasets/D4/", relative=True)
    # plot_orientation_vs_displacement(clean_df, orientation='error',sanity_check=False, only_7=True,  object_number=object_number)
    # plot_orientation_vs_flux(clean_df, sanity_check=True, object_number=object_number)
    # plot_displacement_vs_flux(clean_df) 
    # get_precise_hit_position(clean_df)

    # plot_hit_position(clean_df, plot="on object" , use_mplcursors=False)
    # plot_orientation_vs_distance(clean_df, axis="z")
    # plot_object_trajectory_onefig(clean_df, use_mplcursors=True, selection="all")
    # plot_object_trajectory(clean_df, use_mplcursors=True, selection="selected")

    ######### USED IN PAPER ###########

    # plot_distance_vs_flux(clean_df, colors="iiwa", with_linear_regression=True, use_mplcursors=False)
    # plot_distance_vs_flux(df_combined, colors=agnostic_param, with_linear_regression=True, use_mplcursors=False)
    # plot_distance_vs_flux(clean_df2, colors="config", with_linear_regression=True, use_mplcursors=False)
    # plot_object_trajectory_onefig(clean_df, dataset_path="varying_flux_datasets/D1/", use_mplcursors=False, selection="selected")

    # save_all_figures(folder_name=csv_fn)
    # save_all_figures(folder_name=clean_dataset_folder)
    # save_all_figures(folder_name=f"{agnostic_param}_agnostic")
    plt.show()


    # test_gmm_torch(clean_df)


