import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
import mplcursors
from matplotlib.patches import Rectangle
import pybullet 

import sys
sys.path.append('/home/maxime/Workspace/i_am_project/python_data_processing/gmm_torch')
# from gmm_torch.gmm import GaussianMixture
# from gmm_torch.example import plot
import torch

from process_data import parse_value, parse_list, parse_strip_list, parse_strip_list_with_commas

# CLEANING FUNCTION
def clean_data(df, distance_threshold=0.05, flux_threshold=0.35):
    
    ### Remove low outliers -> due to way of recording and processing
    # Distance
    clean_df = df[df['DistanceTraveled']>distance_threshold]
    # Flux
    clean_df = clean_df[clean_df['HittingFlux']>flux_threshold]

    # Reset index
    clean_df.reset_index(drop=True, inplace=True)       
    
    ## REMOVING datapoints manually -> after checking plots with plot_hit_data, these points are badly recorded
    ## maybe : 
    ## double hits (from 14): 202, 221, 250, 409, 791
    ## didnt hit box ?? : 789, 438 to remove for plot_hit_point_on_object
    idx_to_remove = []

    clean_df = clean_df[~clean_df.index.isin(idx_to_remove)]
    clean_df.reset_index(drop=True, inplace=True)   

    print(f"Removed {len(df.index)-len(clean_df.index)} outlier datapoints")
 
    return clean_df

# PLOTTING FUNCTIONS
def test_gmm_torch(df): 

    temp_array = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    data = torch.tensor(temp_array, dtype=torch.float32) 

    # Next, the Gaussian mixture is instantiated and ..
    model = GaussianMixture(n_components=5, n_features=2, covariance_type="full")
    model.fit(data,delta=1e-5, n_iter=500)
    # .. used to predict the data points as they where shifted
    # y = model.predict(data)

    # flux_test = torch.tensor(np.linspace(0.4,1.2,100).reshape(-1,1), dtype=torch.float32)
    # TODO : do bic test on predicted results ???
    # print(model.bic(flux_test))

    plot_distance_vs_flux(df, with_linear_regression=True, gmm_model=model)
    # plot(data, y)

def plot_distance_vs_flux(df, colors="iiwa", with_linear_regression=True, gmm_model=None, use_mplcursors=True):
    ## use colors input to dtermine color of datapoints

    # Plot Flux
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()


    if colors == "iiwa":
        ax.scatter(df_iiwa7['HittingFlux'], df_iiwa7['DistanceTraveled'], color='red', alpha=0.5, label='Iiwa 7')
        ax.scatter(df_iiwa14['HittingFlux'], df_iiwa14['DistanceTraveled'], color='blue', alpha=0.5, label='Iiwa 14')

    elif colors == "orientation":
        ###TODO : check this is correct ?? 
        df["OrientationError"] = df.apply(lambda row : np.linalg.norm(np.array(row["HittingOrientation"])-np.array(row["ObjectOrientation"])),axis=1)
        scatter = ax.scatter(df['HittingFlux'], df['DistanceTraveled'], c=df['OrientationError'], cmap="viridis")
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Orientation error')

    ## Add linear regression
    if with_linear_regression: 
        lin_model = LinearRegression()
        lin_model.fit(df['HittingFlux'].values.reshape(-1,1), df['DistanceTraveled'].values)

        flux_test = np.linspace(0.4,1.2,100).reshape(-1,1)
        distance_pred = lin_model.predict(flux_test)
        ax.plot(flux_test,distance_pred,color='black', label='Linear Regression')


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
    low_to_med_threshold = 0.65
    med_to_high_threshold = 0.85
    print(f"Dataset info : \n"
          f" Iiwa 7 points : {len(df_iiwa7.index)} \n"
          f" Iiwa 14 points : {len(df_iiwa14.index)} \n"
        #   f" Low Flux points (below {low_to_med_threshold}): {len(df[df['HittingFlux'] < low_to_med_threshold].index)} \n"
        #   f" Medium Flux points : {len(df[df['HittingFlux'] > low_to_med_threshold].index) - len(df[df['HittingFlux'] > med_to_high_threshold].index)} \n"
        #   f" High Flux points (above {med_to_high_threshold}) : {len(df[df['HittingFlux'] > med_to_high_threshold].index)} \n"
          f" Total points : {len(df.index)}")
    
    # Adding info when hovering cursor
    if use_mplcursors:
        mplcursors.cursor(hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df['RecSession'][sel.index]}, hit #{df['HitNumber'][sel.index]}, iiwa{df['IiwaNumber'][sel.index]}"))   

    ax.set_xlabel('Hitting flux [m/s]')
    ax.set_ylabel('Distance Traveled [m]')
    ax.grid(True)
    plt.legend()
    fig.suptitle(f"Distance over Flux")
    fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    plt.show()

def plot_hit_position(df, plot="on object", use_mplcursors=True):
    # plot choices : "on object" , "flux", "distance"

    # for each hit, get relative error in x,y,z 
    df["RelPosError"]=df.apply(lambda row : [a-b for a,b in zip(row["HittingPos"],row["ObjectPos"])], axis=1)
    df["RelPosError"] = df["RelPosError"].apply(lambda list : [x*100 for x in list]) # Read as cm

    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    # Show on object 
    if plot == "on object":
        # plot x,z with y as color 
        # scatter = plt.scatter(df_iiwa7["RelPosError"].apply(lambda x: x[0]),df_iiwa7["RelPosError"].apply(lambda x: x[2]), c=df_iiwa7["RelPosError"].apply(lambda x: abs(x[1])), cmap='viridis')
        scatter = plt.scatter(df_iiwa7["RelPosError"].apply(lambda x: x[0]),df_iiwa7["RelPosError"].apply(lambda x: x[2]), c=df_iiwa7["HittingFlux"], cmap='viridis')
        plt.scatter(0,0, c="red", marker="x")
        
        # Add box - TODO : define it better
        rect = Rectangle((-10, -10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        cbar = plt.colorbar(scatter)
        # cbar.set_label('Y-axis [cm]')
        cbar.set_label('Flux')
        plt.xlabel('X-axis [cm]')
        plt.ylabel('Z-Axis[cm]')
        plt.title('Hitting Point shown on object - IIWA 7')

        ###SECOND FIG FOR IIWA 14
        plt.figure()
        # plot x,z with y as color 
        # scatter = plt.scatter(df_iiwa14["RelPosError"].apply(lambda x: x[0]),df_iiwa14["RelPosError"].apply(lambda x: x[2]), c=df_iiwa14["RelPosError"].apply(lambda x: abs(x[1])), cmap='viridis')
        scatter = plt.scatter(df_iiwa14["RelPosError"].apply(lambda x: x[0]),df_iiwa14["RelPosError"].apply(lambda x: x[2]), c=df_iiwa14["HittingFlux"], cmap='viridis')
        plt.scatter(0,0, c="red", marker="x")
        
        # Add box - TODO : define it better
        rect = Rectangle((-10, -10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
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
        mplcursors.cursor(hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df['RecSession'][sel.index]}, hit #{df['HitNumber'][sel.index]}, iiwa{df['IiwaNumber'][sel.index]}"))   

    plt.show()

def plot_orientation_vs_distance(df, axis="z", use_mplcursors=True):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    ## calculate quaternion diff
    df["OrientationError"] = df.apply(lambda row : pybullet.getEulerFromQuaternion(pybullet.getDifferenceQuaternion(row["ObjectOrientation"],row["HittingOrientation"])),axis=1)

    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()

    if axis == "x":
        # scatter = axes.scatter(df['DistanceTraveled'], df['OrientationError'].apply(lambda x : x[0]), c=df['HittingFlux'], cmap="viridis")
        axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa7['DistanceTraveled'], cmap="viridis")
        scatter = axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa14['DistanceTraveled'], cmap="viridis")
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Hitting Flux [m/s]')
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

        scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['OrientationError'].apply(lambda x : x[2]), c=df_iiwa7['DistanceTraveled'], cmap="viridis")
        axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['OrientationError'].apply(lambda x : x[2]), c=df_iiwa14['DistanceTraveled'], cmap="viridis")

        # scatter = axes[0].scatter(df_iiwa7['DistanceTraveled'], df_iiwa7['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa7['HittingFlux'], cmap="viridis")
        # axes[1].scatter(df_iiwa14['DistanceTraveled'], df_iiwa14['HittingOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[0]), c=df_iiwa14['HittingFlux'], cmap="viridis")

        # scatter = axes[0].scatter(df_iiwa7['HittingFlux'], df_iiwa7['ObjectOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[1]), c=df_iiwa7['DistanceTraveled'], cmap="viridis")
        # axes[1].scatter(df_iiwa14['HittingFlux'], df_iiwa14['ObjectOrientation'].apply(lambda x : pybullet.getEulerFromQuaternion(x)[1]), c=df_iiwa14['DistanceTraveled'], cmap="viridis")


        cbar = plt.colorbar(scatter, ax=axes.ravel().tolist())
        # cbar.set_label('Hitting Flux [m/s]')
        cbar.set_label('Distance traveled [m]')

        axes[0].set_ylabel('Orientation in Z-axis [rad]')
    


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
    fig.suptitle(f"Orientation over Flux")
    # fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

    plt.show()


def flux_hashtable(df, use_mplcursors=True):
    # Plot Flux
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

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

    ax.set_xlabel('Desired flux [m/s]')
    ax.set_ylabel('Hitting Flux [m/s]')
    ax.grid(True)
    plt.legend()
    fig.suptitle(f"Flux Hashtable")
    fig.tight_layout(rect=(0.01,0.01,0.99,0.99))
    
    plt.show()


def plot_object_trajectory(df, use_mplcursors=True):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    df["OrientationError"] = df.apply(lambda row : pybullet.getEulerFromQuaternion(pybullet.getDifferenceQuaternion(row["ObjectOrientation"],row["HittingOrientation"])),axis=1)

    start_pos0 =[] # list for color
    start_pos1 = []

    for index,row in df.iterrows():

        if row['IiwaNumber']==7 :
            
            # get object trajectory from file name
            data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ "/data/airhockey/"
            obj_fn = data_folder + row["RecSession"] + f"/object_hit_{row['HitNumber']}.csv"
            
            df_obj = pd.read_csv(obj_fn, converters={'RosTime' : parse_value, 'Position': parse_list, 'Orientation': parse_list})

            # inverted to adpat to optitrack reference frame
            axes[0].plot(-df_obj['Position'].apply(lambda x: x[0]), df_obj['Position'].apply(lambda x: x[1]), alpha=0.8)
            # axes[0].scatter(-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1], c=row['OrientationError'][2], cmap="viridis")
            start_pos0.append([-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1]])

            axes[0].set_title("IIWA 7")

        if row['IiwaNumber']==14 :
            
            # get object trajectory from file name
            data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ "/data/airhockey/"
            obj_fn = data_folder + row["RecSession"] + f"/object_hit_{row['HitNumber']}.csv"
            
            df_obj = pd.read_csv(obj_fn, converters={'RosTime' : parse_value, 'Position': parse_list, 'Orientation': parse_list})

            # inverted to adpat to optitrack reference frame
            axes[1].plot(-df_obj['Position'].apply(lambda x: x[0]), df_obj['Position'].apply(lambda x: x[1]))
            # scatter = axes[1].scatter(-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1], c=row['OrientationError'][2], cmap="viridis")
            start_pos1.append([-df_obj['Position'].iloc[0][0], df_obj['Position'].iloc[0][1]])

            axes[1].set_title("IIWA 14")
    

    # Adding info when hovering cursor
    if use_mplcursors:
        mplcursors.cursor(hover=True).connect('add', lambda sel: sel.annotation.set_text(
            f"IDX: {sel.index} Rec:{df['RecSession'][sel.index]}, hit #{df['HitNumber'][sel.index]}, iiwa{df['IiwaNumber'][sel.index]}"))   

    # add colors
    df_iiwa7 = df[df['IiwaNumber']==7].copy()
    df_iiwa14 = df[df['IiwaNumber']==14].copy()
    axes[0].scatter(np.array(start_pos0)[:,0],np.array(start_pos0)[:,1], c=df_iiwa7['OrientationError'].apply(lambda x : x[2]), cmap="viridis")
    scatter = axes[1].scatter(np.array(start_pos1)[:,0],np.array(start_pos1)[:,1], c=df_iiwa14['OrientationError'].apply(lambda x : x[2]), cmap="viridis")
    cbar = plt.colorbar(scatter, ax=axes.ravel().tolist())
    cbar.set_label('Orienttion Error in Z-axis [rad]')

    for ax in axes:
        ax.set_xlabel('Y Axis [m]')
        ax.set_ylabel('X-axis [m]')
        ax.grid(True)
    plt.legend()
    fig.suptitle(f"Object trajectories")
    # fig.tight_layout(rect=(0.01,0.01,0.99,0.99))
    
    plt.show()

if __name__== "__main__" :
    
    processed_data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ "/data/airhockey_processed/"
    
    ### Datafile to use
    csv_fn = "data_consistent_march" #"data_consistent_march"


    ## Reading and cleanign data 
    df = pd.read_csv(processed_data_folder+csv_fn+".csv", index_col="Index", converters={'ObjectPos' : parse_strip_list_with_commas, 'HittingPos': parse_strip_list_with_commas, 'ObjectOrientation' : parse_strip_list, 'HittingOrientation': parse_strip_list})
    clean_df = clean_data(df)
    # Saving clean df
    clean_df.to_csv(processed_data_folder+csv_fn+"_clean.csv",index_label="Index")


    ### Plot functions
    # plot_distance_vs_flux(clean_df, colors="iiwa", with_linear_regression=True)
    plot_hit_position(clean_df, plot="on object" , use_mplcursors=True)
    # plot_orientation_vs_distance(clean_df, axis="x")
    # flux_hashtable(clean_df)
    # plot_object_trajectory(clean_df)

    # test_gmm_torch(clean_data(df))

