import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression
import mplcursors
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import pybullet 
import math

import sys
sys.path.append('/home/maxime/Workspace/air_hockey/python_data_processing/gmm_torch')
# from gmm_torch.gmm import GaussianMixture
# from gmm_torch.example import plot
# import torch

from gmr.utils import check_random_state
from gmr import MVN, GMM, plot_error_ellipses

from process_data import parse_value, parse_list, parse_strip_list, parse_strip_list_with_commas, get_orientation_error_x_y_z, PATH_TO_DATA_FOLDER


# PROCESSING
def wrap_angle(angle_rad):
    # wraps an angle
    angle_deg = math.degrees(angle_rad)
    mod_angle = (angle_deg) % 180
    if angle_deg >= 180 : return 180-mod_angle
    if angle_deg <= -180: return -mod_angle
    else: return angle_deg

# CLEANING FUNCTION
def clean_data(df, distance_threshold=0.05, flux_threshold=0.35, save_clean_df=False):
    
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
    ## didnt hit box ?? : 448 to remove for plot_hit_point_on_object
    # idx_to_remove = [436,437,438,439,440] # march dataset
    idx_to_remove = []

    clean_df = clean_df[~clean_df.index.isin(idx_to_remove)]
    clean_df.reset_index(drop=True, inplace=True)   

    print(f"Removed {len(df.index)-len(clean_df.index)} outlier datapoints")
   
   # Saving clean df
    if save_clean_df :
        processed_clean_folder = PATH_TO_DATA_FOLDER+"/airhockey_processed/clean/"
        if not os.path.exists(processed_clean_folder):
            os.makedirs(processed_clean_folder)
            
        clean_df.to_csv(processed_clean_folder+csv_fn+"_clean.csv",index_label="Index")
 
    return clean_df

# PLOTTING FUNCTIONS
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

def plot_gmr(df, n=3, plot="only_gmm"):
    
    X = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    mvn = MVN(random_state=0)
    mvn.from_samples(X)

    X_test = np.linspace(0.45, 1.1, 100)
    mean, covariance = mvn.predict(np.array([0]), X_test[:, np.newaxis])
    
    gmm = GMM(n_components=n, random_state=0)
    gmm.from_samples(X)
    Y = gmm.predict(np.array([0]), X_test[:, np.newaxis])

    plt.figure(figsize=(10, 5))
    colors_list = ["r","g","b","y","m","c","k"]

    if plot == "with_mvn":
        plt.subplot(1, 2, 1)
        plt.title("Linear: $p(Y | X) = \mathcal{N}(\mu_{Y|X}, \Sigma_{Y|X})$")
        plt.xlabel("Hitting Flux [m/s]")   
        plt.ylabel("Distance travelled [m]") 
        plt.scatter(X[:, 0], X[:, 1])
        y = mean.ravel()
        s = 1.96 * np.sqrt(covariance.ravel())  # interval covers 95% of the data
        plt.fill_between(X_test, y - s, y + s, alpha=0.2)
        plt.plot(X_test, y,  c="k",lw=2)

        plt.subplot(1, 2, 2)
        plt.title("Mixture of Experts: $p(Y | X) = \Sigma_k \pi_{k, Y|X} "
                "\mathcal{N}_{k, Y|X}$")
        plt.xlabel("Hitting Flux [m/s]")   
        plt.ylabel("Distance travelled [m]") 
        
        plt.scatter(X[:, 0], X[:, 1])
        plot_error_ellipses(plt.gca(), gmm, colors=colors_list[0:n], alpha = 0.12)
        plt.plot(X_test, Y.ravel(), c="k", lw=2)

    elif plot == "only_gmm":
        plt.title("Mixture of Experts: $p(Y | X) = \Sigma_k \pi_{k, Y|X} "
        "\mathcal{N}_{k, Y|X}$")
        plt.xlabel("Hitting Flux [m/s]")   
        plt.ylabel("Distance travelled [m]") 
        
        plt.scatter(X[:, 0], X[:, 1])
        plot_error_ellipses(plt.gca(), gmm, colors=colors_list[0:n], alpha = 0.12)
        plt.plot(X_test, Y.ravel(), c="k", lw=2)

    elif plot == "compare_gmm":
        plt.subplot(1, 2, 1)
        plt.title("Mixture of Experts: $p(Y | X) = \Sigma_k \pi_{k, Y|X} "
        "\mathcal{N}_{k, Y|X}$")
        plt.xlabel("Hitting Flux [m/s]")   
        plt.ylabel("Distance travelled [m]") 
        
        plt.scatter(X[:, 0], X[:, 1])
        plot_error_ellipses(plt.gca(), gmm, colors=colors_list[0:n], alpha = 0.12)
        plt.plot(X_test, Y.ravel(), c="k", lw=2)
        
        plt.subplot(1, 2, 2)
        
        gmm = GMM(n_components=n+1, random_state=0)
        gmm.from_samples(X)
        Y = gmm.predict(np.array([0]), X_test[:, np.newaxis])
        plt.title("Mixture of Experts: $p(Y | X) = \Sigma_k \pi_{k, Y|X} "
        "\mathcal{N}_{k, Y|X}$")
        plt.xlabel("Hitting Flux [m/s]")   
        plt.ylabel("Distance travelled [m]") 
        
        plt.scatter(X[:, 0], X[:, 1])
        plot_error_ellipses(plt.gca(), gmm, colors=colors_list[0:n+1], alpha = 0.12)
        plt.plot(X_test, Y.ravel(), c="k", lw=2)
    
    plt.show()


def plot_distance_vs_flux(df, colors="iiwa", with_linear_regression=True, gmm_model=None, use_mplcursors=True, show_plot=False):
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
    plt.title(f"Distance over Flux")
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


def flux_hashtable(df, use_mplcursors=True, show_plot = False):
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
    plt.title(f"Flux Hashtable")
    fig.tight_layout(rect=(0.01,0.01,0.99,0.99))
    
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

def plot_object_trajectory_onefig(df, use_mplcursors=True, selection="all", show_plot = False):

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)# 

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

    # # get wrapped orientation error
    # df["OrientationError"] = df.apply(lambda row : pybullet.getEulerFromQuaternion(pybullet.getDifferenceQuaternion(row["ObjectOrientation"],row["HittingOrientation"])),axis=1).copy()
    # df["OrientationError"] = df["OrientationError"].apply(lambda x : [wrap_angle(i) for i in x]).copy()
    
    # df["OrientationError2"] = df.apply(lambda row : [pybullet.getEulerFromQuaternion(row["HittingOrientation"])[i]- pybullet.getEulerFromQuaternion(row["ObjectOrientation"])[i] for i in range(3)],axis=1).copy()

    
    start_pos0 = [] # list for color
    start_pos1 = []

    for index,row in df.iterrows():

        # get object trajectory from file name
        data_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ "/data/airhockey/"
        if os.name == "nt": # Windows OS
            rec_sess = row["RecSession"].replace(":","_")
        else : rec_sess = row["RecSession"]
        
        obj_fn = data_folder + rec_sess + f"/object_hit_{row['HitNumber']}.csv"
        
        df_obj = pd.read_csv(obj_fn, converters={'RosTime' : parse_value, 'PositionForIiwa7': parse_list, 'PositionForIiwa14': parse_list})

        if row['IiwaNumber']==7 :
            
            axes[0].plot(df_obj['PositionForIiwa7'].apply(lambda x: x[1]), df_obj['PositionForIiwa7'].apply(lambda x: x[0]), alpha=0.8)
            scatter_iiwa7 = axes[0].scatter(df_obj['PositionForIiwa7'].iloc[0][1], df_obj['PositionForIiwa7'].iloc[0][0], label=f"idx:{index}")
            start_pos0.append([df_obj['PositionForIiwa7'].iloc[0][1], df_obj['PositionForIiwa7'].iloc[0][0]])

            axes[0].set_title("IIWA 7")
            
            # Adding info when clicking cursor
            if use_mplcursors:
                mplcursors.cursor(scatter_iiwa7, hover=False).connect('add', lambda sel: sel.annotation.set_text(sel.artist.get_label()))   

        if row['IiwaNumber']==14 :
            
            axes[1].plot(df_obj['PositionForIiwa14'].apply(lambda x: x[1]), df_obj['PositionForIiwa14'].apply(lambda x: x[0]))
            scatter_iiwa14 = axes[1].scatter(df_obj['PositionForIiwa14'].iloc[0][1], df_obj['PositionForIiwa14'].iloc[0][0], label=f"idx:{index}")
            start_pos1.append([df_obj['PositionForIiwa14'].iloc[0][1], df_obj['PositionForIiwa14'].iloc[0][0]])
            
            axes[1].set_title("IIWA 14")
            
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
    cbar.set_label('Hitting Flux [m/s]')

    for ax in axes:
        ax.set_xlabel('Y Axis [m]')
        ax.set_ylabel('X-axis [m]')
        ax.grid(True)
        # ax.set_aspect('equal')
        
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


def save_all_figures(dataset): 

    # Specify the directory where you want to save the figures
    save_dir = PATH_TO_DATA_FOLDER + "figures/" + dataset

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save all figures without window borders
    for fig in plt.get_fignums():
        plt.figure(fig)
        title = plt.gca().get_title()  # Get the title of the figure
        print(title)
        file_name = f"{title}.png" if title else f"figure_{fig}.png"
        plt.gca().set_frame_on(True)  # Turn on frame
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight", pad_inches=0.1)

    print("All figures saved successfully!")


if __name__== "__main__" :
    
    processed_raw_folder = PATH_TO_DATA_FOLDER + "airhockey_processed/raw/"
    
    ### Datafile to use
    # csv_fn ="100_hits-object_1-config_1-fixed_start-random_flux-IIWA_7-reduced_inertia" #"data_test_april"#  #"data_consistent_march"
    csv_fn ="D1_clean" #"data_test_april"#  #"data_consistent_march"


    ## Reading and cleanign data 
    df = pd.read_csv(processed_raw_folder+csv_fn+".csv", index_col="Index", converters={
        'ObjectPos' : parse_strip_list_with_commas, 'HittingPos': parse_strip_list_with_commas, 
        'ObjectOrientation' : parse_strip_list, 'HittingOrientation': parse_strip_list_with_commas})#
    
    clean_df = clean_data(df, save_clean_df=True)

    ### Plot functions
    plot_distance_vs_flux(clean_df, colors="iiwa", with_linear_regression=True)
    # plot_hit_position(clean_df, plot="on object" , use_mplcursors=False)
    # plot_orientation_vs_distance(clean_df, axis="z")
    flux_hashtable(clean_df)
    # plot_object_trajectory_onefig(clean_df, use_mplcursors=True, selection="all")
    # plot_object_trajectory(clean_df, use_mplcursors=True, selection="all")

    save_all_figures(dataset=csv_fn)
    plt.show()


    # test_gmm_torch(clean_df)
    # plot_gmr(clean_df, n=3, plot="only_gmm")

