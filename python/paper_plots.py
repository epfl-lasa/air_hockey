import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
from sklearn.linear_model import LinearRegression
import mplcursors

from process_data import  parse_list, parse_value, PATH_TO_DATA_FOLDER

from analyse_data import read_airhockey_csv, read_and_clean_data, save_all_figures, get_object_based_on_dataset, restructure_for_agnostic_plots


# Fontsize for axes and titles 
GLOBAL_FONTSIZE = 30
AXIS_TICK_FONTSIZE = 20

PROCESSED_RAW_FOLDER = PATH_TO_DATA_FOLDER + "airhockey_processed/raw/"
SAVE_FOLDER_FOR_PAPER = "for_paper"   

## Plots
def plot_distance_vs_flux(df, colors="iiwa", title="Distance over Flux", with_linear_regression=True, use_mplcursors=True, show_plot=False):
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
    plt.title(title ,fontsize=GLOBAL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE) 
    fig.tight_layout(rect=(0.01,0.01,0.99,0.99))

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


### Functions to call to recreate paper plots
def object_traj_plot_for_paper(dataset_to_use='clean'):

    ### Datasets to use
    csv_fn = "D1_clean" 

    dataset_new_name = "D1-object_traj_plot"

    ### Read OR clean datasets
    if dataset_to_use == "raw":
        clean_df = read_and_clean_data(csv_fn, dataset_name=dataset_new_name, resample=False, save_folder=SAVE_FOLDER_FOR_PAPER, save_clean_df=True)
    elif dataset_to_use == "clean":
        clean_df = read_airhockey_csv(fn=dataset_new_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{SAVE_FOLDER_FOR_PAPER}/")

    plot_object_trajectory_onefig(clean_df, dataset_path="varying_flux_datasets/D1/", use_mplcursors=False, selection="selected")

    save_all_figures(folder_name=SAVE_FOLDER_FOR_PAPER, title=dataset_new_name)

    plt.show()

def robot_agnostic(dataset_to_use='clean'):

    ### Datasets to use
    csv_fn = "D1_clean" 

    dataset_new_name = "D1-robot_agnostic"

    ### Read OR clean datasets
    if dataset_to_use == "raw":
        # TODO - separate per robot, sample, then recombine to get same number fo poitns per robot
        clean_df = read_and_clean_data(csv_fn, dataset_name=dataset_new_name, resample=False, n_samples=2000, max_flux=0.85, save_folder=SAVE_FOLDER_FOR_PAPER, save_clean_df=True)
    elif dataset_to_use == "clean":
        clean_df = read_airhockey_csv(fn=dataset_new_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{SAVE_FOLDER_FOR_PAPER}/")

    plot_distance_vs_flux(clean_df, colors="iiwa", with_linear_regression=True, use_mplcursors=False)

    save_all_figures(folder_name=SAVE_FOLDER_FOR_PAPER, title=dataset_new_name)

    plt.show()

def object_agnostic(dataset_to_use='clean'):

    ### Datasets to use
    csv_fn = "D1_clean"
    csv_fn2 = "D2_clean" 

    dataset_new_name = "D1-D2-object_agnostic"

    ### Read OR clean datasets
    if dataset_to_use == "raw":
        clean_df = read_and_clean_data(csv_fn, resample=False, n_samples=2000, only_7=True,  save_clean_df=False)
        clean_df2 = read_and_clean_data(csv_fn2, resample=True, n_samples=500, only_7=True,  save_clean_df=False)

        ## combine both into one df
        df_combined = restructure_for_agnostic_plots(clean_df, clean_df2, resample=False, parameter="object", 
                                                     dataset_name=dataset_new_name, save_folder=SAVE_FOLDER_FOR_PAPER, save_new_df=True)

    elif dataset_to_use == "clean":
        df_combined = read_airhockey_csv(fn=dataset_new_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{SAVE_FOLDER_FOR_PAPER}/")
        
    plot_distance_vs_flux(df_combined, colors="object", with_linear_regression=True, use_mplcursors=False)

    save_all_figures(folder_name=SAVE_FOLDER_FOR_PAPER, title=dataset_new_name)

    plt.show()

def config_agnostic(dataset_to_use='clean'):

    ### Datasets to use
    csv_fn = "D1_clean"
    csv_fn2 = "D3_clean" 

    dataset_new_name = "D1-D3-config_agnostic"

    ### Read OR clean datasets
    if dataset_to_use == "raw":
        clean_df = read_and_clean_data(csv_fn, resample=True, n_samples=100, max_flux=0.9, only_7=True,  save_clean_df=False)
        clean_df2 = read_and_clean_data(csv_fn2, resample=False, n_samples=500, max_flux=0.9, only_7=True,  save_clean_df=False)

        ## combine both into one df
        df_combined = restructure_for_agnostic_plots(clean_df, clean_df2, resample=False, parameter="config", 
                                                     dataset_name=dataset_new_name, save_folder=SAVE_FOLDER_FOR_PAPER, save_new_df=True)

    elif dataset_to_use == "clean":
        df_combined = read_airhockey_csv(fn=dataset_new_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{SAVE_FOLDER_FOR_PAPER}/")
        
    plot_distance_vs_flux(df_combined, colors="config", with_linear_regression=True, use_mplcursors=False)

    save_all_figures(folder_name=SAVE_FOLDER_FOR_PAPER, title=dataset_new_name)

    plt.show()



if __name__== "__main__" :
    
    ###### Call these functions to display and save each plot for the paper #####
    ## Tune parameters inside each respective function
    ## Use 'raw' dataset to sample different datapoints for plots

    object_traj_plot_for_paper()
    # robot_agnostic(dataset_to_use='clean')
    # object_agnostic(dataset_to_use='clean')
    # config_agnostic(dataset_to_use="clean")

