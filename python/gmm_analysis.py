import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.patches as patches
from gmr import MVN, GMM, plot_error_ellipses
from scipy.stats import entropy

from process_data import  PATH_TO_DATA_FOLDER

from analyse_data import read_airhockey_csv, read_and_clean_data, restructure_for_agnostic_plots, save_one_figure, resample_uniformally

# Fontsize for axes and titles 
GLOBAL_FONTSIZE = 40
AXIS_TICK_FONTSIZE = 30

PROCESSED_FOLDER = PATH_TO_DATA_FOLDER + "airhockey_processed/"

## GMM ANALYISIS
def plot_gmr(df, n=3, plot="only_gmm", title="Gaussian Mixture Model fit", save_fig=False, save_folder="", show_plot=False):
    
    X = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    mvn = MVN(random_state=0)
    mvn.from_samples(X)

    X_test = np.linspace(df['HittingFlux'].min(), df['HittingFlux'].max(), 100) 
    mean, covariance = mvn.predict(np.array([0]), X_test[:, np.newaxis])
    
    gmm = GMM(n_components=n, random_state=0)
    gmm.from_samples(X, R_diff=1e-5, n_iter=1000, init_params='kmeans++')
    Y = gmm.predict(np.array([0]), X_test[:, np.newaxis])

    plt.figure(figsize=(20, 10))
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
        # plt.title("Mixture of Experts: $p(Y | X) = \Sigma_k \pi_{k, Y|X} "
        # "\mathcal{N}_{k, Y|X}$")
        # plt.title(title, fontsize=GLOBAL_FONTSIZE)
        plt.xlabel("Hitting Flux [m/s]", fontsize=GLOBAL_FONTSIZE)   
        plt.ylabel("Distance travelled [m]", fontsize=GLOBAL_FONTSIZE) 
        
        plt.scatter(X[:, 0], X[:, 1], s=100)
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
    
    # Increase the size of the tick labels
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(2)  # Set left spine thickness
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE) 

    if save_fig : save_one_figure(save_folder, title)

    if show_plot: plt.show()

#### Functions to calculate BIC and AIC for a range of components
def calculate_bic_aic(X, max_components):
    bic_scores = []
    aic_scores = []
    n_components_range = range(1, max_components + 1)
    
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=0, tol=1e-5, max_iter=1000, init_params='k-means++')
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
    
    return n_components_range, bic_scores, aic_scores

def plot_bic_aic_with_sklearn(df, title='BIC and AIC Scores for Different Number of GMM Components', save_fig=False, save_folder="", show_plot=False):

    X = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    max_components = 10
    n_components_range, bic_scores, aic_scores = calculate_bic_aic(X, max_components)

    # Plotting BIC and AIC scores
    plt.figure(figsize=(20, 10))
    plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
    plt.plot(n_components_range, aic_scores, label='AIC', marker='o')
    plt.xlabel('Number of Components', fontsize=GLOBAL_FONTSIZE)
    plt.ylabel('Score', fontsize=GLOBAL_FONTSIZE)
    plt.title(title, fontsize=GLOBAL_FONTSIZE)
    plt.legend(fontsize=GLOBAL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE) 

    if save_fig : save_one_figure(save_folder, title)
    if show_plot: plt.show()

#### Functions to calculate KL divergence 
def calculate_KL(df1, df2, n=2, save_to_file=False, save_folder=""):

    X1 = np.column_stack((df1['HittingFlux'].values, df1['DistanceTraveled'].values))
    X2 = np.column_stack((df2['HittingFlux'].values, df2['DistanceTraveled'].values))

    # Fit Gaussian Mixture Models
    gmm1 = GaussianMixture(n_components=n, random_state=0, tol=1e-5, max_iter=1000, init_params='k-means++')
    gmm2 = GaussianMixture(n_components=n, random_state=0, tol=1e-5, max_iter=1000, init_params='k-means++')
    gmm1.fit(X1)
    gmm2.fit(X2)

    # Calculate the KL divergence between gmm1 and gmm2
    kl_div_resampled = kl_divergence(gmm1, gmm2, n_samples=1000)
    kl_div_resampled_inv = kl_divergence(gmm2, gmm1, n_samples=1000)
    kl_div = kl_divergence(gmm1, gmm2, X1)
    kl_div_inv = kl_divergence(gmm2, gmm1, X2)

    print(f"\nKL Divergence: {kl_div} \n")
    print(f"KL Divergence Resampled: {kl_div_resampled} \n")
    print(f"KL Divergence Inverse: {kl_div_inv} \n")
    print(f"KL Divergence Resampled Inverse: {kl_div_resampled_inv} \n")

    if save_to_file :
        fn = f"KL_div={kl_div:.4f}.txt"
        save_dir = PATH_TO_DATA_FOLDER + "figures/" + save_folder

        # Create and write to the text file
        with open(os.path.join(save_dir, fn), 'w') as file:
            file.write("This file contains the KL Divergence value.\n")
            file.write(f"KL Divergence: {kl_div}\n")
            file.write(f"KL Divergence Resampled: {kl_div_resampled}\n")
            file.write(f"KL Divergence Inverse: {kl_div_inv}\n")
            file.write(f"KL Divergence Resampled Inverse: {kl_div_resampled_inv}\n")


        print(f"File '{fn}' created successfully.")

def kl_divergence(gmm_p, gmm_q, X=None, n_samples=1000):
    # Sample points from gmm_p
    if X is None:
        X, _ = gmm_p.sample(n_samples)

    # Compute log probabilities under both gmm_p and gmm_q
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    
    # Calculate the KL divergence
    kl_div = np.mean(log_p_X - log_q_X)
    return kl_div

#### Cross validation
def cross_validate_gmm(dataset_name='D1-robot_agnostic', predict_value="flux", n_gaussians=2, n_folds=10):
    
    ## DATASET TO USE 
    df = read_airhockey_csv(fn=dataset_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/for_paper/")
    
    X = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    for i in range(n_folds):

        ## Randomly sample X to get X_train and X_test
        indices_train = np.random.choice(X.shape[0], size=int(0.7*len(X)), replace=False)
        indices_test = [i for i in range(0, len(X)) if i not in indices_train]
        X_train = X[indices_train]
        
        if predict_value == "flux":
            X_test = X[indices_test]
            X_test_for_predict = X_test[:, 0] # flux
            X_test_for_error = X_test[:, 1] # distance
            predict_axis = np.array([0])

        elif predict_value == "distance":
            X_test = X[indices_test]
            X_test_for_predict = X_test[:, 1] # distance
            X_test_for_error = X_test[:, 0] # flux
            predict_axis = np.array([1])
        
        ## Create GMM
        gmm = GMM(n_components=n_gaussians, random_state=0)
        gmm.from_samples(X_train, R_diff=1e-5, n_iter=1000, init_params='kmeans++')

        Y = gmm.predict(predict_axis, X_test_for_predict[:, np.newaxis])

        ## Compare predict with actual data 
        rms_error = np.sqrt(np.mean((Y[:,0] - X_test_for_error)**2))
        rms_error_relative = np.sqrt(np.mean(((Y[:,0] - X_test_for_error)/X_test_for_error)**2)) * 100

        if predict_value == "flux" : print(f"RMS Error : {rms_error*100:.2f} cm")
        elif predict_value == "distance": print(f"RMS Error : {rms_error*100:.2f} cm/s")
        print(f"RMS Error Relative: {rms_error_relative:.2f} % \n")


## NOTE - used to confirm sklearn works
def plot_gmm_with_sklearn(df, show_plot=False):
    
    X = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    # Fit Gaussian Mixture Model
    n_components = 2  # Example: 3 components
    gmm = GaussianMixture(n_components=n, random_state=0, tol=1e-5, max_iter=1000, init_params='k-means++')
    gmm.fit(X)

    # Predict Y for X_test
    # X_test = np.linspace(0.45, 1.1, 100).reshape(-1, 1)
    # Y = gmm.predict(X_test)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.title("GMM fit")
    plt.xlabel("Hitting Flux [m/s]")
    plt.ylabel("Distance travelled [m]")

    plt.scatter(X[:, 0], X[:, 1])

    # Define colors for the ellipses
    colors_list = plt.cm.viridis(np.linspace(0, 1, n_components))

    # Plot error ellipses
    for n, color in enumerate(colors_list):
        mean = gmm.means_[n]
        covariances = gmm.covariances_[n]

        if covariances.shape == (2, 2):
            covariances = covariances
        else:
            covariances = np.diag(covariances)

        v, w = np.linalg.eigh(covariances)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi

        ell = patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=0.12)
        plt.gca().add_patch(ell)

    # plt.plot(X_test, Y, c="k", lw=2)
    if show_plot: plt.show()

### Pre-made functions to reproduce plots 
def object_agnostic(use_clean_dataset = True):

    #### SAVE CLEAN DATASET TO THIS FOLDER
    clean_dataset_folder = "KL_div-object_agnostic"

    new_dataset_name = "D1-D2-object_agnostic"

    ## Datasets to use
    csv_fn = "D1_clean" 
    csv_fn2 = "D2_clean"

    ############ USING RAW DATASETS ############
    if not use_clean_dataset : 

        ## Read and clean datasets
        clean_df = read_and_clean_data(csv_fn, resample=False, n_samples=2000, only_7=True, save_folder=clean_dataset_folder, save_clean_df=True)
        clean_df2 = read_and_clean_data(csv_fn2, resample=True, n_samples=800, only_7=True, save_folder=clean_dataset_folder, save_clean_df=True)

        ## combine both into one df
        df_combined = restructure_for_agnostic_plots(clean_df, clean_df2, resample=False, parameter="object", 
                                                     dataset_name=new_dataset_name, save_folder=clean_dataset_folder, save_new_df=True)

    ######### USING RESAMPLED AND CLEAN DATASETS ###########
    # NOTE - use these to reproduce plots for paper
    else :
        df_combined = read_airhockey_csv(fn=new_dataset_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{clean_dataset_folder}/")
        
        ## Separate per config
        clean_df = df_combined[df_combined['object']==1].copy()
        clean_df2 = df_combined[df_combined['object']==2].copy()
    
    print(f"Dataset info : \n"
        f" Object 1 points : {len(clean_df.index)} \n"
        f" Object 2 points : {len(clean_df2.index)} \n")

    ######### GMM Scores #########
    n_gaussians=2

    plot_gmr(clean_df, n=n_gaussians, plot="only_gmm", title=f"GMM fit for {csv_fn}", save_folder=clean_dataset_folder, save_fig=True)
    plot_gmr(clean_df2, n=n_gaussians, plot="only_gmm", title=f"GMM fit for {csv_fn2}", save_folder=clean_dataset_folder, save_fig=True)
    calculate_KL(clean_df, clean_df2, n=n_gaussians, save_folder=clean_dataset_folder, save_to_file=True)
    plot_bic_aic_with_sklearn(clean_df, title=f"AIC-BIC for {csv_fn}", save_folder=clean_dataset_folder, save_fig=True)
    plot_bic_aic_with_sklearn(clean_df2, title=f"AIC-BIC for {csv_fn2}", save_folder=clean_dataset_folder, save_fig=True)

    plt.show()

def robot_agnostic(use_clean_dataset = True):
    
    #### SAVE CLEAN DATASET + FIGURES TO THIS FOLDER
    clean_dataset_folder = "KL_div-robot_agnostic-D1"

    new_dataset_name = "D1-robot_agnostic"

    ## Datasets to use
    csv_fn = "D1_clean" 

    ############ USING RAW DATASETS ############
    if not use_clean_dataset : 

        ## Read and clean datasets
        clean_df = read_and_clean_data(csv_fn, resample=False, min_flux=0.55, max_flux=0.8, save_folder=clean_dataset_folder, save_clean_df=False)

        df_iiwa7 = clean_df[clean_df['IiwaNumber']==7].copy()
        df_iiwa14 = clean_df[clean_df['IiwaNumber']==14].copy()

        # df_iiwa7 = resample_uniformally(df_iiwa7, n_samples=2000)

        ## combine both into one df and save to data folder
        df_combined = restructure_for_agnostic_plots(df_iiwa7, df_iiwa14, resample=False, 
                                                        dataset_name=new_dataset_name, save_folder=clean_dataset_folder, save_new_df=True)
      
    ######### USING RESAMPLED AND CLEAN DATASETS ###########
    # NOTE - use these to reproduce plots for paper
    else :
        # clean_df = read_airhockey_csv(fn=f"{csv_fn}_clean", folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{clean_dataset_folder}/")
        clean_df = read_airhockey_csv(fn=new_dataset_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{clean_dataset_folder}/")
        
        df_iiwa7 = clean_df[clean_df['IiwaNumber']==7].copy()
        df_iiwa14 = clean_df[clean_df['IiwaNumber']==14].copy()


    print(f"Dataset info : \n"
        f" Iiwa 7 points : {len(df_iiwa7.index)} \n"
        f" Iiwa 14 points : {len(df_iiwa14.index)} \n")

    ######### GMM Scores #########
    n_gaussians = 2

    plot_gmr(df_iiwa7, n=n_gaussians, plot="only_gmm", title=f"GMM fit for IIWA 7", save_folder=clean_dataset_folder, save_fig=True)
    plot_gmr(df_iiwa14, n=n_gaussians, plot="only_gmm", title=f"GMM fit for IIWA 14", save_folder=clean_dataset_folder, save_fig=True)
    plot_gmr(clean_df, n=n_gaussians, plot="only_gmm", title=f"GMM fit for D1", save_folder=clean_dataset_folder, save_fig=True)
    calculate_KL(df_iiwa7, df_iiwa14, n=n_gaussians, save_folder=clean_dataset_folder, save_to_file=True)
    plot_bic_aic_with_sklearn(df_iiwa7, title=f"AIC-BIC for IIWA 7", save_folder=clean_dataset_folder, save_fig=True)
    plot_bic_aic_with_sklearn(df_iiwa14, title=f"AIC-BIC for IIWA 14", save_folder=clean_dataset_folder, save_fig=True)
    plot_bic_aic_with_sklearn(clean_df, title=f"AIC-BIC for D1", save_folder=clean_dataset_folder, save_fig=True)

    plt.show()

def config_agnostic(use_clean_dataset=True):

    #### SAVE CLEAN DATASET TO THIS FOLDER
    clean_dataset_folder = "KL_div-config_agnostic"

    new_dataset_name = "D1-D3-config_agnostic"

    ## Datasets to use
    csv_fn = "D1_clean" 
    csv_fn2 = "D3_clean"

    ############ USING RAW DATASETS ############
    if not use_clean_dataset : 

        ## Read and clean datasets
        clean_df = read_and_clean_data(csv_fn, resample=False, max_flux=0.8, only_7=True, save_folder=clean_dataset_folder, save_clean_df=False)
        clean_df2 = read_and_clean_data(csv_fn2, resample=False, max_flux=0.8, only_7=True, save_folder=clean_dataset_folder, save_clean_df=False)

        # use only data from same day recording 
        clean_df = clean_df[clean_df['RecSession']=="2024-05-29_15:40:48__clean_paper"].copy()
        clean_df2 = clean_df2[clean_df2['RecSession']=="2024-05-29_14:30:29__clean"].copy()
        
        ## combine both into one df and save to data folder
        df_combined = restructure_for_agnostic_plots(clean_df, clean_df2, resample=False, parameter="config", 
                                                     dataset_name=new_dataset_name, save_folder=clean_dataset_folder, save_new_df=True)


    ######### USING RESAMPLED AND CLEAN DATASETS ###########
    # NOTE - use these to reproduce plots for paper
    else :
        df_combined = read_airhockey_csv(fn=new_dataset_name, folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{clean_dataset_folder}/")

        ## Separate per config
        clean_df = df_combined[df_combined['config']==1].copy()
        clean_df2 = df_combined[df_combined['config']==2].copy()
        
    
    print(f"Dataset info : \n"
        f" Config 1 points : {len(clean_df.index)} \n"
        f" Config 2 points : {len(clean_df2.index)} \n")

    ######### GMM Scores #########
    n_gaussians=2

    plot_gmr(clean_df, n=n_gaussians, plot="only_gmm", title=f"GMM fit for {csv_fn}", save_folder=clean_dataset_folder, save_fig=True)
    plot_gmr(clean_df2, n=n_gaussians, plot="only_gmm", title=f"GMM fit for {csv_fn2}", save_folder=clean_dataset_folder, save_fig=True)
    calculate_KL(clean_df, clean_df2, n=n_gaussians, save_folder=clean_dataset_folder, save_to_file=True)
    plot_bic_aic_with_sklearn(clean_df, title=f"AIC-BIC for {csv_fn}", save_folder=clean_dataset_folder, save_fig=True)
    plot_bic_aic_with_sklearn(clean_df2, title=f"AIC-BIC for {csv_fn2}", save_folder=clean_dataset_folder, save_fig=True)

    plt.show()


if __name__== "__main__" :

    ## Run one of these to get the plots for agnosticism 

    # object_agnostic(use_clean_dataset=True)
    # robot_agnostic(use_clean_dataset=True)
    # config_agnostic(use_clean_dataset=True)

    cross_validate_gmm('D1-robot_agnostic', predict_value="distance")