import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.patches as patches
from gmr import MVN, GMM, plot_error_ellipses

from process_data import  PATH_TO_DATA_FOLDER

from analyse_data import read_airhockey_csv, read_and_clean_data, save_all_figures

# Fontsize for axes and titles 
GLOBAL_FONTSIZE = 30
AXIS_TICK_FONTSIZE = 20

PROCESSED_FOLDER = PATH_TO_DATA_FOLDER + "airhockey_processed/"

## GMM ANALYISIS
def plot_gmr(df, n=3, plot="only_gmm", title="Gaussian Mixture Model fit", show_plot=False):
    
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
        plt.title(title, fontsize=GLOBAL_FONTSIZE)
        plt.xlabel("Hitting Flux [m/s]", fontsize=GLOBAL_FONTSIZE)   
        plt.ylabel("Distance travelled [m]", fontsize=GLOBAL_FONTSIZE) 
        
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
    
    # Increase the size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=AXIS_TICK_FONTSIZE)  # Change 14 to the desired size

    if show_plot: plt.show()

#### Functions to calculate BIC and AIC for a range of components
def calculate_bic_aic(X, max_components):
    bic_scores = []
    aic_scores = []
    n_components_range = range(1, max_components + 1)
    
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
    
    return n_components_range, bic_scores, aic_scores

def plot_bic_aic_with_sklearn(df, title='BIC and AIC Scores for Different Number of GMM Components',  show_plot=False):

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
    if show_plot: plt.show()

#### Functions to calculate KL divergence 
def calculate_KL(df1, df2):

    X1 = np.column_stack((df1['HittingFlux'].values, df1['DistanceTraveled'].values))
    X2 = np.column_stack((df2['HittingFlux'].values, df2['DistanceTraveled'].values))

    # Fit Gaussian Mixture Models
    gmm1 = GaussianMixture(n_components=3, random_state=0)
    gmm2 = GaussianMixture(n_components=3, random_state=0)
    gmm1.fit(X1)
    gmm2.fit(X2)

    # Calculate the KL divergence between gmm1 and gmm2
    kl_div = kl_divergence(gmm1, gmm2)
    print(f"KL Divergence: {kl_div}")

def kl_divergence(gmm_p, gmm_q, n_samples=1000):
    # Sample points from gmm_p
    X, _ = gmm_p.sample(n_samples)
    
    # Compute log probabilities under both gmm_p and gmm_q
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    
    # Calculate the KL divergence
    kl_div = np.mean(log_p_X - log_q_X)
    return kl_div


## NOTE - used to confirm sklearn works, gmm_torch wok
def plot_gmm_with_sklearn(df, show_plot=False):
    
    X = np.column_stack((df['HittingFlux'].values, df['DistanceTraveled'].values))

    # Fit Gaussian Mixture Model
    n_components = 2  # Example: 3 components
    gmm = GaussianMixture(n_components=n_components, random_state=0)
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


if __name__== "__main__" :
    
    #### SAVE CLEAN DATASET TO THIS FOLDER

    clean_dataset_folder = "KL_div"

    use_raw_datasets = True
    

    ############ USING RAW DATASETS ############
    if use_raw_datasets : 
        ## Datasets to use
        csv_fn = "D1_clean" 
        csv_fn2 = "D2_clean"

        ## Read and clean datasets
        clean_df = read_and_clean_data(csv_fn, resample=True, n_samples=2000, only_7=False, save_folder=clean_dataset_folder, save_clean_df=True)
        clean_df2 = read_and_clean_data(csv_fn2, resample=True, n_samples=900, only_7=False, save_folder=clean_dataset_folder, save_clean_df=True)


    ######### USING RESAMPLED AND CLEAN DATASETS ###########
    # NOTE - use these to reproduce plots for paper
    else :
        clean_df = read_airhockey_csv(fn="D1_clean_clean", folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{clean_dataset_folder}/")
        clean_df2 = read_airhockey_csv(fn="D2_clean_clean", folder=PATH_TO_DATA_FOLDER + f"airhockey_processed/clean/{clean_dataset_folder}/")
    

    ######### GMM Scores #########
    plot_gmr(clean_df, n=2, plot="only_gmm", title="Gaussian Mixture Model fit for D1")
    plot_gmr(clean_df2, n=2, plot="only_gmm", title="Gaussian Mixture Model fit for D2")
    calculate_KL(clean_df, clean_df2)
    plot_bic_aic_with_sklearn(clean_df, title='D1 - BIC and AIC Scores for Different Number of GMM Components')
    plot_bic_aic_with_sklearn(clean_df2, title='D2 - BIC and AIC Scores for Different Number of GMM Components')

    save_all_figures(folder_name=clean_dataset_folder)

    plt.show()



