from gmr import MVN, GMM, plot_error_ellipses
import h5py
import os
import numpy as np
import rospy
from air_hockey.srv import Prediction, PredictionResponse

PATH_TO_DATA_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + "/data/"
GMM_FOLDER = PATH_TO_DATA_FOLDER + "gaussian_models/"

def read_model_for_gmr(model_name):
    
    # Open the HDF5 file in read mode
    hf_1 = h5py.File(os.path.join(GMM_FOLDER, model_name+".h5"), 'r')

    # Read the parameters
    n_components = hf_1['n_components'][()]
    means = hf_1['means'][()]
    covariances = hf_1['covariances'][()]
    weights = hf_1['weights'][()]

    # Close the HDF5 file
    hf_1.close()

    # Put into gmr model
    gmm = GMM(n_components=n_components, means=means, covariances=covariances, priors=weights, random_state=0)

    return gmm

def get_flux_for_distance_with_gmr(d1, d2):

    des_distances = np.array([d1, d2])

    # Predict fluxes for desired distances
    flux_iiwa7 = gmm_iiwa7.predict(np.array([1]), des_distances[:, np.newaxis])
    flux_iiwa14 = gmm_iiwa14.predict(np.array([1]), des_distances[:, np.newaxis])

    return flux_iiwa7[0,0], flux_iiwa14[1,0]

def handle_prediction(req):
    rospy.loginfo(f"Received input: {req.distance_iiwa7},{req.distance_iiwa14}")
    # Get flxu prediction based on distance 
    flux_1, flux_2 = get_flux_for_distance_with_gmr(req.distance_iiwa7, req.distance_iiwa14)
    rospy.loginfo(f"Returning result: {flux_1}, {flux_2}")
    return PredictionResponse(flux_1, flux_2)

def prediction_server():
    rospy.init_node('prediction_server')
    s = rospy.Service('prediction', Prediction, handle_prediction)
    rospy.loginfo("Ready to predict.")
    rospy.spin()

if __name__ == "__main__":

    gmm_iiwa7 = read_model_for_gmr(model_name='GMM_fit_for_complete_D1-iiwa_7')
    gmm_iiwa14 = read_model_for_gmr(model_name='GMM_fit_for_complete_D1-iiwa_14')

    prediction_server()