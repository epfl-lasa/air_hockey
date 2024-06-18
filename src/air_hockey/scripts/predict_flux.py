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

def get_flux_for_distance_with_gmr(distance):

    des_distances = np.array([distance])

    # Predict fluxes for desired distances
    flux = gmm.predict(np.array([1]), des_distances[:, np.newaxis])

    return flux[0,0]

def handle_calculation(req):
    rospy.loginfo(f"Received input: {req.input_value}")
    # Get flxu prediction based on distance 
    result = get_flux_for_distance_with_gmr(req.input_value)
    rospy.loginfo(f"Returning result: {result}")
    return PredictionResponse(result)

def calculation_server():
    rospy.init_node('prediction_server')
    s = rospy.Service('prediction', Prediction, handle_calculation)
    rospy.loginfo("Ready to predict.")
    rospy.spin()

if __name__ == "__main__":

    
    gmm = read_model_for_gmr(model_name='GMM_fit_for_D1')

    calculation_server()