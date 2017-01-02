import numpy as np
import scipy.linalg as linalg

"""
Complete phiV and phiT functions
"""

def phiV(V):
    #complete
    return phi_V

def phiT(T):
    #complete
    return phi_T

def computeCovarianceMatrix(features):
    number_views = len(features)
    dimension = np.zeros((number_views))

    for i in range(number_views):
        dimension[i] = features[i].shape[1]

    S = np.zeros((np.sum(dimensions), np.sum(dimensions)))
    SD = np.zeros((np.sum(dimensions), np.sum(dimensions)))

    for i in range(number_views):
        for j in range(i):
"""
A revoir pas sûr

            S[i:i+dimension[i]+1, j:j+dimension[j]+1] = np.dot(features[i].T, features[j])
        SD[i:i+dimension[i]+1, i:i+dimension[i]+1] = np.dot(features[i].T, features[i])
"""
    S = S + S.T + SD

    return S, SD

def runCCA(S, SD, d, constant):
    """
    d : embedding dimension
    constant : constant added to the diagonal of the covariance matrix

    """
    regularization = constant * np.eye(len(S))
    S = S + regularization
    SD = SD + regularization

"""
Complete
"""

    return 

def CCA(imgFeatures,wordFeatures):
    """
    Perform CCA on image and text features extracted from n training images
    """
    # Visual features
    V = imgFeatures
    phi_V = phiV(V) # embed the image features into a non linear kernel space

    # Tag features
    T = wordFeatures
    phi_T = phiT(T) # embed the tag features into a non linear kernel space

    # Covariance matrix composed of block covariance matrices between the different views
    features = [phi_V, phi_T]
    S, SD = computeCovarianceMatrix(features)

    #
    runCCA(S, SD, 128, 1e-4)
