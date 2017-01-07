"""
n training images each of which is associated with a v-dimensional
visual feature vector and a t-dimensional tag feature vector
"""

featuresDir = '/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/'
extractImageFeatures('val2014')
extractWordFeatures()
imgFeatures = np.load(featuresDir + 'imgFeatures.npy')
wordFeatures = np.load(featuresDir + 'wordFeatures.npy')
CCA(imgFeatures,wordFeatures)
