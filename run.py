"""
n training images each of which is associated with a v-dimensional
visual feature vector and a t-dimensional tag feature vector
"""

featuresDir = './data'
imgFeatures = np.load(featuresDir + 'img_feat/imgFeatures.npy')
wordFeatures = np.load(featuresDir + 'cat_feat/wordFeatures.npy')
CCA = CCA(imgFeatures,wordFeatures,2)
CCA.predict()
