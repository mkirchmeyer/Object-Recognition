from sklearn.cross_decomposition import CCA

class cca():
    def __init__(self,img_features, word_features,n_components):
        self.cca = CCA(n_components)
        cat_feat = word_features.values()
        img_feat = img_features.values()
        (self.cca).fit(img_feat, cat_feat)

    def predict(self,word_vector):
        return self.cca.predict(word_vector)
"""
Example of CCA

from sklearn.cross_decomposition import CCA
X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
X_dummy = [0.,5.,100,]
cca = CCA(n_components=1)
cca.fit(X, Y)

print X
print Y

X_c, Y_c = cca.transform(X, Y)
X_dummy_c = cca.predict(X_dummy)
print X_c
print Y_c
print X_dummy_c
"""
