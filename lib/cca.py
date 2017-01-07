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
