from sklearn.cross_decomposition import CCA
import numpy as np
import scipy.linalg as LA


def train_save_cca(img_features,word_features,n_components,name):
    img_feat = np.array(img_features.values())
    word_feat = np.array(word_features.values())

    d1 = word_feat.shape[1]

    S, S_D = covariance_matrix(word_feat,img_feat)
    print d1, S.shape, S_D.shape
    W, D = solve(S,S_D,n_components,5)

    W_T = W[:d1,:]
    W_I = W[d1:,:]
    D_W = D[:d1,:d1]
    D_I = D[d1:,d1:]

    cca_params = {}
    cca_params["S"] = S
    cca_params["S_D"] = S_D
    cca_params["D"] = D
    cca_params["W"] = W
    cca_params["d1"] = d1
    np.save(name,cca_params)
    return


class perso_cca():
    def __init__(self,S,S_D,D,W,d1):
        self.S = S.real
        self.S_D = S_D.real
        self.W_T = (W.real)[:d1,:]
        self.W_I = (W.real)[d1:,:]
        if np.any(D!=D.real):
            print "oops"
        self.D = D.real

    def word_transform(self,word_vector):
        return np.dot(word_vector,self.W_T)

    def img_transform(self,img_vector):
        return np.dot(img_vector,self.W_I)

    ## similarity function here
    def sim(self,word_vector,img_vector):
        latent_word = self.word_transform(word_vector)
        latent_img = self.img_transform(img_vector)
        result = np.dot(np.dot(latent_word,self.D),np.dot(latent_img,self.D).T)
        result /= np.linalg.norm(np.dot(latent_word,self.D))
        result /= np.linalg.norm(np.dot(latent_img,self.D))
        return result

    def sim_I2T(self,img_vector,word_vector):
        latent_word = self.word_transform(word_vector)
        latent_img = self.img_transform(img_vector)
        result = np.dot(np.dot(latent_word,self.D),np.dot(latent_img,self.D).T)
        result /= np.linalg.norm(np.dot(latent_word,self.D))
        result /= np.linalg.norm(np.dot(latent_img,self.D))
        return result

def covariance_matrix(word_features,img_features):
    # word_features and img_features are np arrays
    assert (len(word_features)==len(img_features)), "number of dimensions mismatch"
    n_obs = word_features.shape[0]

    d1 = word_features.shape[1]
    d2 = img_features.shape[1]

    S = np.zeros((d1+d2,d1+d2))
    S_D = np.zeros((d1+d2,d1+d2))

    S[:d1,:d1] = np.dot(word_features.T,word_features)
    S[d1:,d1:] = np.dot(img_features.T,img_features)

    S_D[:d1,:d1] = np.dot(word_features.T,word_features)
    S_D[d1:,d1:] = np.dot(img_features.T,img_features)

    S[d1:,:d1] = np.dot(img_features.T,word_features)
    S[:d1,d1:] = np.dot(word_features.T,img_features)

    return S, S_D

def solve(S,S_D,n_components,p):
    S = S + 0.0001 * np.eye(len(S))
    S_D = S_D + 0.0001 * np.eye(len(S_D))

    eigval, eigvec = LA.eig(S,S_D)

    # get the first n_components eigenvectors

    idx = eigval.argsort()[::-1][:n_components]
    #print idx

    ## BUILD W AND D
    #print eigval[idx]
    D = np.diag(eigval[idx]**p)
    W = eigvec[:, idx]

    return W, D
