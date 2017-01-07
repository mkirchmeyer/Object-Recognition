from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import skimage.io as io

dataDir = '/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/coco-master/'
dataType = 'val2014'

def load_train(path_train_img_features, path_train_word_features):
    cat_feat = np.load(path_train_word_features).item()
    img_feat = np.load(path_train_img_features).item()

def train_CCA(img_feat, cat_feat):
    cca = CCA(n_components=1)
    cca.fit(img_feat, cat_feat)

def display_top(tag, score_dict, number):
    ids = score_dict.keys()
    score = score_dict.values()
    index = np.argpartition(score,-number)[-number:]
    for i in range(len(index)):
        I = io.imread(dataDir + '/images/' + dataType + '/' + 'COCO_' + dataType + '_%012d' % ids[index[i]] + '.jpg')
        nb = i + 1
        io.imsave(tag + '_top%d' %nb, I)
