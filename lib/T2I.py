import matplotlib.pyplot as plt
import skimage.io as io
import glove
import cca
import numpy as np

dataDir ='~/coco/images/'
dataDir = '../coco-master/images/'
dataType = 'val2014'

def display_top(tag, score_dict, number):
#Display top number images
    ids = score_dict.keys()
    score = score_dict.values()
    index = np.argpartition(score,-number)[-number:]
    for i in range(len(index)):
        I = io.imread(dataDir + dataType + '/COCO_' + dataType + '_%012d' % ids[index[i]] + '.jpg')
        nb = i + 1
        io.imsave('output/' + tag + '_top%d' %nb, I)

def search_img(img_feat,img_feats):
    output = {}
    for key in img_feats.keys():
        # need to normalize the vectors
        img_feat_tmp = img_feat.copy() / np.linalg.norm(img_feat)
        img_feats_tmp = img_feats[key].copy() / np.linalg.norm(img_feats[key])

        output[key] = np.dot(img_feat_tmp,img_feats_tmp)
    return output


def tag2image(tag, cca_object, glove_object, test_img_feat):
    # retrieve the tag vector
    tag_vector = glove_object.vec_expression(tag)

    # predict with CCA
    img_vector = cca_object.predict(tag_vector)

    # search image
    scores = search_img(img_vector,test_img_feat)

    return scores

