import numpy as np

def search_img(img_feat,img_feats):
    output = {}
    for key in img.feats.keys():
        output[key] = np.dot(img_feat,img_feats[key])
    return output
