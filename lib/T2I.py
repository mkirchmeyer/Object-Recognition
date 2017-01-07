import matplotlib.pyplot as plt
import skimage.io as io

dataDir = './coco-master/images/'
dataType = 'val2014'

def display_top(tag, score_dict, number):
"""
Display top number images
"""
    ids = score_dict.keys()
    score = score_dict.values()
    index = np.argpartition(score,-number)[-number:]
    for i in range(len(index)):
        I = io.imread(dataDir + dataType + '/COCO_' + dataType + '_%012d' % ids[index[i]] + '.jpg')
        nb = i + 1
        io.imsave('output/' + tag + '_top%d' %nb, I)
