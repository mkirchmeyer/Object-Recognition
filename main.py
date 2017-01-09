import numpy as np
from lib import cca
from lib import T2I
from lib import evaluate
from lib import glove
#from lib import cnn
import argparse
import matplotlib.pyplot as plt
import os
import operator
# need to specify path to COCO and import pycocotools
import sys
sys.path.append('~/S1/coco/PythonAPI/')
#sys.path.append("./coco-master/PythonAPI/")
from pycocotools.coco import COCO

def parseArguments():
    parser = argparse.ArgumentParser(description="Test Tag2Image / Image2Tag retrieval")

    parser.add_argument('method',
        type=str, help="T2I or I2T")

    parser.add_argument('path_to_train_img_feat',
        type=str, help="path to train image features (.npy)")

    parser.add_argument('path_to_train_word_feat',
        type=str, help="path to train word features (.npy)")

    parser.add_argument('path_to_test_img_feat',
        type=str, help="path to test image features (.npy)")

    parser.add_argument('path_to_test_word_feat',
        type=str, help="path to test word features (.npy)")

    parser.add_argument('path_to_glove',
        type=str, help="path to glove vectors (.txt)")

    parser.add_argument('coco_json_annotation',
        type=str, help='path to json file')

    parser.add_argument("-o",
        type=str, help="output path for plots")

    parser.add_argument("-t",
        type=str, help="title for plots")
    # parser.add_argument("-t",
    #     type=str, help="input tag")

    # parser.add_argument("-i",
    #    type=str, help="input image path")

    args = parser.parse_args()
    return args

def main():
    args = parseArguments()
    method = args.method
    path_to_train_img_feat = args.path_to_train_img_feat
    path_to_train_word_feat = args.path_to_train_word_feat
    path_to_test_img_feat = args.path_to_test_img_feat
    path_to_test_word_feat = args.path_to_test_word_feat
    path_to_glove = args.path_to_glove
    output_folder = args.o
    title = args.t

    if title is None:
        title = "Precision/Recall"
    if output_folder is None:
        output_folder = 'output'

    train_word = np.load(path_to_train_word_feat).item()
    train_img = np.load(path_to_train_img_feat).item()

    clean_word_vectors(train_word, train_img)

    test_word = np.load(path_to_test_word_feat).item()
    test_img = np.load(path_to_test_img_feat).item()

    # create COCO API instance
    coco = COCO(args.coco_json_annotation)

    print "running evaluation ..."
    if method == 'T2I':
        print "creating CCA instance"
        cca_object = cca.cca(train_img, train_word, 2)
        # get coco test images ground truth categories
        ids = test_img.keys()
        GT = evaluate.id2tag(coco,ids)
        print "creating glove instance"
        glove_object = glove.glove(path_to_glove)
        def t2i(tag):
            return T2I.tag2image(tag,cca_object,glove_object,test_img)
        precision, recall, mAP = evaluate.evaluateROCT2I(coco,t2i,GT)
        print mAP
        plot_curves(precision,recall,mAP,output_folder,title+'.eps',title=title)
    elif method == 'I2T':
        print "creating CCA instance"
        cca_object = cca.cca(train_word, train_img, 2)
        # get coco test images ground truth categories
        ids = test_word.keys()
        GT = evaluate.tag2id(coco,ids)
        print "creating cnn instance"
        cnn_object = cnn.cnn()
        def i2t(img):
            return T2I.image2tag(img,cca_object,cnn_object,test_word, coco)
        precision = evaluate.evaluatePrecisionI2T(coco,i2t,GT)
        print precision

def plot_curves(precision,recall,mAP,output_folder,output_name,title='Precision/Recall curves'):
    best_instances = dict(sorted(mAP.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot(111)
    plt.grid()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    for key in best_instances.keys():
        plt.plot(recall[key],precision[key],label='%s: %.3f mAP'%(key,best_instances[key]))

    # Shrink current axis by 20%
    plt.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})

    plt.savefig(os.path.join(output_folder,output_name))
    print "Figure saved at %s" % os.path.join(output_folder,output_name)


def plot_ROC(precision,recall,title='ROC',output_folder='output',output_name='figure'):
    plt.figure()
    plt.plot(recall,precision,'r')
    plt.grid()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.show()
    plt.savefig(os.path.join(output_folder,output_name))

def clean_word_vectors(word_feat_dict, img_feat_dict):
    # need to remove zero vectors which correspond to no annotation in the image
    counter = 0
    for key in word_feat_dict.keys():
        if np.sum(np.array(word_feat_dict[key]) != 0) == 0:
            word_feat_dict.pop(key)
            img_feat_dict.pop(key)
            counter += 1
    print "in clean_word_vectors: %d poppped ids" % counter

if __name__ == '__main__':
    main()
