import numpy as np
from lib import cca
from lib import T2I
from lib import evaluate
from lib import glove
import argparse
import matplotlib.pyplot as plt
import os

# need to specify path to COCO and import pycocotools
import sys
sys.path.append('~/S1/coco/PythonAPI/')
from pycocotools.coco import COCO


def parseArguments():
    parser = argparse.ArgumentParser(description="Test Tag2Image retrieval")

    parser.add_argument('path_to_train_img_feat',
        type=str, help="path to train img features (.npy)")

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

    # parser.add_argument("-t",
    #     type=str, help="input tag")

    # parser.add_argument("-i",
    #    type=str, help="input image path")

    args = parser.parse_args()
    return args

def main():
    args = parseArguments()
    path_to_train_img_feat = args.path_to_train_img_feat
    path_to_train_word_feat = args.path_to_train_word_feat
    path_to_test_img_feat = args.path_to_test_img_feat
    path_to_test_word_feat = args.path_to_test_word_feat
    path_to_glove = args.path_to_glove
    output_folder = args.o

    if output_folder is None:
        output_folder = 'output'

    train_word = np.load(path_to_train_word_feat).item()
    train_img = np.load(path_to_train_img_feat).item()

    clean_word_vectors(train_word, train_img)

    test_word = np.load(path_to_test_word_feat).item()
    test_img = np.load(path_to_test_img_feat).item()

    print "creating glove instance"
    glove_object = glove.glove(path_to_glove)

    print "creating CCA instance"
    cca_object = cca.cca(train_img, train_word, 2)


    # create COCO API instance
    coco = COCO(args.coco_json_annotation)
    # get coco test images ground truth categories
    ids = test_img.keys()
    GT = evaluate.id2tag(coco,ids)

    def t2i(tag):
        return T2I.tag2image(tag,cca_object,glove_object,test_img)

    # test run on dummy tag
    print "running evaluation ..."
    precision, recall, mAP = evaluate.evaluateROC(coco,t2i,GT)

    print mAP

    #title = "Img feature: %s, Cat feature: %s, mAP: %.2f" % (path_to_test_img_feat.split('_')[0], path_to_test_word_feat.split('_')[0],mAP)
    #plot_ROC(precision, recall, title, output_folder)


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