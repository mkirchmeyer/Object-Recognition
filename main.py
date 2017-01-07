import numpy as np
from lib import cca
from lib import T2I
from lib import evaluate
from lib import glove
import argparse
import matplotlib.pyplot as plt

# need to specify path to COCO and import pycocotools

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

    parser.add.argument('coco_json_annotation',
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

    train_word = np.load(path_to_train_word_feat).item()
    train_img = np.load(path_to_train_img_feat).item()
    test_word = np.load(path_to_test_word_feat).item()
    test_img = np.load(path_to_test_img_feat).item()
    glove_object = glove(path_to_glove)
    cca_object = cca(train_img, train_word, 2)


    # create COCO API instance
    coco = COCO(args.coco_json_annotation)
    # get coco test images ground truth categories
    ids = test_img.keys()
    GT = evaluate.id2tag(coco,ids)

    def t2i(tag):
        return tag2image(tag,cca_object,glove_object,test_img)

    # test run on dummy tag

    precision, recall = evaluateROC(coco,t2i,GT)
    mAP = computeAP(precision,recall)

    title = "Img feature: %s, Cat feature: %s" % (path_to_test_img_feat.split('_')[0], path_to_test_word_feat.split('_')[:2])
    plot_ROC(precision, recall, title=title, output_folder,"%s_%s.eps"%(path_to_test_img_feat.split('_')[0], path_to_test_word_feat.split('_')[:2]))


def plot_ROC(precision,recall,title='ROC',output_folder='output',output_name='figure'):
    plt.figure()
    plt.plot(recall,precision,'r')
    plt.grid()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.savefig(os.path.join(output_folder,output_name))