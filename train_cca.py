import numpy as np
from lib import cca
import argparse
import os

def parseArguments():
    parser = argparse.ArgumentParser(description="train and save a CCA module")


    parser.add_argument('path_to_train_img_feat',
        type=str, help="path to train image features (.npy)")

    parser.add_argument('path_to_train_word_feat',
        type=str, help="path to train word features (.npy)")

    parser.add_argument('output',
        type=str, help="path to output cca object folder")

    args = parser.parse_args()
    return args

def main():

    args = parseArguments()
    path_to_train_img_feat = args.path_to_train_img_feat
    path_to_train_word_feat = args.path_to_train_word_feat
    output = args.output


    train_word = np.load(path_to_train_word_feat).item()
    train_img = np.load(path_to_train_img_feat).item()

    clean_word_vectors(train_word, train_img)

    cca.train_save_cca(train_img,train_word,100,output)

    return


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