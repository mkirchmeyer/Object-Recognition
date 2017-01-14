import numpy as np
from lib import cca
from lib import T2I
import sys
from main import clean_word_vectors
sys.path.append('../coco/PythonAPI/')
#sys.path.append("./coco-master/PythonAPI/")
from pycocotools.coco import COCO
from lib import evaluate
from lib import glove
import argparse
import matplotlib.pyplot as plt

def parseArguments():
    parser = argparse.ArgumentParser(description="Test Tag2Image / Image2Tag retrieval")

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

    print "creating CCA instance"
    cca_object = cca.cca(train_img, train_word, 2)

    # create COCO API instance
    coco = COCO(args.coco_json_annotation)
    imgs = coco.loadImgs(coco.getImgIds())
    # create id to file name dict
    imgsNames = [img['file_name'] for img in imgs]
    imgsIds = [img['id'] for img in imgs]

    dict_id_filename = {}
    for k in range(len(imgsIds)):
        dict_id_filename[imgsIds[k]] = imgsNames[k]

    glove_object = glove.glove(path_to_glove)


    def t2i(tag):
        return T2I.tag2image(tag,cca_object,glove_object,test_img)

    #tags = ["dog", "person", "zebra", "blue car", "car", "frisbee", "skateboard", "dining table", "dolphin", "tennis", "running"]
    tags = ["antelope", "skiing", "snow", "rain", "shoes", "donut", "food", "face", "sport"]

    for tag in tags:
        print "retrieving from tag: "+tag
        evaluate.retrieve(t2i, dict_id_filename, tag, output_folder)

if __name__ == '__main__':
    main()
