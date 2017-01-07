import numpy as np
import sys
from pycocotools.coco import COCO

# threshold sampling
thr_array = np.linspace(0,1,100)

# input is a dictionary id:score
def normalize_output(scores):
    total = 0
    for key in scores.keys():
        total += scores[key]
    for key in scores.keys():
        scores[key] /= float(total)

def id2tag(coco,ids):
    # returns dict id:str_array
    result = {}
    for idx in ids:
        annIds = coco.getAnnIds(imgIds=idx)
        anns = coco.loadAnns(annIds)
        strings = []
        for ann in anns:
            instance = coco.loadCats(ann['category_id'])[0]['name']
            strings.append(instance)
        result[idx] = strings
    return result

def evaluateROC(coco,tag2img,GT):
    # coco is the COCO API instance that was constructed from the categories .json
    # tag2img is the function returning the scores on the test images
    # imgs_test are the coco images on which evaluation is performed
    # (be careful, the ids returned by tag2img and imgs_test have to be corresponding)
    # loop over possible category tags of the coco API instance
    # why not put them all in a same object named imgQueryObject
    cats = coco.loadCats(coco.getCatIds())

    precision_array = np.zeros(len(thr_array))
    recall_array = np.zeros(len(thr_array))

    # loop over possible queries
    for k in range(len(cats)):
        cat = cats[k]
        query = cat['name']
        result = tag2img(query) # dictionnary id:score
        normalize_output(result)

        pos_array = np.zeros(len(thr_array))
        true_pos_array = np.zeros(len(thr_array))

        for thr in thr_array:
            filtered_result = {idx: score for idx, score in result.iteritems() if score > thr}
            for idx in filtered_result.keys():
                pos_array[thr] += 1
                if query in GT[idx]:
                    true_pos_array[thr] += 1

        precision_array = precision_array + true_pos_array / pos_array
        recall_array = recall_array + true_pos_array / len({idx:categories from idx, categories in GT/iteritems() if (query in categories)})

    # take the mean of all ROC curves over all possible queries
    precision_array = precision_array / len(cats)
    recall_array = recall_array / len(cats)

    return precision_array, recall_array

#def top_five_precision(coco,tag2Img,imgs_test):
#    ids = np.zeros(len(imgs_test))
#    for k in range(len(imgs_test)):
#        ids[k] = imgs_test[k]['id']
    #ids = []
    #for img in imgs_test:
    #    ids.append(img['id'])

#    GT = id2tag(coco,ids) # ground truth tags per image index


def computeAP(precision_array, recall_array):
    assert precision_array.shape == recall.array.shape, "in compute AP, shape mismatch"

    result = 0
    for k in range(len(precision_array)-1):
        result += precision_array[k] * (recall_array[k+1] - recall_array[k])
    assert result <= 1 and result >= 0, "in computeAP, result non consistant"

    return result
