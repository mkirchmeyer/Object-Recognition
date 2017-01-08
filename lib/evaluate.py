import numpy as np
import sys
sys.path.append('/home/struong/Documents/MVA/S1/coco/PythonAPI/')
from pycocotools.coco import COCO

# threshold sampling
thr_array = np.linspace(0,1,100)

# input is a dictionary id:score
def normalize_output(scores):
    total = 0
    for value in scores.values():
        total += value
    for key in scores.keys():
        scores[key] /= float(total)

def spread(scores):
    maxi = max(scores.values())
    mini = min(scores.values())
    for key in scores.keys():
        scores[key] = (scores[key] - mini) / float(maxi-mini)

def softmax(scores):
    # scores is a dict {id:score}
    somme = 0
    for key in scores.keys():
        scores[key] = np.exp(scores[key])
        somme += scores[key]
    for key in scores.keys():
        scores[key] /= float(somme)

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

def evaluateROCT2I(coco,tag2img,GT):
    # coco is the COCO API instance that was constructed from the categories .json
    # tag2img is the function returning the scores on the test images
    # imgs_test are the coco images on which evaluation is performed
    # (be careful, the ids returned by tag2img and imgs_test have to correspond)
    # loop over possible category tags of the coco API instance
    # why not put them all in a same object named imgQueryObject
    precision = {}
    recall = {}
    mAP = {}

    cats = coco.loadCats(coco.getCatIds())

    precision_array = np.zeros(len(thr_array))
    recall_array = np.zeros(len(thr_array))

    # loop over possible queries
    counter = 0
    for k in range(len(cats)):
        cat = cats[k]
        query = cat['name']

        print "\r>> Computing retrieval statistics for \'%s\' (%d/%d)               " % (query,k,len(cats)) ,
        sys.stdout.flush()

        # check images of this instance exist in the test set
        if not len({idx:categories for idx, categories in GT.iteritems() if (query in categories)}):
            print "no \'%s\' images in test set" % query
            continue

        result = tag2img(query) # dictionnary id:score
        softmax(result)
        spread(result)

        pos_array = np.zeros(len(thr_array))
        true_pos_array = np.zeros(len(thr_array))

        for thr_index in range(len(thr_array)):
            thr = thr_array[thr_index]
            filtered_result = {idx: score for idx, score in result.iteritems() if score >= thr}
            for idx in filtered_result.keys():
                pos_array[thr_index] += 1
                if query in GT[idx]:
                    true_pos_array[thr_index] += 1

        precision[query] = true_pos_array / pos_array
        recall[query] = true_pos_array / len({idx:categories for idx, categories in GT.iteritems() if (query in categories)})
        mAP[query] = computeAP(precision[query],recall[query])
        counter += 1
    print '\nComputed all queries!'
    # take the mean of all ROC curves over all possible queries

    return precision, recall, mAP

def evaluateROCI2T(coco,img2tag,GT):
    # coco is the COCO API instance that was constructed from the categories .json
    # tag2img is the function returning the scores on the test tags
    # imgs_test are the coco images on which evaluation is performed
    # (be careful, the ids returned by tag2img and imgs_test have to correspond)
    # loop over possible category tags of the coco API instance
    # why not put them all in a same object named imgQueryObject
    precision = {}
    recall = {}
    mAP = {}

    cats = coco.loadCats(coco.getCatIds())

    precision_array = np.zeros(len(thr_array))
    recall_array = np.zeros(len(thr_array))

    # loop over possible queries
    counter = 0
    for k in range(len(cats)):
        cat = cats[k]
        query = cat['name']

        print "\r>> Computing retrieval statistics for \'%s\' (%d/%d)               " % (query,k,len(cats)) ,
        sys.stdout.flush()

        # check images of this instance exist in the test set
        if not len({idx:categories for idx, categories in GT.iteritems() if (query in categories)}):
            print "no \'%s\' images in test set" % query
            continue

        result = tag2img(query) # dictionnary id:score
        softmax(result)
        spread(result)

        pos_array = np.zeros(len(thr_array))
        true_pos_array = np.zeros(len(thr_array))

        for thr_index in range(len(thr_array)):
            thr = thr_array[thr_index]
            filtered_result = {idx: score for idx, score in result.iteritems() if score >= thr}
            for idx in filtered_result.keys():
                pos_array[thr_index] += 1
                if query in GT[idx]:
                    true_pos_array[thr_index] += 1

        precision[query] = true_pos_array / pos_array
        recall[query] = true_pos_array / len({idx:categories for idx, categories in GT.iteritems() if (query in categories)})
        mAP[query] = computeAP(precision[query],recall[query])
        counter += 1
    print '\nComputed all queries!'
    # take the mean of all ROC curves over all possible queries

    return precision, recall, mAP

#def top_five_precision(coco,tag2Img,imgs_test):
#    ids = np.zeros(len(imgs_test))
#    for k in range(len(imgs_test)):
#        ids[k] = imgs_test[k]['id']
    #ids = []
    #for img in imgs_test:
    #    ids.append(img['id'])

#    GT = id2tag(coco,ids) # ground truth tags per image index

def computeAP(precision_array, recall_array):
    assert precision_array.shape == recall_array.shape, "in compute AP, shape mismatch"

    result = 0
    for k in range(len(precision_array)-1):
        result += precision_array[k] * (recall_array[k]-recall_array[k+1])
    assert result <= 1 and result >= 0, "in computeAP, result non consistant"

    return result
