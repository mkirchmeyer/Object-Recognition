import numpy as np
import sys
#sys.path.append('/home/struong/Documents/MVA/S1/coco/PythonAPI/')
sys.path.append('/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/coco-master/PythonAPI/')
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
        name_area = {}
        for ann in anns:
            instance = coco.loadCats(ann['category_id'])[0]['name']
            name_area[instance] = ann['area']

        name_area = dict(sorted(name_area.iteritems(), key=operator.itemgetter(1), reverse=True))

        result[idx] = name_area.keys()
        #for ann in anns:
        #    instance = coco.loadCats(ann['category_id'])[0]['name']
        #    strings.append(instance)
        #result[idx] = strings
    return result

def tag2id(coco,ids):
    # returns dict cat:str_array of id
    result = {}
    cats = []
    for idx in ids:
        annIds = coco.getAnnIds(imgIds=idx)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            instance = coco.loadCats(ann['category_id'])[0]['name']
            cats.append(instance)
    cats = np.unique(cats)
    for cat in cats:
        catId = coco.getCatIds(catNms = cat)
        imIds = coco.getImgIds(catIds = catId)
        strings = []
        for imId in imIds:
            if imId in ids:
                instance = coco.loadImgs(imId)[0]['id']
                strings.append(instance)
        result[cat] = strings
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

    # loop over possible queries
    counter = 0
    for k in range(len(cats)):
        cat = cats[k]
        query = cat['name']

        print "\r>> Computing retrieval statistics for \'%s\' (%d/%d)               " % (query,k,len(cats)) ,
        sys.stdout.flush()

        # check images of this instance exist in the test set
        if not len({idx:categories for idx, categories in GT.iteritems() if (query in categories[:2])}):
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
                if query in GT[idx][:2]:
                    true_pos_array[thr_index] += 1

        precision[query] = true_pos_array / pos_array
        recall[query] = true_pos_array / len({idx:categories for idx, categories in GT.iteritems() if (query in categories[:2])})
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

def evaluatePrecisionI2T(coco,img2tag,test_img,GT):
    # coco is the COCO API instance that was constructed from the categories .json
    # img2tag is the function returning the scores on the test tags
    # tags_test are the coco tags on which evaluation is performed
    # (be careful, the ids returned by img2tag and tags_test have to correspond)
    # loop over possible image names of the coco API instance
    precision = {}

    imgs = coco.loadImgs(coco.getImgIds())
    imgsNames = [img['file_name'] for img in imgs]
    imgsIds = [img['id'] for img in imgs]

    true_pos = 0

    # loop over possible queries
    for k in range(len(imgsIds)):
        query = imgsIds[k]

        print "\r>> Computing retrieval statistics for \'%s\' (%d/%d)               " % (query,k,len(imgsIds)),
        sys.stdout.flush()

        # check if tags of this instance exist in the test set
        if not len({categories:idx for categories, idx in GT.iteritems() if (query in idx)}):
            continue

        result = img2tag(test_img[query]) # dictionnary cat:score

        #Display top number tags
        cats = result.keys()
        score = result.values()
        index = np.argsort(score)
        top_cat = []
        for i in range(5):
            top_cat.append(cats[index[len(index)-1-i]])

        for cat in top_cat:
            if query in GT[cat]:
                true_pos += 1
                break

    precision = float(true_pos) / len(test_img)

    print '\nComputed all queries!'

    return precision
