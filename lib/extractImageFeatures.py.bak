import numpy as np
import os
os.environ['GLOG_minloglevel'] = '2' # only keep warning messages
import sys
sys.path.append("/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/caffe-home/caffe/build/install/python") # path to caffe
import caffe
sys.path.append("/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/coco-master/PythonAPI/") # path to COCO
from pycocotools.coco import COCO

# Load image names through COCO API
print("Loading image names")
dataType = 'val2014'
dataDir = '/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/coco-master/'
annFile = '%sannotations/instances_%s.json' % (dataDir, dataType)
coco = COCO(annFile)
imgs = coco.loadImgs(coco.getImgIds())
imgsNames = [img['file_name'] for img in imgs]
imgsIds = [img['id'] for img in imgs]
nbImages = len(imgs)
print("%d image names loaded" % nbImages)

# Defining CNN variables
print("Defining CNN variables")
projectDir = '/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/'
model_prototxt = projectDir + 'VGG_ILSVRC_19_layers_deploy.prototxt'
model_trained = projectDir + 'VGG_ILSVRC_19_layers.caffemodel'
mean_path = projectDir + 'caffe-home/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

# Loading the Caffe model, setting preprocessing parameters
print("Loading caffe model and configure preprocessing")
caffe.set_mode_cpu()
net = caffe.Net(model_prototxt, model_trained, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)

# Extract features
print("Extract features from %s (%d images)" % (dataType, nbImages))
imgsFeaturesfc6Train = {}
imgsFeaturesfc7Train = {}
imgsFeaturesfc8Train = {}
imgsFeaturesfc6Test = {}
imgsFeaturesfc7Test = {}
imgsFeaturesfc8Test = {}
trainingSize = 5000
testSize = 1000

# Run CNN train
for i in range(trainingSize):
    print("Processing image %d" % (i+1))
    net.blobs['data'].reshape(1, 3, 224, 224) # reshape network blob
    image_path = dataDir + 'images/' + dataType + '/' + imgsNames[i]
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path)) # run the image through the preprocessor
    net.forward() # run the image through the network
    imgsFeaturesfc6Train[imgsIds[i]] = net.blobs['fc6'].data[0].copy()
    imgsFeaturesfc7Train[imgsIds[i]] = net.blobs['fc7'].data[0].copy()
    imgsFeaturesfc8Train[imgsIds[i]] = net.blobs['fc8'].data[0].copy()

print("Saving features train")
np.save('imgsFeaturesfc6Train', imgsFeaturesfc6Train) # extract the feature vector from the layer of interest
np.save('imgsFeaturesfc7Train', imgsFeaturesfc7Train) # extract the feature vector from the layer of interest
np.save('imgsFeaturesfc8Train', imgsFeaturesfc8Train) # extract the feature vector from the layer of interest

# Run CNN test
for i in range(testSize):
    print("Processing image %d" % (i+1+trainingSize))
    net.blobs['data'].reshape(1, 3, 224, 224) # reshape network blob
    image_path = dataDir + 'images/' + dataType + '/' + imgsNames[i+trainingSize]
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path)) # run the image through the preprocessor
    net.forward() # run the image through the network
    imgsFeaturesfc6Test[imgsIds[i+trainingSize]] = net.blobs['fc6'].data[0].copy()
    imgsFeaturesfc7Test[imgsIds[i+trainingSize]] = net.blobs['fc7'].data[0].copy()
    imgsFeaturesfc8Test[imgsIds[i+trainingSize]] = net.blobs['fc8'].data[0].copy()

print("Saving features test")
np.save('imgsFeaturesfc6Test', imgsFeaturesfc6Test) # extract the feature vector from the layer of interest
np.save('imgsFeaturesfc7Test', imgsFeaturesfc7Test) # extract the feature vector from the layer of interest
np.save('imgsFeaturesfc8Test', imgsFeaturesfc8Test) # extract the feature vector from the layer of interest
