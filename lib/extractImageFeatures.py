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
net.blobs['data'].reshape(6000, 3, 224, 224) # reshape network blob

# Extract features
print("Extract features from %s (%d images)" % (dataType, nbImages))

# Run CNN
for i in range(6000):
    print("Processing image %d" % (i+1))
    image_path = dataDir + 'images/' + dataType + '/' + imgsNames[i]
    net.blobs['data'].data[i] = transformer.preprocess('data', caffe.io.load_image(image_path)) # run the image through the preprocessor
print("Forward")
output = net.forward() # run the image through the network

# Save image features
print("Saving features fc6")
imgsFeaturesfc6Train = {}
for i in range(5000):
    imgsFeaturesfc6Train[imgsIds[i]] = net.blobs['fc6'].data[i]
np.save('imgsFeaturesfc6Train', imgsFeaturesfc6Train) # save as .npy # extract the feature vector from the layer of interest

# Train fc7
print("Saving features fc7")
imgsFeaturesfc7Train = {}
for i in range(5000):
    imgsFeaturesfc7Train[imgsIds[i]] = net.blobs['fc7'].data[i]
np.save('imgsFeaturesfc7Train', imgsFeaturesfc7Train) # save as .npy # extract the feature vector from the layer of interest

# Train fc8
print("Saving features fc8")
imgsFeaturesfc8Train = {}
for i in range(5000):
    imgsFeaturesfc8Train[imgsIds[i]] = net.blobs['fc8'].data[i]
np.save('imgsFeaturesfc8Train', imgsFeaturesfc8Train) # save as .npy # extract the feature vector from the layer of interest

# Test fc6
print("Saving features fc6")
imgsFeaturesfc6Test = {}
for i in range(5000, 6000):
    imgsFeaturesfc6Test[imgsIds[i]] = net.blobs['fc6'].data[i]
np.save('imgsFeaturesfc6Test', imgsFeaturesfc6Test) # save as .npy # extract the feature vector from the layer of interest

# Test fc7
print("Saving features fc7")
imgsFeaturesfc7Test = {}
for i in range(5000, 6000):
    imgsFeaturesfc7Test[imgsIds[i]] = net.blobs['fc7'].data[i]
np.save('imgsFeaturesfc7Test', imgsFeaturesfc7Test) # save as .npy # extract the feature vector from the layer of interest

# Test fc8
print("Saving features fc8")
imgsFeaturesfc8Test = {}
for i in range(5000, 6000):
    imgsFeaturesfc8Test[imgsIds[i]] = net.blobs['fc8'].data[i]
np.save('imgsFeaturesfc8Test', imgsFeaturesfc8Test) # save as .npy # extract the feature vector from the layer of interest
