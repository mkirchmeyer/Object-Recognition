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
dataDir = '/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/coco-master/'
dataType = 'val2014'
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
layer_name = 'fc8'

# Loading the Caffe model, setting preprocessing parameters
print("Loading caffe model and configure preprocessing")
caffe.set_mode_cpu() # setting computational mode to CPU
net = caffe.Net(model_prototxt, model_trained, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
transformer.set_transpose('data', (2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255.0)
imgsFeatures = {} # feature vector extracted from the images

def extractFeatures():
    net.blobs['data'].reshape(1, 3, 224, 224) # reshape network blob

    for i in range(nbImages):
        print("Processing image %d" % (i+1))
        image_path = dataDir + 'images/' + dataType + '/' + imgsNames[i]
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path)) # run the image through the preprocessor
        output = net.forward() # run the image through the network
        imgsFeatures[imgsIds[i]] = net.blobs[layer_name].data[0] # extract the feature vector from the layer of interest

def main():
    print("Extract features from %s (%d images)" % (dataType, nbImages))
    extractFeatures()

    print("Saving features")
    np.save('imgsFeatures', imgsFeatures) # save as .npy

if __name__ == '__main__':
    main()
