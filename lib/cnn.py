import numpy as np
import os
os.environ['GLOG_minloglevel'] = '2' # only keep warning messages
import sys
sys.path.append("/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/caffe-home/caffe/build/install/python") # path to caffe
import caffe

class cnn():
    def __init__(self):
        # Defining CNN variables
        print("Defining CNN variables")
        projectDir = '/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/'
        model_prototxt = projectDir + 'VGG_ILSVRC_19_layers_deploy.prototxt'
        model_trained = projectDir + 'VGG_ILSVRC_19_layers.caffemodel'
        mean_path = projectDir + 'caffe-home/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

        # Loading the Caffe model, setting preprocessing parameters
        print("Loading caffe model and configure preprocessing")
        caffe.set_mode_cpu()
        self.net = caffe.Net(model_prototxt, model_trained, caffe.TEST)
        (self.transformer) = caffe.io.Transformer({'data': (self.net).blobs['data'].data.shape})
        (self.transformer).set_mean('data', np.load(mean_path).mean(1).mean(1))
        (self.transformer).set_transpose('data', (2, 0, 1))
        (self.transformer).set_channel_swap('data', (2, 1, 0))
        (self.transformer).set_raw_scale('data', 255.0)

    def extractFeatures(self,imgName):
        dataType = 'val2014'
        dataDir = '/media/matthieu/Documents/2016_2017_3A_Mines_Paristech/MVA/Recvis/Project/coco-master/'
        # Extract features
        print("Extract features")
        (self.net).blobs['data'].reshape(1, 3, 224, 224) # reshape network blob
        image_path = dataDir + 'images/' + dataType + '/' + imgName
        (self.net).blobs['data'].data[...] = (self.transformer).preprocess('data', caffe.io.load_image(image_path)) # run the image through the preprocessor
        (self.net).forward() # run the image through the network
        imgFeaturesfc6 = (self.net).blobs['fc6'].data[0].copy()
        imgFeaturesfc7 = (self.net).blobs['fc7'].data[0].copy()
        imgFeaturesfc8 = (self.net).blobs['fc8'].data[0].copy()
        imgFeaturesprob = (self.net).blobs['prob'].data[0].copy()

        print("Returning features train")
        return imgFeaturesfc6, imgFeaturesfc7, imgFeaturesfc8, imgFeaturesprob # extract the feature vector from the layer of interest
