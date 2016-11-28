import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import caffe

### Basic structure of this file is based on: http://www.cc.gatech.edu/~zk15/deep_learning/classify_test.py

caffe.set_mode_cpu()

### For these four lines of code, thank you, Rachel: https://sites.duke.edu/rachelmemo/2015/04/30/convert-binaryproto-to-npy/
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( 'video_mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )
###

mean_arr = np.squeeze(mean_arr) # has some pointless dimensions

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'unsup_net_deploy.prototxt'
PRETRAINED = 'color_model.caffemodel'

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=mean_arr.mean(1).mean(1),
                     #  channel_swap=(2,1,0),
                       raw_scale=255,
                    #  image_dims=(256, 256)
                      )
### Loading in the images
images_dir  = './img/'
categories = ['cat', 'bike',
            ]
paths = [images_dir+category for category in categories]
images = []
image_paths = []
for p, path in enumerate(paths):
    for i, image in enumerate(os.listdir(path)):
     if os.path.isfile(os.path.join(path, image)):
        images.append(caffe.io.load_image(os.path.join(path, image)))
        image_paths.append(os.path.join(path, image))
###

### Asking network to predict the classes
predictions = net.predict(images)  # predict takes any number of images, and formats them for the Caffe net automatically
for i, pre in enumerate(predictions):
    print '\ninput image:', image_paths[i]
    print 'prediction shape:', pre.shape
    print 'predicted class:', pre.argmax()
