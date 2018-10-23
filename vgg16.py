from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = VGG16(weights='imagenet') # include_top=True
model.save("model/vgg16.h5")