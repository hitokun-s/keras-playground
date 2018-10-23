from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = ResNet50(weights='imagenet') # include_top=True
model.save("model/resnet50.h5")