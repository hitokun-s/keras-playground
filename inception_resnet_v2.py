from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = InceptionResNetV2(weights='imagenet') # include_top=True
model.save("model/inception_resnet_v2 .h5")