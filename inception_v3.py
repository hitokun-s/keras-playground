from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = InceptionV3(weights='imagenet') # include_top=True
model.save("model/inception_v3 .h5")