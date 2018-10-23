from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

model = VGG19(weights='imagenet') # include_top=True
model.save("model/vgg19.h5")