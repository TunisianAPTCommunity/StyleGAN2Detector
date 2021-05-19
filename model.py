import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import os

from keras import applications
from keras.applications.vgg16 import VGG16

# build the VGG16 network
#model_vgg16 = applications.VGG16(include_top=True, weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
model_vgg16 = VGG16(include_top=True,weights="imagenet")
model_vgg16.summary() 
type(model_vgg16)

model1 = Sequential()
for layer in model_vgg16.layers:
    model1.add(layer)
type(model1)

for layer in model1.layers:
    layer.trainable = False

model1.add(Dense(2,activation = 'softmax', name='output'))
model1.summary()

train_path = 'traindata/train'
test_path = 'testset/test'
valid_path = 'validset/valid'


train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['fake','real'],batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),classes=['fake','real'],batch_size=12)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224),classes=['fake','real'],batch_size=4)

model1.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model1.fit_generator(train_batches,steps_per_epoch=10,validation_data=valid_batches,validation_steps=4,epochs=10,verbose=2)

