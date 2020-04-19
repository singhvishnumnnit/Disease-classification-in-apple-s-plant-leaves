# -*- coding: utf-8 -*-
"""
Created on Tue Jun  17 11:04:24 2019

@author: Vishnu
"""

# Independent variables here are pixels. So traditional method won't work.

# Building the CNN

# Squential is used to initialize ANN
# Convolution,Pooling,Flatenning have their regular uses
# Dense is used to add fully connected layers to ANN
''' Part b'''
    '''b.a'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import load_model


    '''b. b'''

# For augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,#Scaling pixels values b/w 0 to 1
        shear_range=0.2,# Random shear
        zoom_range=0.2,# Random zoom
        horizontal_flip=True)# Flipping of images horizontaly is allowed.

# Ensuring the size of test data b/w 0 to 1 
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),#64 and 64 specified above
        batch_size=32,# Number of images in a batch
        class_mode='binary')# Diseased and Healthy only so binary

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

''' Part c'''

    '''c. a'''

classifier = Sequential()


# filters = 32 i.e. the number of feature detectors/kernel/filters are 32 (Use 64 if you're on GPU)
# kernel_size = 3, 3 i.e. the size of these kernels is 3x3
# input_shape = (3, size, size) is the format for theano backend. Since we're using tensorflow backend we'll use
#               (size, size, 3). 3 is the number of channels. RGB channels so 3 for a colored image. (256, 256 size on a GPU)
# activation = 'relu' is Rectfied Linear Units method for removing the linearity. (Rectifier used as activation on neuron)
classifier.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu'))

# pool_size = (2,2) ie. the size of the pool that will check for max values is 2x2
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

# Here we have 32 feature detectors, each with many pixels  then after all the steps of convolution, pooling and 
# flattening we'll still get a high number of nodes consisting vector, that'll act as input layer for fully connected layer.
# output_dim here represents the number of nodes in the hidden layer. These should be b/w the input and output nodes.
# Generally it is observed that choosing around 100 helps. It's not too compute intensive nor too small.

classifier.add(Dense(output_dim=128, activation = 'relu'))

# This one is for the output layer. Here output_dim = 1 since it's either a Diseased or a Healthy.
# Since outcome is binary so we'll use 'sigmoid' as the activation function
classifier.add(Dense(output_dim=1, activation = 'sigmoid'))


# Compiling CNN
# Optimizer = 'adam' means the stochastic gradient decent
# loss = 'binary_crossentropy' is the loss/cost function. If it was more than 2 outcomes then would've chosen categorical_crossentropy.
# Here outcome is binary ie. either Diseased or Healthy.
# metrics=['accuracy'] is performance metric over which it will be evaulated
    '''c. b'''
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Preparing Data
# Here we have used augmentation of images as we were not having quite large number of images.


classifier.fit_generator(
        training_set,
        steps_per_epoch=752,#364 images of Diseased and 388 images of Healthy
        epochs=1,# After which we'll be updating weights
        validation_data=test_set,#evaluating over the test_set
        nb_val_samples=120)#Images in test set. 60 Diseased and 60 Healthy




    '''c. d'''

classifier.save('my_model.h5')
classifier = load_model('my_model.h5')
# For individual testing

''' Part d'''
    # Accuracy can be seen right after the epochs have been completed.
    
''' Part e '''
path = 'dataset/single_prediction/diseased_or_healthy_2.jpg'
test_image = image.load_img(path, target_size=(128,128)) #Dimensions already specified
# Now since the data that we need to have should be a 3D, 3rd being the 3 for RGB
test_image = image.img_to_array(test_image)
# The data for the prediction needs to be 4D. 4th being the batch. Even though it is single but predict function
# expects the data to be in form of batches
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
# training_set.class_indices can be used to check the category encoding
animal = ""
if result==0:
    animal = "Diseased"
else:
    animal = "Healthy"
    
from PIL import Image, ImageDraw, ImageFont
image = Image.open(path)
font_type = ImageFont.truetype('arial.ttf',30)
draw = ImageDraw.Draw(image)
draw.text(xy=(0,0),text=animal,fill=(0,0,0),font=font_type)
image.show()
    
""" To improve accuracy, one more convulation layer can be added with either same or different parameters.
    Add it after the max pooling and before the flattening.
    Next layer won't be having input_shape parameter. Keras will automatically accept the pool of previous layer."""



