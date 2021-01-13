# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:45:46 2021

@author: azizu

CNN classifier for normal and pneumonia chest x-ray images 

Inspiration of code from here
https://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a

Dataset:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""

"""
Import basic libraries 

"""

import matplotlib.pyplot as plt # visualization
import numpy as np #handing arrays
import pandas as pd #handling data
import os

#Define Directories for train, test & Validation Set
train_path = 'chest_xray/train'
test_path = 'chest_xray/test'
valid_path = 'chest_xray/val'

testing_path = 'chest_xray/beep'
#batch size refers to the number of training samples used in each iteration 
batch_size = 16

#define image size = width x height
img_width = 500
img_height = 500

#size 500 chosen with batch size 16 to ensure that the RAM does not crash from overuse. A low dimension size with ...
# higher batch size is a better choice

""" 
The downloaded dataset contained a higher number of pneumonia chest-xrays than the normal ones. In order to mitigate
any bias in the classification, there needs to be an equal number of normal and pneumonia x-rays.
This is performed through data augmentation where the existing images chest x-rays are artifically modified and are 
created as new images. This additionally serves as adding variability to the dataset and improving the ability
of the model to predict new images. 
"""

from PIL import Image
import glob
import os


path = 'chest_xray/beep' #the path where to save resized images

#resize all images to a standard 500 x 500

def resize_images (image_path,resize_height,resize_width):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # loop over existing images and resize
    # change path to your path
    for filename in glob.glob(image_path + '/*.jpeg'): #path of raw images
        img = Image.open(filename).resize((resize_height,resize_width))
        # save resized images to new folder with existing filename
        img.save('{}{}{}'.format(image_path,'/',os.path.split(filename)[1]))

resize_images(train_path + '/NORMAL',img_width,img_height)
resize_images(train_path + '/PNEUMONIA',img_width,img_height)

resize_images(test_path + '/NORMAL',img_width,img_height)
resize_images(test_path + '/PNEUMONIA',img_width,img_height)

resize_images(valid_path + '/NORMAL',img_width,img_height)
resize_images(valid_path + '/PNEUMONIA',img_width,img_height)

#Data augmentation to create new images to ensure similar number of normal pneumonia

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
                                  rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,          
                               )
img = load_img('chest_xray/beep/IM-0115-0001.jpeg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

plt.imshow(x)


# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in image_gen.flow(x, batch_size=1,
                          save_to_dir='chest_xray/beep/preview',save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely



# rescale = scales array of original image to be between 0 and 1 which allows images to contribute equally to overall 
# loss.
# shear_range = fixes one axis and stretches the image at a certain angle
# zoom_range =  the image is enlarged by a zoom of less than 1.0 (zoom out). 
# horizontal_flip = flip the image horizontally (images chosen at random)



# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale = 1./255)

#test images dont need to be modified only rescaled to fit the sizes of all the training images


"""
Loading images

Read in images from directory using the flow_from_directory method on image_gen

"""

#train = image_gen.flow_from_directory(train_path,
#                                      target_size = (img_height,img_width),
#                                      color_mode = 'grayscale',
#                                      class_mode = 'binary',
#                                      classes = ['NORMAL','PNEUMONIA'],
#                                      batch_size = batch_size)

#class_mode set to binary since we are predicting if it is either normal or pneumonia x-ray. If it was multiple,
#it would be changed into categorical and an autoencoder would be needed

#test = test_data_gen.flow_from_directory(test_path,
#                                         target_size = (img_height,img_width),
#                                         color_mode = 'grayscale',
#                                         shuffle = False)

# Important to set shufffle to False so that there will be no indexing issue when comparing to predicted values

#valid = test_data_gen.flow_from_directory(
#      valid_path,
#      target_size=(img_height, img_width),
#      color_mode='grayscale',
#      class_mode='binary',
#      classes = ['NORMAL','PNEUMONIA'],
#      batch_size=batch_size
#      )

import tf.keras.applications.vgg16.preprocess_input
import tensorflow as tf

train_batch = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input,
                                 rescale = 1./255,
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 horizontal_flip = True,) \
.flow_from_directory(directory = train_path, 
                     target_size = (img_width,img_height), 
                     classes = ['NORMAL','PNEUMONIA'], 
                     batch_size = batch_size)

test_batch = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input,
                                   rescale = 1./255) \
.flow_from_directory(directory = test_path, 
                     target_size = (img_width,img_height), 
                     classes = ['NORMAL','PNEUMONIA'], 
                     batch_size = batch_size)

valid_batch = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input,
                                   rescale = 1./255) \
.flow_from_directory(directory = testing_path, 
                     target_size = (img_width,img_height), 
                     classes = ['NORMAL','PNEUMONIA'], 
                     batch_size = batch_size)



#testing = image_gen.flow_from_directory(testing_path,
#                                      target_size = (img_height,img_width),
#                                      color_mode = 'grayscale',
#                                      class_mode = 'binary',
#                                      classes = ['NORMAL','PNEUMONIA'],
#                                      batch_size = batch_size) 

train_img, train_labels = next(train_batch)
test_img, test_labels = next(test_batch)
valid_img, valid_labels = next (valid_batch)



assert train.n == 5216
assert test.n == 624
assert valid.n == 16

"""
Visualize the data augmented images from the training dataset

"""

imgs, labels = next(train)

imgs = img_to_array(imgs)
imgs = np.ndarray(imgs)


def plotImages(images_arr):
    fig, axes = plt.subplots(10,1,figsize = (12,12))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(train_img)
print(train_labels)


plotImages(testing_img)
print(testing_labels)


import matplotlib.image as mpimg
#img = mpimg.imread('your_image.png')
imgplot = plt.imshow(imgs)
plt.show()

plt.imshow(r'chest_xray\train\NORMAL\IM-0115-0001.jpeg')


testing_image = Image.open(r'chest_xray\train\NORMAL\IM-0115-0001.jpeg')


w = img_to_array(testing_image)

im_data = mpimg.imread(r'chest_xray\train\NORMAL\IM-0115-0001.jpeg')


plt.imshow(im_data)


import numpy as np

print(im_data)
print(np.shape(im_data))


plt.imshow(w)

plt.figure(figsize=(12,12))
for i in range (0,10):
    plt.subplot(2,5,i+1)
    for X_batch, Y_batch in train_batch:
        image = X_batch[0]
        dic = {0:'NORMAL',1:'PNEUMONIA'}
        #plt.title(dic.get(Y_batch[0]))
        plt.axis('off')
        
plt.imshow(np.squeeze(image),cmap='grey',interpolation='nearest')

plt.tight_layout()
plt.show()

"""
Import necessary CNN libraries 

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

"""
Building the CNN architecture

"""
cnn = Sequential()

cnn.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (img_width,img_height,1)))
cnn.add(MaxPooling2D(pool_size = (2,2)))

cnn.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (img_width,img_height,1)))
cnn.add(MaxPooling2D(pool_size = (2,2)))

cnn.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (img_width,img_height,1)))
cnn.add(MaxPooling2D(pool_size = (2,2)))

cnn.add(Conv2D(64,(3,3),activation = 'relu',input_shape = (img_width,img_height,1)))
cnn.add(MaxPooling2D(pool_size = (2,2)))

cnn.add(Conv2D(64,(3,3),activation = 'relu',input_shape = (img_width,img_height,1)))
cnn.add(MaxPooling2D(pool_size = (2,2)))

cnn.add(Flatten())

cnn.add(Dense(activation = 'relu',units = 128))
cnn.add(Dense(activation = 'relu',units = 64))
cnn.add(Dense(activation = 'sigmoid',units = 1))



























