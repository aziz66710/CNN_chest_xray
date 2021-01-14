# Convolutional Neural Network(CNN) for Image Classification of Normal and Pneumonia Chest X-rays

## Introduction
Pneumonia is an infection within the lungs where the air sacs in one or both lungs have become inflamed. The air sacs fill with fluid or pus which causes coughing with phlegm, fever and difficulty breathing. It is caused by bacteria, viruses and/or fungi entering the respiratory system. Over 150 million people get infected with pneumonia on an annual basis especially children under 5 years old. Doctors, nurses and hospital staff review chest x-rays in order to diagnose a patient has pneumonia. They look for white spots on the lungs to identify an infection. 

Based on the images of normal and pneumonia chest x-rays seen below, it can be very difficult to determine whether a patient has pneumonia or not. With the emergence of machine learning and AI, computers can be taught to quickly and accurately identify and diagnose pneumonia based on the patients chest x-rays. This would greatly improve efficiency within the hospital and will help doctors streamline their care with their patients.

**Normal Chest X-ray**           | **Pneumonia Chest X-ray**
:-------------------------:|:-------------------------:
![Normal X-ray](https://github.com/aziz66710/CNN_chest_xray/blob/main/normal.jpg)   |  ![Pneumonia X-ray](https://github.com/aziz66710/CNN_chest_xray/blob/main/pneumonia.jpg)

## Purpose
This project will focus on the Image Classification of Normal and Pneumonia chest x-rays using a Convolutional Neural Network (CNN). CNN is one of the most popular and powerful methods of deep learning for image processing, classification and segmentation. Put simply, a CNN looks for key patterns and features within images using a special filter called a convolution filter. Once the features have been found, they are given to a neural network (brain) where it is taught to identify these key features in images. Once a sufficient amount of data is given to the network, it will be considered as "trained" and it can begin making predictions on new, unseen images.   

## Libraries and Skills Used
- Tensorflow - Open source library created by Google for numerical computation and large-scale machine learning. Used with keras to build the CNN model.
- Keras - Deep Learning API written in Python. Used with TensorFlow to create large-scale deep learning models.
- Scikit-learn -  Machine learning library. This was used for generation confusion matrix and classifier performance metrics.
- Matplotlib - This library was used for all the plots generated including the images. 
- Image processing - Images were processed by resizing all images to one size and using the VGG16 (popular CNN architecture) pre-processing function.  
- Image augmentation - Dataset size was increased using the ImageDataGenerator function provided in TensorFlow. 
- Deep Learning - CNN deep learning model with multiple convolution and max pooling layers. See Build the Neural Network below. 

## Dataset
The dataset can be found at the following link: [Chest X-ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). The file structure can be seen below.

![Directory for images](https://github.com/aziz66710/CNN_chest_xray/blob/main/tree.png)

It is important to note that there is **NOT** an equal distribution of images within the given directories. Therefore, it may be necessary to artificially augment the images to increase the size of the dataset (foreshadowing). Additionally, the images come in varying sizes and shapes and is therefore recommended that they are standardized into one size (foreshadowing, again).  

## Method

### Image Data Processing

In order to achieve high accuracy in our image classification, a series of pre-processing steps must be followed. These include:
1. Standardizing the image size and shape to ensure uniformity across the train, test, and validation datasets.
2. Artificially create new images to grow the dataset using the ImageDataGenerator object in the Tensorflow-Keras library where the images have been rescaled, stretched, zoomed in or out and flipped. These new images will add variability and diversity to the dataset which will ensure that the CNN will become robust during training. 

The intial given dataset contained a total of 5,863 X-Ray images and the following total is 10000 where it has been split into train-test-valid as 8000-1000-1000 respectively.

![Image Data Generator](https://github.com/aziz66710/CNN_chest_xray/blob/main/image_data_gen.png)

3. Display images to ensure they have been pre-processed.
4. Prepare the images in batches to be fed into the Neural Network.

The following images displays the pre-processed images (vgg16 pre-processing, re-scaling, zoom, shear (image distortion along an axis)). These images and many more batches will be passed into the CNN for training. 

![Chest X-rays from Training Batch](https://github.com/aziz66710/CNN_chest_xray/blob/main/train_images.png)


### Build the Neural Network

Now that the data has been prepared. The CNN can now be built! The CNN was built using the Tensorflow Keras library. The flow diagram below demonstrates all the different layers of the CNN and their specific details. The '?' indicates the varying number of batch sizes. 

![CNN Architecture](https://github.com/aziz66710/CNN_chest_xray/blob/main/cnn_architecture.png)

Details on the parameters, cost functions and results from training can be found in: ![CNN X-ray classifier](https://github.com/aziz66710/CNN_chest_xray/blob/main/CNN_Normal_Pneumonia.ipynb)


## Results and Discussion

Training accuracy = 96%
Testing Accuracy = 80%

|`Class`        |`Precision`  | `Recall`    | `F1-Score`    |
| -----------   | ----------- |-------------|---------------|
| `Normal`      | `0.78`      | `0.84`      | `0.81`        |
| `Pneumonia`   | `0.83`      | `0.77`      | `0.80`        |

The images below demonstrates the non-normalized and normalized confusion matrices, respectively, of the test dataset containing the number of correct/incorrect classifications of the normal and pneumonia x-ray images. 

In case you have not heard of a confusion matrix, it is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. An image is classified correctly when the predicted label is the same as the true label. Similarily an image is classified incorrectly when the predicted label is NOT the same as the true label. Given this, the top right indicates the CORRECTLY classified pneumonia chest x-ray images and the bottom left indicates the CORRECTLY classified normal images. The other 2 boxes are the misclassifications where the CNN incorrectly predicted an image as either normal or pneumonia.  

Based on these confusion matrices, a great deal of chest-xrays were incorrectly classified as 'PNEUMONIA' when it was truely 'NORMAL'. Misclassifications generally occur due to the CNN being unable to clearly identify the key distinguishing features within an image such as the white spots found in a pneumonia x-ray. Despite this, the CNN demonstrates promising results with accuracy at 80% and can be further improved upon as discussed in the Future Work and Improvements section. 

![Non-normalized CM](https://github.com/aziz66710/CNN_chest_xray/blob/main/non-norm_cm.png)




![normalized CM](https://github.com/aziz66710/CNN_chest_xray/blob/main/norm_cm.png)

 

## Future Work and Improvements
- Improve accuracy of the model through hyperparameter optimization (increase number of layers, learning rate,etc)
- Increase speed of computation through the use of GPU instead of CPU and reducing image size
- Expand classification to identify what kind of pneumonia is present (bacterial or viral).




