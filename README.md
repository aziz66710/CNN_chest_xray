# Convolutional Neural Network(CNN) for Image Classification of Normal and Pneumonia Chest X-rays

## Introduction
Pneumonia is an infection within the lungs where the air sacs in one or both lungs have become inflamed. The air sacs fill with fluid or pus which causes coughing with phlegm, fever and difficulty breathing. It is caused by bacteria, viruses and/or fungi entering the respiratory system. Over 150 million people get infected with pneumonia on an annual basis especially children under 5 years old. Doctors, nurses and hospital staff review chest x-rays in order to diagnose a patient has pneumonia. They look for white spots on the lungs to identify an infection. With the emergence of machine learning and AI, computers can be taught to quickly and effectively identify and diagnose pneumonia based on the chest x-rays. This would greatly improve efficiency within the hospital and will help doctors streamline their care with their patients.


![Pneumonia X-ray](https://github.com/aziz66710/CNN_chest_xray/blob/main/Webp.net-resizeimage%20(1).jpg)

## Purpose
This project will focus on the Image Classification of Normal and Pneumonia chest x-rays using a Convolutional Neural Network (CNN). CNN is one of the most popular and powerful methods of deep learning for image processing, classification and segmentation. Its power stems from the convolution operation where it acts as a filter to identify key features within an image that can be used for accurate classification. 

## Skills used
Tensorflow, Keras, Scikit-learn, Matplotlib, Image processing, Image augmentation, Machine Learning/AI, Deep Learning

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

The images below demonstrates the confusion matrix of the test dataset containing the number of correct/incorrect classifications of the normal and pneumonia x-ray images.

![Non-normalized CM](https://github.com/aziz66710/CNN_chest_xray/blob/main/non-norm_cm.png)

![normalized CM](https://github.com/aziz66710/CNN_chest_xray/blob/main/norm_cm.png)

 

## Future Work and Improvements
- Improve accuracy of the model through hyperparameter optimization (increase number of layers, learning rate,etc)
- Increase speed of computation through the use of GPU instead of CPU and reducing image size
- Expand classification to identify what kind of pneumonia is present (bacterial or viral).




