# Convolutional Neural Network(CNN) for Image Classification of Normal and Pneumonia Chest X-rays

## Introduction
Pneumonia is an infection within the lungs where the air sacs in one or both lungs have become inflamed. The air sacs fill with fluid or pus which causes coughing with phlegm, fever and difficulty breathing. It is caused by bacteria, viruses and/or fungi entering the respiratory system. Over 150 million people get infected with pneumonia on an annual basis especially children under 5 years old. Doctors, nurses and hospital staff review chest x-rays in order to diagnose a patient has pneumonia. They look for white spots on the lungs to identify an infection. With the emergence of machine learning and AI, computers can be taught to quickly and effectively identify and diagnose pneumonia based on the chest x-rays. This would greatly improve efficiency within the hospital and will help doctors streamline their care with their patients.   

## Purpose
This project will focus on the Image Classification of Normal and Pneumonia chest x-rays using a Convolutional Neural Network (CNN). CNN is one of the most popular and powerful methods of deep learning for image processing, classificafication and segmentation. Its power stems from the convolution operation where it acts as a filter to identify key features within an image that can be used for accurate classification. 

## Skills used
Tensorflow, Keras, Scikit-learn, Matplotlib, Image processing, Image augmentation, Machine Learning/AI, Deep Learning

## Dataset
The dataset can be found at the following link: [Chest X-ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

It is important to note that there is **NOT** an equal distribution of images within the given directories. Therefore, it may be necessary to artificially augment the images to increase the size of the dataset (foreshadowing). Additionally, the images come in varying sizes and shapes and is therefore recommended that they are standardized into one size (foreshadowing, again).  

## Method

### Image Data Processing

In order to achieve high accuracy in our image classification, a series of pre-processing steps must be followed. These include:
1. Standardizing the image size and shape to ensure uniformity across the train, test, and validation datasets.
2. Artificially create new images using the ImageDataGenerator object in the Tensorflow-Keras library.
3. Display images to ensure they have been pre-processed
4. Prepare the images in batches to be fed into the Neural Network.




![Chest X-rays from Training Batch](https://github.com/aziz66710/CNN_chest_xray/blob/main/train_images.png)


|`Class`        |`Precision`  | `Recall`    | `F1-Score`    |
| -----------   | ----------- |-------------|---------------|
| `Normal`      | `0.78`      | `0.84`      | `0.81`        |
| `Pneumonia`   | `0.83`      | `0.77`      | `0.80`         |

Accuracy = 80% on Test Data



