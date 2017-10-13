## German Traffic Sign Classifier

Overview
---
In this project, I created a convolutional neural network to classify traffic signs. The model was trained and validated with data from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

### Dataset Summary & Exploration

A random image from the training dataset was chosen and plotted to have an idea of what the images look like. Then an exploration of the datasets was performed using numpy and python and found out the following:

![General info about the data after Augmentation](/readme_images/data_size.png?raw=true "General info about the data after augmentation")

The data was augmented to provide an extra 10000 images in the training dataset. The script that executes this is in the root of the repository and is named 'augment_data.py'. The images were augmented using a variety of techniques such as Rotating the image, translating it to a different position and changing the shear and the brightness. In the augmentation process a random image was chosen 10000 times and augmented then added to the existing dataset to produce a new dataset stored in a file named 'train_augmented.p'.

Training Dataset
![Class Distribution for Training dataset](/readme_images/distribution_training.png?raw=true "Class Distribution for Training dataset")

Validtion Dataset
![Class Distribution for Validation dataset](/readme_images/distributed_validation.png?raw=true "Class Distribution for Validation dataset")

Test Dataset
![Class Distribution for Test dataset](/readme_images/distribution_test.png?raw=true "Class Distribution for Test dataset")

The 3 above charts shows the class distribution over the training, validation and test set respectively(After augmentation). As we can observe the data is skewed in favor of certain classes but this is consistent across all the datasets so there was no need to supplement the training dataset to have a uniform distribution across classes. 

### Design and Testing of model Architecture

#### The model was preprocessed using the following techniques:
* Converting it to grayscale to reduce the color channels that the network needs to focus on to 1 so that it can train better.
* Normalization using by simply applying the formula (pixel-180)/180, to keep the values between -1 and 1. 
