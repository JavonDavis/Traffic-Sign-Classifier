## German Traffic Sign Classifier

Overview
---
In this project, I created a convolutional neural network to classify traffic signs. The model was trained and validated with data from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

### Dataset Summary & Exploration

A random image from the training dataset was chosen and plotted to have an idea of what the images look like. Then an exploration of the datasets was performed using numpy and python and produced the following:

![General info about the data after Augmentation](/readme_images/data_size.png?raw=true "General info about the data after augmentation")

The data was augmented to provide an extra 10000 images in the training dataset. The script that executes this is in the root of the repository and is named 'augment_data.py'. The images were augmented using a variety of techniques such as Rotating the image, translating it to a different position and changing the shear and the brightness. In the augmentation process a random image was chosen 10000 times and augmented then added to the existing dataset to produce a new dataset stored in a file named 'train_augmented.p'. * Note This script would need to be executed prior to the running of the notebook *

Training Dataset
![Class Distribution for Training dataset](/readme_images/distribution_training.png?raw=true "Class Distribution for Training dataset")

Validtion Dataset
![Class Distribution for Validation dataset](/readme_images/distributed_validation.png?raw=true "Class Distribution for Validation dataset")

Test Dataset
![Class Distribution for Test dataset](/readme_images/distribution_test.png?raw=true "Class Distribution for Test dataset")

The 3 above charts shows the class distribution over the training, validation and test set respectively(After augmentation). As we can observe the data is skewed in favor of certain classes but this is consistent across all the datasets so there was no need to supplement the training dataset to have a uniform distribution across classes. 

### Design and Testing of model Architecture

#### The model was preprocessed using the following techniques:
* Converting it to grayscale along with applying a histogram equalization to reduce the color channels that the network needs to focus on to 1 so that it can train better.
* Normalization by simply applying the formula (pixel-180)/180 to the pixels in the images, to keep the values between -1 and 1 instead of 0 and 255. 

#### Final Model Architecure 

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout       | Keep probability of 0.8 |
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Dropout       | Keep probability of 0.8 |
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x6				|
| Flatten         | Input 5x5x16, output 400                |
| Fully connected		| Inputed 400, output 120       									|
| RELU					|												|
| Dropout       | Keep probability of 0.8 |
| Fully connected		| Inputed 120, output 84      									|
| RELU					|												|
| Dropout       | Keep probability of 0.8 |
| Fully connected		| Inputed 84, output 43      									|

This architecture is a variation of the LeNet Architecture that was updated from previously recognizing characters from the MNIST dataset to recognizing traffic signs. A dropout with a keep probability of 80% was added at each layer in the network and resulted in a improvement in the networks performance on the Traffic Sign data. The reason LeNet was chosen is due to its reputation for being a suitable architecture for OCR and Document recognition along with being recommended as a starting point for the project from the Udacity mentors :)

### Model Training 
The model was trained with a learning rate of 0.001 over 20 EPOCHS and a batch size of 128. The Adam algorithm was then used to minimize the loss in the data passed to the network compared to the ground truth labels.
My final model results was an accurracy of about 96% on the validation set.

![Accuracy on the validation set](/readme_images/accuracy_validation.png?raw=true "Accuracy on the Validation Set after 20 EPOCHS")
