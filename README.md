# Face-Mask-Detection-Python
Face Mask Detector Using Python3 OpenCV and Tensorflow

## Dataset
The dataset was made by [Prajna Bhandary](https://github.com/prajnasb/observations)
The dataset consists of **1,376 images**
* ``` with_mask ``` : 690 images
* ``` without_mask ``` : 686 images

## Libraries Used
* Tensorflow
* Keras
* Sklearn's train_test_split
* os
* imutils
* numpy
* MobileNetV2
* OpenCV 
* Caffe Model OpenCV's face detector

## Files
* **dataset** - Consists of the dataset we will train the model on
* **caffe** - Consists of the caffe face detection model
* **generating_dataset.py** - Converts data into labels and numpy arrays and pre-process it for mobilenetv2
* **training_model.py** - Trains the model with output from generating_dataset.py file
* **detecting_mask.py** - Uses the Caffe face dectection model to detect faces and pass it to the trained model for predictions
* **model** - Trained Model saved from training_model.py file

## Running 
* Run **generating_dataset.py** to generate data and labels from dataset, it will convert images into array and preprocess it for the mobilenetv2 model save the data and labels in the same directory with names **data.npy** & **labels.npy**
* Run **training_model.py** to load the saved **data.npy** & **labels.npy** files and the data is augmenented and uses the base model MobileNetV2 and top of model are few layers of Pooling, Flatten, Dense and 50% Dropout, the model is compiled using Adam optimizer with 32 Batch Size and the model is saved as **model file** in the same directory 
* Run **detecting_mask.py** file to load the **model** file and load the Caffe Face Detection model and use it to detect face using webcam stream and using the cropped part from face as input to our trained model.

## Accuracy
* Training Accuracy - *99.81%*
* Validation Accuracy - *99.64%*

## Acknowledments
* [Prajna Bhandary](https://github.com/prajnasb/observations) for the dataset
* PyImageSearch for the caffe model tutorial
