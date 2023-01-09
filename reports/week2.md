# Week 02 Report

## Choosing Tensorflow and Keras Framework

Decided on using Tensorflow and Keras Framework for implementing the Deep Learning Models rather than PyTorch. The reasoning behind this is mainly preference due to Tensorflow’s higher level of abstraction when compared to Pytorch. For testing out different architectures just for the sake of comparative evalutation, Tensorflow seemed to be an ideal choice. Using Pytorch in the future is definitely not ruled out.

## Data Preparation

Augmented, Preprocessed and Prepared the Dataset with images from PV01 section of the dataset selected using Sklearn and Tensorflow.

## U-Net

Implemented U-Net using Tensorflow. The training time was around 10 minutes for 200 Epochs on a Google Compute Engine GPU with an accuracy of 97.5%.

Model Summary of the U-Net Architecture on top of a MobileNetV2 base layer:

Reference: https://www.kaggle.com/code/dikshabhati2002/image-segmentation-u-net

Training time: 10m 12s on a Google Compute Engine GPU.  
Accuracy: 98.35% - Validation Accuracy 97.77%

The Model’s performance is mediocre and increasing the epochs or changing the loss function had shown no significant improvements. There is a lot of loss and distortion and no amount of training seems to help improve the model. 

## DeepLabV3+

Implemented DeepLabV3+ using Tensorflow. The architecture is significantly more complex than U-Net hence requiring more training time but the accuracy it yields is well worth the training time.

Model summary of DeepLabV3+ Architecture:

https://raw.githubusercontent.com/bharathraj-v/geoml-wd/main/reports/deeplabv3-architecture.png

Reference: https://keras.io/examples/vision/deeplabv3_plus/

Training time: 35m 27s on a Google Compute Engine GPU.  
Accuracy: 99.90% - Validation Accuracy 98.48%

The architecture is highly stable and the keras implementation, referred from https://keras.io/examples/vision/deeplabv3_plus/, is miles ahead of U-Net.

Both Models are saved in https://github.com/bharathraj-v/geoml-wd/tree/main/models

## Conclusion

DeepLabV3+ is a strong choice for the task at hand with very good accuracy. Although the GPU usage is twice as high and the time taken to yield good results are around 30 minutes with a Google Compute Engine GPU, Computing resources should not be a problem given the availability of a super-computer.