# **Fashion MNIST classification using a "numpy" Bare Bones Feed forward Neural Network**

This folder contains the code base for Assignment 1 as part of CS6910: Deep Learning Fundamentals course at IIT Madras

The wandb report can be found in the following link:

https://wandb.ai/ee21s125/EE21S125_DL_A1/reports/CS6910-Assignment-1--Vmlldzo3MjI3ODI0

## Dataset:

Fashion MNIST data set has been used here in this assignment instead of the traditional MNIST hand written digits dataset. Train - 60000 Test - 10000 Validation - 6000

For the hyper parameter optimisation stage, 10% of the randomly shuffled training data set (around 6000 images and corresponding labels) are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 54000 images from the randomly shuffled training data set.

Once the best configuration is identified with the help of wandb, the full training dataset is used to train the best model configuration and the test accuracy is calculated. The resulting confusion matrix is plotted therafter.

## How to run the code:
There are totally 6 files uploaded with code.

The train.py can be used for generating the sweep of different configurations and training and validating the neural network with it.You can execute it by
> python train.py

The test.py can be used for training and testing the neural network with the best configuration. You can execute it by
> python test.py

Alternatively, there is CS6910-Assignment-1.ipynb. You can extract it and run all the code snippets in google collab in a sequential manner 
1) Visualize the sample dataset.
2) Initialize the Feedforward Neural Network
3) For training the sweep of configurations
4) For training and testing the neural network with best configuration with Fashion MNIST dataset
5) For training and testing the neural network with best configuration with MNIST dataset

## Result:

The neural network had a testing accuracy of 87.46% with the Fashion-MNIST dataset and 97.96% with the MNIST dataset using the following parameters:

Activation Function: 'sigmoid' 
Batch Size: 16, 
Number of Hidden Layer nodes: 128 
Learning Rate: 0.001 
Number of epochs: 15 
Number of Hidden Layers: 4 
Optimizer: 'NADAM', 
Weight Decay: 0, 
Weight Initializer: 'Xavier'
