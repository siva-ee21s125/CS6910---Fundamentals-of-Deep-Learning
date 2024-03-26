# **Fashion MNIST classification using a "numpy" Bare Bones Feed forward Neural Network**

This folder contains the code base for Assignment 1 as part of CS6910: Deep Learning Fundamentals course at IIT Madras during the Spring of 2021.

The wandb report can be found in the following link:

## Dataset:

Fashion MNIST data set has been used here in this assignment instead of the traditional MNIST hand written digits dataset. Train - 60000 Test - 10000 Validation - 6000

For the hyper parameter optimisation stage, 10% of the randomly shuffled training data set (around 6000 images and corresponding labels) are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 54000 images from the randomly shuffled training data set.

Once the best configuration is identified with the help of wandb, the full training dataset is used to train the best model configuration and the test accuracy is calculated. The resulting confusion matrix is plotted therafter.

## Result:

The neural network had a testing accuracy of 87.46% with the Fashion-MNIST dataset and 97.96% with the MNIST dataset.
