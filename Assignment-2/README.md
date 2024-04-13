### iNaturalist Dataset Classification using Convolutional Neural Networks

This folder contains the code base for Assignment 2 as part of CS6910: Deep Learning Fundamentals course at IIT Madras

The wandb report can be found in the following link:


Dataset:

iNaturalist data set has been used here in this assignment. Train - 8000 Test - 2000 Validation - 2000

For the hyper parameter optimisation stage, 20% of the randomly shuffled training data set (around 2000 images and corresponding labels) are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 8000 images from the randomly shuffled training data set.

Once the best configuration is identified with the help of wandb, the full training dataset is used to train the best model configuration and the test accuracy is calculated. The resulting confusion matrix is plotted therafter.

How to run the code:

There are totally 2 folders uploaded with code, namely Part-A and Part-B .
## PART A
The train.py can be used for generating the sweep of different configurations and training and validating the neural network with it.You can execute it by

    python train.py

The test.py can be used for training and testing the neural network with the best configuration. You can execute it by

    python test.py
## PART B

Alternatively, there is CS6910_Assignment-2_Part_A.ipynb and CS6910_Assignment-2_Part_B.ipynb. You can extract it and run all the code snippets in google collab in a sequential manner

    # Part A
    Download the dataset from Google drive
    Preprocess the data
    For training the sweep of configurations
    For training and testing the neural network with best configuration

    # Part B
    Download the dataset from Google drive
    Preprocess the data
    For training and testing different neural network (RESNET50 and GOOGLENET)
    

## Result:
#   Part A:
    The neural network had a testing accuracy of 43.35% with the following hyper parameters:

- Number of filters [32]
- Activation ['ReLU']
- Filter Organization ['same']
- Data Augmentation [True] ((Mirroring, Rotating and Random Crop)
- Batch Normalization [True]
- Number of Dense Neurons [128]
- Dropout [0.2]
- Learning Rate [1e-4]
- Weight Decay [0]
- Optimizer ['NAdam']
- Number of Epochs [15]

#   Part B:
  The RESNET50 had a testing accuracy of 70.95% and GOOGLENET had a testing accuracy of 69.45%
    
