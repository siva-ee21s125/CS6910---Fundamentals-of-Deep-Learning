## Training the Dataset and vaidating the dataset
import wandb

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

from FeedForwardNN import neuralNetwork

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data();

x_train = x_train.astype('float32') / 255;
x_test = x_test.astype('float32') / 255;

num_inputs = x_train.shape[1]*x_train.shape[2];
num_validation_data = int(0.1*x_train.shape[0]);
num_train_data = x_train.shape[0] - num_validation_data;

x_train = x_train.reshape(x_train.shape[0],num_inputs);

x_valid = [];
y_valid = [];
for i in range(num_validation_data):
    x_valid.append(x_train[-i]);
    y_valid.append(y_train[-i]);


sweep_config = {
    'method' : 'random',
    'metric' :
    {
        'name' : 'accuracy',
        'goal' : 'maximize'
    },

    'parameters' :
    {
        'num_epochs' : { 'values' : [5, 10, 15]},
        'num_hidden_layers' : { 'values' : [3, 4, 5]},
        'hidden_layer_nodes' : { 'values' : [32, 64, 128]},
        'weight_decay' : { 'values' : [0, 0.0005, 0.5]},
        'learning_rate' : { 'values' : [1e-3, 1e-4]},
        'optimiser' : { 'values' : ['sgd', 'momentum', 'nesterov', 'rmsProp', 'adam', 'nadam']},
        'batch_size' : { 'values' : [16, 32, 64]},
        'weight_init' : {'values' : ['random', 'xavier']},
        'activation_fn' : {'values' : ['sigmoid', 'tanh', 'ReLU']},
        'loss_fn' : {'values' : ['crossEntropy']}
    }
    

}

sweep_id = wandb.sweep(sweep=sweep_config, entity = 'ee21s125', project='EE21S125_DL_A1');



def main(config = None):
        with wandb.init(config = config):
            print(wandb.config)
            num_epochs = wandb.config.num_epochs;
            learning_rate = wandb.config.learning_rate;
            num_hidden_layers = wandb.config.num_hidden_layers;
            hidden_layer_nodes = wandb.config.hidden_layer_nodes;
            weight_init = wandb.config.weight_init;
            activation_fn = wandb.config.activation_fn;
            loss_fn = wandb.config.loss_fn;
            optimiser = wandb.config.optimiser;
            weight_decay = wandb.config.weight_decay;
            batch_size = wandb.config.batch_size;
            gamma = 0.9;
            epsilon = 1e-8;
            beta = 0.95;
            beta1 = 0.9;
            beta2 = 0.999;

            wandb.run.name = activation_fn+"_hl_"+str(hidden_layer_nodes)+"_loss_"+loss_fn+"_epochs_"+str(num_epochs)+"__bs__"+str(batch_size);
            trainNN = neuralNetwork(num_inputs, num_hidden_layers, hidden_layer_nodes, learning_rate, num_epochs, weight_init, activation_fn, loss_fn, num_train_data, num_validation_data, optimiser, weight_decay, batch_size,
                  gamma, epsilon, beta, beta1, beta2, x_train, y_train, x_valid, y_valid);


            if(wandb.config.optimiser == "sgd"):
                        training_loss, training_accuracy, validation_accuracy, Y_pred = trainNN.sgd();
            elif(wandb.config.optimiser == "momentum"):
                        training_loss, training_accuracy, validation_accuracy, Y_pred = trainNN.mgd();
            elif(wandb.config.optimiser == "nesterov"):
                        training_loss, training_accuracy, validation_accuracy, Y_pred = trainNN.nag();
            elif(wandb.config.optimiser == "rmsProp"):
                        training_loss, training_accuracy, validation_accuracy, Y_pred = trainNN.rmsProp();
            elif(wandb.config.optimiser == "adam"):
                        training_loss, training_accuracy, validation_accuracy, Y_pred = trainNN.adam();
            elif(wandb.config.optimiser == "nadam"):
                        training_loss, training_accuracy, validation_accuracy, Y_pred = trainNN.nadam();

wandb.agent(sweep_id,function=main,count=100);
wandb.finish();