#Testing the dataset with the best parameters

import wandb

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data();

x_train = x_train.astype('float32') / 255;
x_test = x_test.astype('float32') / 255;

num_inputs = x_train.shape[1]*x_train.shape[2];
num_validation_data = int(0.1*x_train.shape[0]);
num_train_data = x_train.shape[0] - num_validation_data;

x_train = x_train.reshape(x_train.shape[0],num_inputs);
x_test = x_test.reshape(x_test.shape[0],num_inputs);
print(x_test.shape);
x_valid = [];
y_valid = [];
y_train_only = [];

for i in range(num_train_data):
    y_train_only.append(y_train[i]);

for i in range(num_validation_data):
    x_valid.append(x_train[-i]);
    y_valid.append(y_train[-i]);

num_test_data = x_test.shape[0];
print(num_test_data)
Y_test = np.zeros((num_test_data,10));
for i in range(num_test_data):
    value = y_test[i];
    Y_test[i][int(value)] = 1.0;


sweep_config = {
    'method' : 'random',
    'metric' :
    {
        'name' : 'accuracy',
        'goal' : 'maximize'
    },
    'parameters' :
    {
        'num_epochs' : { 'values' : [15]},
        'num_hidden_layers' : { 'values' : [ 4 ]},
        'hidden_layer_nodes' : { 'values' : [128]},
        'weight_decay' : { 'values' : [0]},
        'learning_rate' : { 'values' : [1e-3]},
        'optimiser' : { 'values' : ['nadam']},
        'batch_size' : { 'values' : [16]},
        'weight_init' : {'values' : ['xavier']},
        'activation_fn' : {'values' : ['sigmoid']},
        'loss_fn' : {'values' : ['crossEntropy']}
    }


}

sweep_id = wandb.sweep(sweep=sweep_config, entity = 'ee21s125', project='EE21S125_DL_A1');



def main(config = None):
        
        test_loss = [];
        y_pred_test = [];
        Y_train_pred = [];
        Y_test_pred = [];
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

            wandb.run.name = activation_fn+"_hl_"+str(hidden_layer_nodes)+"_loss_"+loss_fn+"_epochs_"+str(num_epochs)+"__bs__"+str(batch_size) + str(weight_decay)+"_weight_decay_"+optimiser+"_optimiser_";
            NN = neuralNetwork(num_inputs, num_hidden_layers, hidden_layer_nodes, learning_rate, num_epochs, weight_init, activation_fn, loss_fn, num_train_data, num_validation_data, optimiser, weight_decay, batch_size,
                  gamma, epsilon, beta, beta1, beta2, x_train, y_train, x_valid, y_valid);


            if(wandb.config.optimiser == "sgd"):
                        training_loss, training_accuracy, validation_accuracy, y_pred_train = NN.sgd();
            elif(wandb.config.optimiser == "momentum"):
                        training_loss, training_accuracy, validation_accuracy, y_pred_train = NN.mgd();
            elif(wandb.config.optimiser == "nesterov"):
                        training_loss, training_accuracy, validation_accuracy, y_pred_train = NN.nag();
            elif(wandb.config.optimiser == "rmsProp"):
                        training_loss, training_accuracy, validation_accuracy, y_pred_train = NN.rmsProp();
            elif(wandb.config.optimiser == "adam"):
                        training_loss, training_accuracy, validation_accuracy, y_pred_train = NN.adam();
            elif(wandb.config.optimiser == "nadam"):
                        training_loss, training_accuracy, validation_accuracy, y_pred_train = NN.nadam();

            y_pred_test = NN.predict(x_test);
            print("Test Data Set is done...")
            test_accuracy = NN.accuracy(Y_test, y_pred_test, num_test_data);
            print("Test accuracy:", test_accuracy);
            
            for i in range(num_test_data):
              test_loss.append(NN.calculateCrossEntropyLoss(Y_test[i].reshape(1,10), y_pred_test[i]) + NN.L2RegularisationLoss());
            print("Test Loss:",np.mean(test_loss));
            wandb.log({'Test Accuracy':test_accuracy,'Test Loss':np.mean(test_loss)})

            for i in range(num_train_data):
              Y_train_pred.append(np.argmax(y_pred_train[i]));
            for i in range(num_test_data):
              Y_test_pred.append(np.argmax(y_pred_test[i]));

            conf_mat = confusion_matrix(Y_train_pred,y_train_only);
            cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = np.array(range(10)))

            cm_display.plot();
            fig_path = "train_conf_matrix.png";
            wandb.log({"train_conf_matrix": wandb.Image(Image.open(fig_path))});

            plt.savefig(fig_path);
    
            conf_mat = confusion_matrix(Y_test_pred,y_test);
            cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = np.array(range(10)))

            cm_display.plot();
            fig_path = "test_conf_matrix.png";
            wandb.log({"test_conf_matrix": wandb.Image(Image.open(fig_path))});
            plt.savefig(fig_path);
    


wandb.agent(sweep_id,function=main,count=1);
wandb.finish();