import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0';
import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import time

class neuralNetwork:
  def __init__(self, num_inputs, num_hidden_layers, hidden_layers_nodes, learning_rate, num_epochs, weight_init, activation_fn, loss_fn, num_train_data, num_validation_data, optimiser, weight_decay, batch_size,
               gamma, epsilon, beta, beta1, beta2, x_train, y_train, x_valid, y_valid):

        self.num_inputs = num_inputs;
        self.num_hidden_layers = num_hidden_layers;
        self.hidden_layers_nodes = hidden_layers_nodes;
        self.num_outputs = 10;
        self.num_epochs = num_epochs;
        self.learning_rate = learning_rate;
        self.weight_init = weight_init;
        self.activation_fn = activation_fn;
        self.loss_fn = loss_fn;
        self.num_train_data = num_train_data;
        self.num_validation_data = num_validation_data;
        self.optimiser = optimiser;
        self.weight_decay = weight_decay;
        self.batch_size = batch_size;

        self.gamma = gamma;
        self.epsilon = epsilon;
        self.beta = beta;
        self.beta1 = beta1;
        self.beta2 = beta2;

        self.x_train = x_train;
        self.y_train = y_train;
        self.x_valid = x_valid;
        self.y_valid = y_valid;

        num_hidden = [self.hidden_layers_nodes for i in range(num_hidden_layers)];

        self.num_nodes = [self.num_inputs] +  num_hidden + [self.num_outputs];
        print(self.num_nodes);
        self.weights, self.biases = self.initializeParameters();


  def xavierInitialize(self, in_dim, out_dim):
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim));
        return np.random.normal(0, xavier_stddev, size=(in_dim, out_dim));

  def randomInitialize(self, in_dim, out_dim):
        return np.random.normal(0, 1, size=(in_dim, out_dim));


  def initializeParameters(self):
        weights = [];
        biases = [];
        if(self.weight_init=="random"):
          for i in range(len(self.num_nodes)-1):
            weights.append(self.randomInitialize(self.num_nodes[i],self.num_nodes[i+1]));
            biases.append(self.randomInitialize(1,self.num_nodes[i+1]));
        elif(self.weight_init=="xavier"):
          for i in range(len(self.num_nodes)-1):
            weights.append(self.xavierInitialize(self.num_nodes[i],self.num_nodes[i+1]));
            biases.append(self.xavierInitialize(1,self.num_nodes[i+1]));
        return weights, biases;


  def forwardPropagate(self,X,weights,biases):
        H = [X];
        A = [];
        for i in range(len(weights)):

            A_temp = np.dot(H[-1], weights[i]) + biases[i];
            if(i==len(self.weights)-1):
              H_temp = self.softmax(A_temp);
            else:
                if(self.activation_fn == "sigmoid"):
                  H_temp = self.sigmoid(A_temp);
                elif(self.activation_fn == "ReLU"):
                  H_temp = self.ReLU(A_temp);
                elif(self.activation_fn == "tanh"):
                  H_temp = self.tanh(A_temp);
            H.append(H_temp);
            A.append(A_temp);
        Y = H[-1];

        return Y,H,A;

  def sigmoid(self, Z):
        return np.exp(Z) / (np.exp(Z) + 1);

  def tanh(self, Z):
        return np.tanh(Z);

  def ReLU(self, Z):
        return (Z>0)*(Z) + ((Z<0)*Z*0.01);
        #return np.maximum(Z,0);

  def derievative_sigmoid(self, Z):
        return  (1.0 / (1 + np.exp(-Z)))*(1 -  1.0 / (1 + np.exp(-Z)));

  def derievative_tanh(self, Z):
        return 1 - np.tanh(Z)**2;


  def derievative_ReLU(self, Z):
        return (Z>0)*np.ones(Z.shape) + (Z<0)*(0.01*np.ones(Z.shape));
        #return np.where(Z > 0, 1, 0);

  def softmax(self,Z):
        assert len(Z.shape) == 2;
        s = np.max(Z, axis=1);
        s = s[:, np.newaxis];
        e_x = np.exp(Z - s);
        div = np.sum(e_x, axis=1);
        div = div[:, np.newaxis];
        return e_x / div;

  def backPropagate(self,Y, H, A, Y_train):


        gradients_weights = [];
        gradients_biases = [];

        num_layers = len(self.num_nodes);


        grad_A = [];
        grad_W = [];
        grad_H = [];
        grad_B = [];

        for i in range(num_layers-1):
          if(i==0):
            if(self.loss_fn == "crossEntropy"):
              grad_A_temp = -(Y_train - Y).T;
            elif(self.loss_fn == "MSE"):
              grad_A_temp = np.multiply(2 * (Y - Y_train), np.multiply(Y, (1 - Y))).T;

            if(self.weight_decay==0):
              grad_W_temp = np.outer(grad_A_temp,H[num_layers-1-(i+1)]);
            else:
              grad_W_temp = np.outer(grad_A_temp,H[num_layers-1-(i+1)]) + (self.weight_decay * self.weights[num_layers-1-(i+1)]).T;
            grad_B_temp = grad_A_temp;

            grad_H_temp = np.dot(self.weights[num_layers-1-(i+1)],grad_A_temp);
          else:

            if self.activation_fn == "ReLU":
              grad_A_temp = np.multiply(grad_H[i-1],self.derievative_ReLU(A[num_layers-1-(i+1)]).T);
            elif self.activation_fn == "tanh":
              grad_A_temp = np.multiply(grad_H[i-1],self.derievative_tanh(A[num_layers-1-(i+1)]).T);
            elif self.activation_fn == "sigmoid":
              grad_A_temp = np.multiply(grad_H[i-1],self.derievative_sigmoid(A[num_layers-1-(i+1)]).T);

            if(self.weight_decay==0):
              grad_W_temp = np.outer(grad_A_temp,H[num_layers-1-(i+1)]);
            else:
              grad_W_temp = np.outer(grad_A_temp,H[num_layers-1-(i+1)])+ (self.weight_decay * self.weights[num_layers-1-(i+1)]).T;
            grad_B_temp = grad_A_temp;
            grad_H_temp = np.dot(self.weights[num_layers-1-(i+1)],grad_A_temp);


          grad_A.append(grad_A_temp);
          grad_H.append(grad_H_temp);

          gradients_weights.append(grad_W_temp);
          gradients_biases.append(grad_B_temp);



        return gradients_weights, gradients_biases;


  def predict(self,X):
        Y_pred = []

        for x in X:
            Y, H, A = self.forwardPropagate(x,self.weights,self.biases);
            Y_pred.append(Y);

        return Y_pred;


  def calculateCrossEntropyLoss(self,Y_true,Y_pred):
        epsilon = 1e-15
        CE = [-Y_true[i] * np.log(Y_pred[i] + epsilon) for i in range(len(Y_pred))];
        crossEntropy = np.mean(CE);
        return crossEntropy;


  def calculateMeanSquaredErrorLoss(self, Y_true, Y_pred):
        MSE = np.mean((Y_true - Y_pred) ** 2);
        return MSE;

  def accuracy(self, Y_true, Y_pred, num_data):
        Y_true_label = [];
        Y_pred_label = [];
        ctr = 0;
        for i in range(num_data):
            Y_true_label.append(np.argmax(Y_true[i]));
            Y_pred_label.append(np.argmax(Y_pred[i]));
            if Y_true_label[i] == Y_pred_label[i]:
                ctr += 1;
        accuracy = ctr / num_data;
        return accuracy;

  def L2RegularisationLoss(self):
        return self.weight_decay * np.sum([np.linalg.norm(self.weights[i])** 2 for i in range(len(self.weights))])


  def sgd(self):

        training_loss = [];
        training_accuracy = [];
        validation_accuracy = [];

        num_layers = len(self.num_nodes)-1;

        X_train = self.x_train;
        Y_train = np.zeros((self.num_train_data,self.num_outputs));
        Y_valid = np.zeros((self.num_validation_data,self.num_outputs));
        for i in range(self.num_train_data):
          value = self.y_train[i];
          Y_train[i][int(value)] = 1.0;

        for i in range(self.num_validation_data):
          value = self.y_valid[i];
          Y_valid[i][int(value)] = 1.0;

        valid_loss = [];
        for epoch in range(self.num_epochs):
            start_time = time.time();
            loss = [];
            dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
            dB = [np.zeros((1,self.num_nodes[l + 1]))for l in range(len(self.num_nodes) - 1)];

            for i in range(self.num_train_data):
                Y, H, A = self.forwardPropagate(X_train[i],self.weights,self.biases);

                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[i].reshape(1,self.num_outputs));

                dW = [ grad_weights[num_layers - 1 - i].T for i in range(num_layers)];
                dB = [grad_biases[num_layers - 1 - i].T for i in range(num_layers)];

                if self.loss_fn == "MSE":
                    loss.append(self.calculateMeanSquaredErrorLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                elif self.loss_fn == "crossEntropy":
                    loss.append(self.calculateCrossEntropyLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());


                self.weights = [(self.weights[i] - self.learning_rate * dW[i]) for i in range(len(self.weights))];
                self.biases = [(self.biases[i] - self.learning_rate * dB[i]) for i in range(len(self.biases))];

            elapsed = time.time() - start_time;


            Y_pred = self.predict(X_train);
            training_loss.append(np.mean(loss))
            training_accuracy.append(self.accuracy(Y_train, Y_pred, self.num_train_data));
            Y_pred_valid = self.predict(x_valid);
            validation_accuracy.append(self.accuracy(Y_valid, Y_pred_valid,self.num_validation_data));
            for j in range(self.num_validation_data):
              valid_loss.append(self.calculateCrossEntropyLoss(Y_valid[j].reshape(1,self.num_outputs), Y_pred_valid[j]) + self.L2RegularisationLoss());
            print("Epoch:",epoch," Loss:",training_loss[epoch]," Training accuracy:", training_accuracy[epoch]," Validation Accuracy:",validation_accuracy[epoch] , " Time:",elapsed," Learning Rate:",self.learning_rate, "Validation loss:",np.mean(valid_loss));

            wandb.log({'loss':np.mean(loss), 'training_accuracy':training_accuracy[epoch], 'validation_accuracy':validation_accuracy[epoch],'epoch':epoch,'validation_loss':np.mean(valid_loss) });

        return training_loss, training_accuracy, validation_accuracy, Y_pred;



  def mgd(self):
        training_loss = [];
        training_accuracy = [];
        validation_accuracy = [];

        num_layers = len(self.num_nodes)-1;

        X_train = self.x_train;
        Y_train = np.zeros((self.num_train_data,self.num_outputs));
        Y_valid = np.zeros((self.num_validation_data,self.num_outputs));

        for i in range(self.num_train_data):
          value = self.y_train[i];
          Y_train[i][int(value)] = 1.0;

        for i in range(self.num_validation_data):
          value = self.y_valid[i];
          Y_valid[i][int(value)] = 1.0;

        prev_v_w  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        prev_v_b = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        valid_loss = [];
        num_points_seen = 0;
        for epoch in range(self.num_epochs):
            start_time = time.time();

            loss = [];
            dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
            dB = [np.zeros((1,self.num_nodes[l + 1]))for l in range(len(self.num_nodes) - 1)];

            for i in range(self.num_train_data):
                Y, H, A = self.forwardPropagate(X_train[i],self.weights,self.biases);

                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[i].reshape(1,self.num_outputs));

                dW = [ grad_weights[num_layers - 1 - i].T + dW[i] for i in range(num_layers)];
                dB = [ grad_biases[num_layers - 1 - i].T + dB[i] for i in range(num_layers)];


                if self.loss_fn == "MSE":
                    loss.append(self.calculateMeanSquaredErrorLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                elif self.loss_fn == "crossEntropy":
                    loss.append(self.calculateCrossEntropyLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());

                num_points_seen += 1;
                if num_points_seen % self.batch_size == 0:
                    v_w = [(self.gamma*prev_v_w[i] + self.learning_rate * dW[i]/self.batch_size) for i in range(num_layers)];
                    v_b = [(self.gamma*prev_v_b[i] + self.learning_rate * dB[i]/self.batch_size) for i in range(num_layers)];

                    self.weights =[self.weights[i]  - v_w[i] for i in range(len(self.weights))];
                    self.biases = [self.biases[i]  - v_b[i] for i in range(len(self.biases))];

                    prev_v_w = v_w;
                    prev_v_b = v_b;

                    dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];
                    dB = [np.zeros((1,self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];


            elapsed = time.time() - start_time;

            Y_pred = self.predict(X_train);
            training_loss.append(np.mean(loss))
            training_accuracy.append(self.accuracy(Y_train, Y_pred, self.num_train_data));
            Y_pred_valid = self.predict(x_valid);
            validation_accuracy.append(self.accuracy(Y_valid, Y_pred_valid,self.num_validation_data));
            for j in range(self.num_validation_data):
              valid_loss.append(self.calculateCrossEntropyLoss(Y_valid[j].reshape(1,self.num_outputs), Y_pred_valid[j]) + self.L2RegularisationLoss());
            print("Epoch:",epoch," Loss:",training_loss[epoch]," Training accuracy:", training_accuracy[epoch]," Validation Accuracy:",validation_accuracy[epoch] , " Time:",elapsed," Learning Rate:",self.learning_rate, "Validation loss:",np.mean(valid_loss));

            wandb.log({'loss':np.mean(loss), 'training_accuracy':training_accuracy[epoch], 'validation_accuracy':validation_accuracy[epoch],'epoch':epoch,'validation_loss':np.mean(valid_loss) });

        return training_loss, training_accuracy, validation_accuracy, Y_pred;

  def nag(self):
        training_loss = [];
        training_accuracy = [];
        validation_accuracy = [];

        num_layers = len(self.num_nodes)-1;

        X_train = self.x_train;
        Y_train = np.zeros((self.num_train_data,self.num_outputs));
        Y_valid = np.zeros((self.num_validation_data,self.num_outputs));

        for i in range(self.num_train_data):
          value = self.y_train[i];
          Y_train[i][int(value)] = 1.0;

        for i in range(self.num_validation_data):
          value = self.y_valid[i];
          Y_valid[i][int(value)] = 1.0;

        prev_v_w  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        prev_v_b = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        valid_loss = [];
        num_points_seen = 0;
        for epoch in range(self.num_epochs):
            start_time = time.time();

            loss = [];
            dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
            dB = [np.zeros((1,self.num_nodes[l + 1]))for l in range(len(self.num_nodes) - 1)];

            v_w = [self.gamma*prev_v_w[i] for i in range(len(self.num_nodes)-1)]
            v_b = [self.gamma*prev_v_b[i] for i in range(len(self.num_nodes)-1)]


            for i in range(self.num_train_data):

                w_inter = [self.weights[i] - v_w[i] for i in range(len(self.num_nodes)-1)];
                b_inter = [self.biases[i] - v_b[i] for i in range(len(self.num_nodes)-1)];

                Y, H, A = self.forwardPropagate(X_train[i],w_inter, b_inter);

                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[i].reshape(1,self.num_outputs));

                dW = [ grad_weights[num_layers - 1 - i].T + dW[i] for i in range(num_layers)];
                dB = [ grad_biases[num_layers - 1 - i].T + dB[i] for i in range(num_layers)];


                if self.loss_fn == "MSE":
                    loss.append(self.calculateMeanSquaredErrorLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                elif self.loss_fn == "crossEntropy":
                    loss.append(self.calculateCrossEntropyLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());

                num_points_seen += 1;
                if num_points_seen % self.batch_size == 0:
                    v_w = [self.gamma*prev_v_w[i] + self.learning_rate*dW[i]/self.batch_size for i in range(num_layers)];
                    v_b = [self.gamma*prev_v_b[i] + self.learning_rate*dB[i]/self.batch_size for i in range(num_layers)];

                    self.weights =[self.weights[i]  - v_w[i] for i in range(len(self.weights))];
                    self.biases = [self.biases[i]  - v_b[i] for i in range(len(self.biases))];

                    prev_v_w = v_w;
                    prev_v_b = v_b;

                    dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];
                    dB = [np.zeros((1,self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];


            elapsed = time.time() - start_time;


            Y_pred = self.predict(X_train);
            training_loss.append(np.mean(loss))
            training_accuracy.append(self.accuracy(Y_train, Y_pred, self.num_train_data));
            Y_pred_valid = self.predict(x_valid);
            validation_accuracy.append(self.accuracy(Y_valid, Y_pred_valid,self.num_validation_data));
            for j in range(self.num_validation_data):
              valid_loss.append(self.calculateCrossEntropyLoss(Y_valid[j].reshape(1,self.num_outputs), Y_pred_valid[j]) + self.L2RegularisationLoss());
            print("Epoch:",epoch," Loss:",training_loss[epoch]," Training accuracy:", training_accuracy[epoch]," Validation Accuracy:",validation_accuracy[epoch] , " Time:",elapsed," Learning Rate:",self.learning_rate, "Validation loss:",np.mean(valid_loss));

            wandb.log({'loss':np.mean(loss), 'training_accuracy':training_accuracy[epoch], 'validation_accuracy':validation_accuracy[epoch],'epoch':epoch,'validation_loss':np.mean(valid_loss) });
        return training_loss, training_accuracy, validation_accuracy, Y_pred;

  def rmsProp(self):
        training_loss = [];
        training_accuracy = [];
        validation_accuracy = [];

        num_layers = len(self.num_nodes)-1;

        X_train = self.x_train;
        Y_train = np.zeros((self.num_train_data,self.num_outputs));
        Y_valid = np.zeros((self.num_validation_data,self.num_outputs));

        for i in range(self.num_train_data):
          value = self.y_train[i];
          Y_train[i][int(value)] = 1.0;

        for i in range(self.num_validation_data):
          value = self.y_valid[i];
          Y_valid[i][int(value)] = 1.0;

        v_w  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        v_b = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];
        valid_loss =[];
        num_points_seen = 0;
        for epoch in range(self.num_epochs):
            start_time = time.time();

            loss = [];
            dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
            dB = [np.zeros((1,self.num_nodes[l + 1]))for l in range(len(self.num_nodes) - 1)];

            for i in range(self.num_train_data):
                Y, H, A = self.forwardPropagate(X_train[i],self.weights,self.biases);

                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[i].reshape(1,self.num_outputs));

                dW = [ grad_weights[num_layers - 1 - i].T + dW[i] for i in range(num_layers)];
                dB = [ grad_biases[num_layers - 1 - i].T + dB[i] for i in range(num_layers)];

                if self.loss_fn == "MSE":
                    loss.append(self.calculateMeanSquaredErrorLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                elif self.loss_fn == "crossEntropy":
                    loss.append(self.calculateCrossEntropyLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                    

                num_points_seen += 1;
                if num_points_seen % self.batch_size == 0:
                    v_w = [self.beta*v_w[i] + (1-self.beta)*(dW[i])**2 for i in range(num_layers)];
                    v_b = [self.beta*v_b[i] + (1-self.beta)*(dB[i])**2 for i in range(num_layers)];

                    self.weights = [self.weights[i]  - dW[i]*(self.learning_rate/np.sqrt(v_w[i]+self.epsilon)) for i in range(len(self.weights))];
                    self.biases = [self.biases[i]  - dB[i]*(self.learning_rate/np.sqrt(v_b[i]+self.epsilon)) for i in range(len(self.biases))];

                    dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];
                    dB = [np.zeros((1,self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];


            elapsed = time.time() - start_time;
            #print("Elapsed",elapsed);

            Y_pred = self.predict(X_train);
            training_loss.append(np.mean(loss))
            training_accuracy.append(self.accuracy(Y_train, Y_pred, self.num_train_data));
            Y_pred_valid = self.predict(x_valid);
            validation_accuracy.append(self.accuracy(Y_valid, Y_pred_valid,self.num_validation_data));
            for j in range(self.num_validation_data):
              valid_loss.append(self.calculateCrossEntropyLoss(Y_valid[j].reshape(1,self.num_outputs), Y_pred_valid[j]) + self.L2RegularisationLoss());
            print("Epoch:",epoch," Loss:",training_loss[epoch]," Training accuracy:", training_accuracy[epoch]," Validation Accuracy:",validation_accuracy[epoch] , " Time:",elapsed," Learning Rate:",self.learning_rate, "Validation loss:",np.mean(valid_loss));

            wandb.log({'loss':np.mean(loss), 'training_accuracy':training_accuracy[epoch], 'validation_accuracy':validation_accuracy[epoch],'epoch':epoch,'validation_loss':np.mean(valid_loss) });
        return training_loss, training_accuracy, validation_accuracy, Y_pred;

  def adam(self):
        training_loss = [];
        training_accuracy = [];
        validation_accuracy = [];


        num_layers = len(self.num_nodes)-1;

        X_train = self.x_train;
        Y_train = np.zeros((self.num_train_data,self.num_outputs));
        Y_valid = np.zeros((self.num_validation_data,self.num_outputs));

        for i in range(self.num_train_data):
          value = self.y_train[i];
          Y_train[i][int(value)] = 1.0;

        for i in range(self.num_validation_data):
          value = self.y_valid[i];
          Y_valid[i][int(value)] = 1.0;

        v_w  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        v_b = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        v_w_hat  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        v_b_hat = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        m_w  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        m_b = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        m_w_hat  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        m_b_hat = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        num_points_seen = 0;
        valid_loss = [];
        for epoch in range(self.num_epochs):
            start_time = time.time();

            loss = [];
            dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
            dB = [np.zeros((1,self.num_nodes[l + 1]))for l in range(len(self.num_nodes) - 1)];

            for i in range(self.num_train_data):
                Y, H, A = self.forwardPropagate(X_train[i],self.weights,self.biases);
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[i].reshape(1,self.num_outputs));

                dW = [ grad_weights[num_layers - 1 - i].T + dW[i] for i in range(num_layers)];
                dB = [ grad_biases[num_layers - 1 - i].T + dB[i] for i in range(num_layers)];


                if self.loss_fn == "MSE":
                    loss.append(self.calculateMeanSquaredErrorLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                elif self.loss_fn == "crossEntropy":
                    loss.append(self.calculateCrossEntropyLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());

                num_points_seen += 1;
                ctr = 0;
                if num_points_seen % self.batch_size == 0:
                    ctr += 1

                    m_w = [self.beta1*m_w[i] + (1-self.beta1)*dW[i] for i in range(num_layers)];
                    m_b = [self.beta1*m_b[i] + (1-self.beta1)*dB[i] for i in range(num_layers)];

                    v_w = [self.beta2*v_w[i] + (1-self.beta2)*(dW[i])**2 for i in range(num_layers)];
                    v_b = [self.beta2*v_b[i] + (1-self.beta2)*(dB[i])**2 for i in range(num_layers)];

                    m_w_hat = [m_w[i]/(1-self.beta1**(epoch+1)) for i in range(num_layers)];
                    m_b_hat = [m_b[i]/(1-self.beta1**(epoch+1)) for i in range(num_layers)];

                    v_w_hat = [v_w[i]/(1-self.beta2**(epoch+1)) for i in range(num_layers)];
                    v_b_hat = [v_b[i]/(1-self.beta2**(epoch+1)) for i in range(num_layers)];

                    self.weights = [self.weights[i] - (self.learning_rate/np.sqrt(v_w[i]+self.epsilon))*m_w_hat[i] for i in range(len(self.weights))];
                    self.biases = [self.biases[i] - (self.learning_rate/np.sqrt(v_b[i]+self.epsilon))*m_b_hat[i] for i in range(len(self.biases))];

                    dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];
                    dB = [np.zeros((1,self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];


            elapsed = time.time() - start_time;
            #print("Elapsed",elapsed);

            Y_pred = self.predict(X_train);
            training_loss.append(np.mean(loss))
            training_accuracy.append(self.accuracy(Y_train, Y_pred, self.num_train_data));
            Y_pred_valid = self.predict(x_valid);
            validation_accuracy.append(self.accuracy(Y_valid, Y_pred_valid,self.num_validation_data));
            for j in range(self.num_validation_data):
              valid_loss.append(self.calculateCrossEntropyLoss(Y_valid[j].reshape(1,self.num_outputs), Y_pred_valid[j]) + self.L2RegularisationLoss());
            print("Epoch:",epoch," Loss:",training_loss[epoch]," Training accuracy:", training_accuracy[epoch]," Validation Accuracy:",validation_accuracy[epoch] , " Time:",elapsed," Learning Rate:",self.learning_rate, "Validation loss:",np.mean(valid_loss));

            wandb.log({'loss':np.mean(loss), 'training_accuracy':training_accuracy[epoch], 'validation_accuracy':validation_accuracy[epoch],'epoch':epoch,'validation_loss':np.mean(valid_loss) });
        return training_loss, training_accuracy, validation_accuracy, Y_pred;

  def nadam(self):
        training_loss = [];
        training_accuracy = [];
        validation_accuracy = [];

        num_layers = len(self.num_nodes)-1;

        X_train = self.x_train;
        Y_train = np.zeros((self.num_train_data,self.num_outputs));
        Y_valid = np.zeros((self.num_validation_data,self.num_outputs));

        for i in range(self.num_train_data):
          value = self.y_train[i];
          Y_train[i][int(value)] = 1.0;

        for i in range(self.num_validation_data):
          value = self.y_valid[i];
          Y_valid[i][int(value)] = 1.0;

        v_w  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        v_b = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        v_w_hat  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        v_b_hat = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        m_w  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        m_b = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        m_w_hat  = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
        m_b_hat = [np.zeros((1,self.num_nodes[l+1]))for l in range(len(self.num_nodes) - 1)];

        num_points_seen = 0;
        valid_loss = [];
        mse_loss = [];
        for epoch in range(self.num_epochs):
            start_time = time.time();

            loss = [];
            dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes) - 1)];
            dB = [np.zeros((1,self.num_nodes[l + 1]))for l in range(len(self.num_nodes) - 1)];

            for i in range(self.num_train_data):
                Y, H, A = self.forwardPropagate(X_train[i],self.weights,self.biases);
            
                grad_weights, grad_biases = self.backPropagate(Y, H, A, Y_train[i].reshape(1,self.num_outputs));
               
                dW = [ grad_weights[num_layers - 1 - i].T + dW[i] for i in range(num_layers)];
                dB = [ grad_biases[num_layers - 1 - i].T + dB[i] for i in range(num_layers)];

         
                if self.loss_fn == "MSE":
                    loss.append(self.calculateMeanSquaredErrorLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                elif self.loss_fn == "crossEntropy":
                    loss.append(self.calculateCrossEntropyLoss(Y_train[i].reshape(1,self.num_outputs), Y) + self.L2RegularisationLoss());
                
                num_points_seen += 1;

                if num_points_seen % self.batch_size == 0:

                    m_w = [self.beta1*m_w[i] + (1-self.beta1)*dW[i] for i in range(num_layers)];
                    m_b = [self.beta1*m_b[i] + (1-self.beta1)*dB[i] for i in range(num_layers)];

                    v_w = [self.beta2*v_w[i] + (1-self.beta2)*(dW[i])**2 for i in range(num_layers)];
                    v_b = [self.beta2*v_b[i] + (1-self.beta2)*(dB[i])**2 for i in range(num_layers)];

                    m_w_hat = [m_w[i]/(1-self.beta1**(epoch+1)) for i in range(num_layers)];
                    m_b_hat = [m_b[i]/(1-self.beta1**(epoch+1)) for i in range(num_layers)];

                    v_w_hat = [v_w[i]/(1-self.beta2**(epoch+1)) for i in range(num_layers)];
                    v_b_hat = [v_b[i]/(1-self.beta2**(epoch+1)) for i in range(num_layers)];

                    self.weights = [self.weights[i] - (self.learning_rate/(np.sqrt(v_w_hat[i])+self.epsilon))*(self.beta1*m_w_hat[i]+ (1-self.beta1)*dW[i]) for i in range(len(self.weights))];
                    self.biases = [self.biases[i] - (self.learning_rate/(np.sqrt(v_b_hat[i])+self.epsilon))*(self.beta1*m_b_hat[i] + (1-self.beta1)*dB[i]) for i in range(len(self.biases))];

                    dW = [np.zeros((self.num_nodes[l], self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];
                    dB = [np.zeros((1,self.num_nodes[l+1])) for l in range(len(self.num_nodes)-1)];


            elapsed = time.time() - start_time;
         

            Y_pred = self.predict(X_train);
            training_loss.append(np.mean(loss))
            training_accuracy.append(self.accuracy(Y_train, Y_pred, self.num_train_data));
            Y_pred_valid = self.predict(x_valid);
            validation_accuracy.append(self.accuracy(Y_valid, Y_pred_valid,self.num_validation_data));
            for j in range(self.num_validation_data):
              valid_loss.append(self.calculateCrossEntropyLoss(Y_valid[j].reshape(1,self.num_outputs), Y_pred_valid[j]) + self.L2RegularisationLoss());
            print("Epoch:",epoch," Loss:",training_loss[epoch]," Training accuracy:", training_accuracy[epoch]," Validation Accuracy:",validation_accuracy[epoch] , " Time:",elapsed," Learning Rate:",self.learning_rate, "Validation loss:",np.mean(valid_loss));

            wandb.log({'loss':np.mean(loss), 'training_accuracy':training_accuracy[epoch], 'validation_accuracy':validation_accuracy[epoch],'epoch':epoch,'validation_loss':np.mean(valid_loss) });
        return training_loss, training_accuracy, validation_accuracy, Y_pred;



