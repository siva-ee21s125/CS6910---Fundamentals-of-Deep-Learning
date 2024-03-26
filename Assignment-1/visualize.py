#Visualizing the dataset
!pip install wandb -qU
import wandb
wandb.login(key='2b25045507d6a89b66edf89be892f3687346ed10')
import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


def visualize(X,Y):
  unique_classes = np.unique(Y);
  class_names = {
        unique_classes[0]: 'T-shirt/top',
        unique_classes[1]: 'Trouser',
        unique_classes[2]: 'Pullover',
        unique_classes[3]: 'Dress',
        unique_classes[4]: 'Coat',
        unique_classes[5]: 'Sandal',
        unique_classes[6]: 'Shirt',
        unique_classes[7]: 'Sneaker',
        unique_classes[8]: 'Bag',
        unique_classes[9]: 'Ankle boot'
  };
  counter = np.zeros(unique_classes.shape);
  plt.figure(figsize=(10, 10));
  for i in range(len(unique_classes)):
    for j in range(len(Y)):
      if((Y[j]==i) and (counter[i]==0)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[j], cmap=plt.get_cmap('gray'));
        plt.xlabel(class_names[i]);
        counter[i] = 1;
  
  plt.show();

def main():
    # Load Fashion-MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data();

    assert x_train.shape == (60000, 28, 28);
    assert x_test.shape == (10000, 28, 28);
    assert y_train.shape == (60000,);
    assert y_test.shape == (10000,);

    visualize(x_train,y_train);



if __name__ == "__main__":
    main()

