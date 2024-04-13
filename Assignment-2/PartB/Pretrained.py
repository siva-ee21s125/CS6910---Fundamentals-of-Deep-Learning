!pip install wandb -qU
import wandb
wandb.login();

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from glob import glob
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

class PreProcessDataset(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir;
        self.mode = mode;
        self.prepareDataset();
        self.imageTransform();

    def prepareDataset(self):
        self.files = [];
        self.labels = [];
        self.classes = [];

        directory_path = os.path.join(self.root_dir, 'train');
        folders = sorted(os.listdir(directory_path));

        for i in folders:
          if i[0]!= ".":
            self.classes.append(i);
        print("Classes:",self.classes);
        if self.mode == 'train':
          for i in range(len(self.classes)):
              files = glob(os.path.join(directory_path, self.classes[i], '*.jpg'));
              self.labels += [i]*len(files);
              self.files += files;
        if self.mode == 'test':
          directory_path = os.path.join(self.root_dir, 'val');
          for i in range(len(self.classes)):
              files = glob(os.path.join(directory_path, self.classes[i], '*.jpg'));
              self.labels += [i]*len(files);
              self.files += files;

    def imageTransform(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]);

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB');
        img = self.transform(img);
        label = self.labels[index];
        label = torch.tensor(label, dtype = torch.long);
        return img, label;

    def __len__(self):
        return len(self.files);

def train(model, dataset, optimizer, criterion, device):
    epoch_loss = 0
    epoch_correct_predictions = 0
    epoch_total_samples = 0
    model.train();
    for i, (images,labels) in enumerate(dataset):
        #setting gradients to zero again to prevent any incorrect calculation
        optimizer.zero_grad();
        images, labels = images.to(device), labels.to(device);
        predicted = model(images);
        loss = criterion(predicted,labels);
        loss.backward();
        optimizer.step();

        _, predicted_labels = torch.max(predicted, 1);
        correct_predictions = (predicted_labels == labels).sum().item();
        batch_accuracy = correct_predictions / len(labels);

        epoch_loss += loss.item() * len(labels);
        epoch_correct_predictions += correct_predictions;
        epoch_total_samples += len(labels);


    epoch_loss /= epoch_total_samples;
    epoch_accuracy = epoch_correct_predictions / epoch_total_samples;

    return epoch_loss, epoch_accuracy;


def validate(model, dataset, criterion, device):

    epoch_loss = 0
    epoch_correct_predictions = 0
    epoch_total_samples = 0
    model.train();
    for i, (images,labels) in enumerate(dataset):
        images, labels = images.to(device), labels.to(device);
        predicted = model(images);
        loss = criterion(predicted,labels);

        _, predicted_labels = torch.max(predicted, 1);
        correct_predictions = (predicted_labels == labels).sum().item();
        batch_accuracy = correct_predictions / len(labels);


        epoch_loss += loss.item() * len(labels);
        epoch_correct_predictions += correct_predictions;
        epoch_total_samples += len(labels);


    epoch_loss /= epoch_total_samples;
    epoch_accuracy = epoch_correct_predictions / epoch_total_samples;

    return epoch_loss, epoch_accuracy;


def main():
    wandb.init(project='EE21S125_DL_A2_PartB');
    NN = wandb.config.nn;
    num_dense_neurons = wandb.config.num_dense_neurons;

    num_epochs = 15;
    dataset_directory = 'inaturalist_12K';
    batch_size = 128;

    mode = 'train';
    num_classes = 10;

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    print("Device:", device);

    training_set = PreProcessDataset(dataset_directory,mode);
    train_size = int(0.8*len(training_set));
    validate_size = len(training_set) - train_size;

    train_dataset, validate_dataset = torch.utils.data.random_split(training_set, [train_size, validate_size]);

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10);
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=10);

    mode = 'test';
    test_dataset = PreProcessDataset(dataset_directory,mode);
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers = 10)


    # Load pre-trained ResNet50 model
    if NN == 'RESNET50':
      model = models.resnet50(pretrained=True)
    elif NN == 'GOOGLENET':
      model = models.googlenet(pretrained = True);

    # Freeze all layers except the last layer
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last layer for the new dataset
    num_ftrs = model.fc.in_features;
    model.fc = nn.Sequential(
        torch.nn.Linear(num_ftrs, num_dense_neurons),
        torch.nn.Mish(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_dense_neurons, num_classes),
    );

    model = model.to(device);

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss();
    optimizer = optim.NAdam(model.fc.parameters(), lr=0.001);
    anneal = CosineAnnealingLR(optimizer, T_max=num_epochs/2, eta_min = 0.0001);

    wandb.run.name = ("NN_"+str(NN)+"_dn_"+str(num_dense_neurons));

    for epoch in range(num_epochs):
      start_time = time.time();
      train_loss,train_accuracy = train(model,train_loader,optimizer,criterion,device);
      validate_loss,validate_accuracy = validate(model,validate_loader,criterion,device);
      anneal.step();
      end_time = time.time();
      elapsed_time = end_time - start_time;

      print("Epoch Time Taken:", elapsed_time);
      print("Training Accuracy:", train_accuracy," Training Loss:", train_loss, " Validation Accuracy:", validate_accuracy, " Validate Loss:", validate_loss," Epoch:", epoch +1);
      wandb.log({'train_loss': train_loss, 'train_accuracy':train_accuracy, 'validate_loss': validate_loss, 'validate_accuracy': validate_accuracy, 'epoch':epoch + 1});
    test_loss,test_accuracy = validate(model,test_loader,criterion,device);
    print("Test Accuracy:", test_accuracy, " test Loss:", test_loss);
    wandb.log({'test_loss':test_loss,'test_accuracy':test_accuracy});
    wandb.finish();
    print("COMPLETE");



if __name__ == "__main__":
   sweep_config = {
                    'method': 'grid',
                    'metric': {
                                'name': 'validate_accuracy',
                                'goal': 'maximize'
                              },
                    'parameters': {
                        'nn':{'values':['RESNET50','GOOGLENET']},
                        'num_dense_neurons':{'values': [128,256]},

                    }
                  }
   sweep_id = wandb.sweep(sweep= sweep_config, project='EE21S125_DL_A2_PartB');
   wandb.agent(sweep_id, function = main,count=4);
  #main();

