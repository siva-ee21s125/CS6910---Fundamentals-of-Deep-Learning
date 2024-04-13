!pip install wandb -qU
import wandb
wandb.login();

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from glob import glob
import torch
import torchvision.transforms as transforms
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt



# Dataset Class for Setting up the data loading process
class PreProcessDataset(Dataset):
    def __init__(self, root_dir, image_size, mode = 'train', augument = 'True'):
        self.root_dir = root_dir;
        self.mode = mode;
        self.augument = augument;
        self.size = image_size;
        self.prepareDataset();
        # if(mode=='test'):
        #   self.plotImages();
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

        elif self.mode == 'test':
            directory_path = os.path.join(self.root_dir, 'val');
            for i in range(len(self.classes)):
                files = glob(os.path.join(directory_path, self.classes[i], '*.jpg'));
                self.labels += [i]*len(files);
                self.files += files;
        else:
            print("Invalid Mode");
            return None;

    def imageTransform(self):
       if (self.augument):
          if(self.mode=='train'):
              self.transform = transforms.Compose([
                  transforms.RandomRotation(30),
                  #transforms.RandomResizedCrop(self.size),
                  transforms.CenterCrop(224),
                  transforms.RandomHorizontalFlip(),
                  transforms.Resize((256,256)),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]);
          elif(self.mode=='test'):
            self.transform = transforms.Compose([
              transforms.CenterCrop(224),
              transforms.Resize((256,256)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]);

       else:
        self.transform = transforms.Compose([transforms.RandomResizedCrop(self.size),transforms.ToTensor()]);

        # self.transforms = transforms.Compose([
        #     transforms.RandomRotation(30),
        #     transforms.RandomResizedCrop((256,256)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #      transforms.Normalize([0.2356, 0.2300, 0.1948], [0.1096, 0.1055, 0.1075])
        # ]);

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB');
        img = self.transform(img);
        label = self.labels[index];
        label = torch.tensor(label, dtype = torch.long);
        return img, label;

    def __len__(self):
        return len(self.files);

class convolutionNN(torch.nn.Module):
    def __init__(self, image_size, num_input, num_output, filter_organization, batch_norm, num_layers, num_filters, filter_size, maxpool_size, dropout, activation, num_dense_neurons):
        super(convolutionNN,self).__init__();
        self.image_size = image_size;
        self.num_input = num_input;
        self.num_classes = num_output;
        self.filter_organization = filter_organization;
        self.batch_norm = batch_norm;
        self.num_layers = num_layers;
        self.num_filters = num_filters;
        self.filter_size = filter_size;
        self.maxpool_size = maxpool_size;
        self.dropout = dropout;
        self.num_dense_neurons = num_dense_neurons;

        if activation == 'ReLU':
          self.activation = torch.nn.ReLU;
        if activation == 'GELU':
          self.activation = torch.nn.GELU;
        if activation == "SiLU":
          self.activation = torch.nn.SiLU;
        if activation == "Mish":
          self.activation = torch.nn.Mish;
        if activation == 'LeakyReLU':
          self.activation = torch.nn.LeakyReLU;

        self.filter_padding = 1;
        self.filter_stride = 1;
        self.maxpool_stride = 2;
        self.layers = torch.nn.ModuleList();
        num_filters_used = [self.num_filters];


        for i in range(self.num_layers):
          #Since filter sizes are reducing significantly
          self.filter_padding = (self.filter_size-1)//2;
          self.convolutionBlock();
          self.num_input = self.num_filters;
          if(self.filter_organization == 'half'):
            self.num_filters = self.num_filters//2;
            num_filters_used.append(self.num_filters);
          if(self.filter_organization == 'same'):
            self.num_filters = self.num_filters;
            num_filters_used.append(self.num_filters);
          if(self.filter_organization == 'double'):
            self.num_filters = self.num_filters*2;
            num_filters_used.append(self.num_filters);

        final_dim = self.computeInputSizeToDense();
        conv_output_dim = final_dim * final_dim * num_filters_used[-2];
        print(final_dim, conv_output_dim ,num_filters_used);


        self.FCN = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(conv_output_dim, self.num_dense_neurons),
            self.activation(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.num_dense_neurons, self.num_classes)
            );

    def convolutionBlock(self):
        self.layers.append(torch.nn.Conv2d(self.num_input,self.num_filters, self.filter_size, padding = self.filter_padding));
        self.layers.append(self.activation());
        self.layers.append(torch.nn.MaxPool2d(self.maxpool_size, stride = self.maxpool_stride));
        if(self.batch_norm == True):
          self.layers.append(torch.nn.BatchNorm2d(self.num_filters));

    def computeInputSizeToDense(self):
        input = self.image_size;
        #print("input:",input)
        for i in range(self.num_layers):
          #print("Padding:", self.filter_padding);
          conv_output = np.ceil(float(input - self.filter_size + (2*self.filter_padding))/self.filter_stride) + 1;
          #print("Conv output Dim:",conv_output);
          maxpool_output = np.floor(float(conv_output - self.maxpool_size)/self.maxpool_stride) + 1;
          #print("maxpool_output:",maxpool_output);
          input = maxpool_output;
        return int(maxpool_output);

    def forward(self,x):
      #print("Input:", self.image_size, "x.size():", x.size())
      for layer in self.layers:
        x = layer(x);
        #print("layer: ", layer, "x.size():", x.size())
      x = self.FCN(x);
      return torch.nn.functional.softmax(x,dim=1);



def train(model, dataset, optimizer, criterion, device):

    epoch_loss = 0;
    epoch_correct_predictions = 0;
    epoch_total_samples = 0;
    model.train();
    for i, (images,labels) in enumerate(dataset):
        #setting gradients to zero again
        optimizer.zero_grad();
        #loading the data accordingly
        images, labels = images.to(device), labels.to(device);
        #forward pass
        predicted = model.forward(images);
        #calculate loss
        loss = criterion(predicted,labels);
        #gradient descent
        loss.backward();
        #optimizer updates the parameters
        optimizer.step();
        #print("Lenth labels:",len(labels));
        _, predicted_labels = torch.max(predicted, 1);
        correct_predictions = (predicted_labels == labels).sum().item();
        batch_accuracy = correct_predictions / len(labels);


        epoch_loss += loss.item() * len(labels);
        epoch_correct_predictions += correct_predictions;
        epoch_total_samples += len(labels);

    epoch_loss /= epoch_total_samples;
    epoch_accuracy = epoch_correct_predictions / epoch_total_samples;

    return epoch_loss, epoch_accuracy;



def validate(model, dataset, criterion, device, mode):
    flag = 0;
    epoch_loss = 0;
    epoch_correct_predictions = 0;
    epoch_total_samples = 0;
    model.train();
    for i, (images,labels) in enumerate(dataset):
        #loading the data accordingly
        images, labels = images.to(device), labels.to(device);
        #forward pass
        predicted = model.forward(images);
        #calculate loss
        loss = criterion(predicted,labels);
        #print("Lenth labels:",len(labels));

        _, predicted_labels = torch.max(predicted, 1);
        correct_predictions = (predicted_labels == labels).sum().item();
        batch_accuracy = correct_predictions / len(labels);

        epoch_loss += loss.item() * len(labels);
        epoch_correct_predictions += correct_predictions;
        epoch_total_samples += len(labels);

        if(mode=='test' and flag==0):
            num_samples = 30
            images = images.cpu().numpy()
            predicted_labels = predicted_labels.cpu().numpy()
            labels = labels.cpu().numpy()
            # Select random samples
            idx = np.random.choice(len(images), size=num_samples, replace=False)
            sample_images = images[idx]
            sample_preds = predicted_labels[idx]
            sample_labels = labels[idx]

            # Plot sample images with predictions
            fig, axs = plt.subplots(num_samples, 1, figsize=(6, 2*num_samples))
            for i in range(num_samples):
                ax = axs[i] if num_samples > 1 else axs
                img = transforms.functional.to_pil_image(sample_images[i].transpose(1, 2, 0));
                ax.imshow(img);
                ax.axis('off');
                ax.set_title(f'Actual: {sample_labels[i]}, Predicted: {sample_preds[i]}');
                plt.tight_layout();
            wandb.log({"predictions": [wandb.Image(fig)]});
            flag = 1;

    epoch_loss /= epoch_total_samples;
    epoch_accuracy = epoch_correct_predictions / epoch_total_samples;
    return epoch_loss, epoch_accuracy;








