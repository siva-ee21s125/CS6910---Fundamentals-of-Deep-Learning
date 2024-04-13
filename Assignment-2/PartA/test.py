from ConvolutionNN import PreProcessDataset
from ConvolutionNN import convolutionNN

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


def main():
    wandb.init(project='EE21S125_DL_A2');
    #Preprocessing Parameters
    # augument = wandb.config.data_augmentation;
    # num_dense_neurons = wandb.config.num_dense_neurons;
    # learning_rate = wandb.config.learning_rate;
    # dropout = wandb.config.dropout;
    # activation = wandb.config.activation;
    # num_filters = wandb.config.num_filters;
    # filter_organization = wandb.config.filter_organization;
    # batch_norm = wandb.config.batch_normalization;
    # num_epochs = wandb.config.num_epochs;
    # weight_decay = wandb.config.weight_decay;
    # optimizer = wandb.config.optimizer;

    augument = True;
    num_dense_neurons = 128;
    learning_rate = 0.0001;
    dropout = 0.2;
    activation = 'ReLU';
    num_filters = 32;
    filter_organization = 'same';
    batch_norm = True;
    num_epochs = 15;
    weight_decay = 0;
    optimizer = 'NAdam'


    dataset_directory = 'inaturalist_12K';
    image_size = 256;
    batch_size = 128;
    num_input = 3;
    num_output = 10;
    num_layers = 5;
    filter_size = 3;
    maxpool_size = 2;
    mode = 'test';

    wandb.run.name = ("Test_"+"Aug_"+str(augument)+
                      "_bn_"+str(batch_norm)+
                      "_fcn_"+str(num_dense_neurons)+
                      "_do_"+str(dropout)+
                      "_act_"+str(activation)+
                      "_nf_"+str(num_filters)+
                      "_fo_"+str(filter_organization)+
                      "_lr_"+str(learning_rate)
                      );

    training_set = PreProcessDataset(dataset_directory, image_size, mode, augument);
    train_size = int(0.8*len(training_set));
    validate_size = len(training_set) - train_size;
    train_dataset, validate_dataset = torch.utils.data.random_split(training_set, [train_size, validate_size]);
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0);
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0);

    mode = 'test';
    test_dataset = PreProcessDataset(dataset_directory, image_size, mode, augument);
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0);

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    print("Device:", device);

    # Creating an instance of the model
    model = convolutionNN(image_size, num_input, num_output, filter_organization, batch_norm, num_layers, num_filters, filter_size, maxpool_size, dropout, activation, num_dense_neurons).to(device);

    criterion = torch.nn.CrossEntropyLoss();
    if optimizer == 'NAdam':
      optimizer=torch.optim.NAdam(model.parameters(),lr=learning_rate,weight_decay=weight_decay);
      anneal = CosineAnnealingLR(optimizer, T_max=num_epochs/2, eta_min = 0.0001);
    elif optimizer == 'SGD':
      optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=weight_decay);
      anneal = CosineAnnealingLR(optimizer, T_max=num_epochs/2, eta_min = 0.0001)

    validation_mode = 'train';
    for epoch in range(num_epochs):
      start_time = time.time();
      train_loss,train_accuracy = train(model,train_loader,optimizer,criterion,device);
      validate_loss,validate_accuracy = validate(model,validate_loader,criterion,device,validation_mode);
      anneal.step();
      end_time = time.time();
      elapsed_time = end_time - start_time;
      print("Epoch Time Taken:", elapsed_time);
      print("Training Accuracy:", train_accuracy," Training Loss:", train_loss, " Validation Accuracy:", validate_accuracy, " Validate Loss:", validate_loss," Epoch:", epoch +1);
      wandb.log({'train_loss': train_loss, 'train_accuracy':train_accuracy, 'validate_loss': validate_loss, 'validate_accuracy': validate_accuracy,'epoch':epoch + 1})
    validation_mode = 'test';
    test_loss,test_accuracy = validate(model,test_loader,criterion,device,validation_mode);
    print("Test Accuracy:", test_accuracy, " test Loss:", test_loss);
    wandb.log({'test_loss':test_loss,'test_accuracy':test_accuracy});
    wandb.finish();
    print("TESTING COMPLETE");




if __name__ == "__main__":
    sweep_config = {
                    'method': 'random',  # Choose 'grid', 'random', 'bayes', etc.
                    'metric': {
                                'name': 'validate_accuracy',
                                'goal': 'maximize'
                              },
                    'parameters': {
                        'num_filters': {'values': [32]},
                        'activation': {'values': ['ReLU']},
                        'filter_organization': {'values': ['same']},
                        'data_augmentation': {'values': [True]},
                        'batch_normalization': {'values': [True]},
                        'num_dense_neurons':{'values': [128]},
                        'dropout': {'values': [0.2]},
                        'learning_rate': {'values': [1e-4]},
                        'weight_decay': {'values': [0]},
                        'optimizer': {'values': ['NAdam']},
                        'num_epochs': {'values': [15]}

                    }
                  }
    sweep_id = wandb.sweep(sweep= sweep_config, project='EE21S125_DL_A2');
    wandb.agent(sweep_id, function = main,count=1);
    #main();

