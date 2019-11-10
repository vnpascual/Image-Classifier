import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict

data_dir = 'data_directory'
save_to = 'checkpoint.pth'
pretrained_model = 'vgg11'
learning_rate = 0.001
epochs = 3
hidden_layers = 500
output_size = 102
gpu = 'TRUE'
drop = 0.2


def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_directory", type=str, default = 'flowers',
                        help="data directory containing training and testing data")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth",
                        help= "directory where to save trained model and hyperparameters")
    parser.add_argument("--arch", type=str, default="vgg11",
                        help="pre-trained model: vgg11")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of epochs to train model")
    parser.add_argument("--hidden_units", type=list, default=500,
                        help="list of hidden layers")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    parser.add_argument("--output", type=int, default=102,
                        help="enter output size")
    
    return parser.parse_args()

def save_model(model, arch, hidden_units, train_data, optimizer, epochs, save_to):
     model.class_to_idx = train_data.class_to_idx
     checkpoint = {'model_name': 'model',
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer_dict':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epoch': epochs,
              'arch': arch,
              'hidden_units': hidden_units}

     torch.save(checkpoint, save_to) 
     print("Model is saved!")
    
def create_model(arch = 'vgg11', hidden_units = 500):
    #PRE-TRAIN MODEL
    # Load pre-trained model
    model = getattr(models, arch)(pretrained = True)
    in_features = model.classifier[0].in_features
    # Freeze Parameters
 
    for param in model.parameters():
        param.requires_grad = False
  
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)# define criterion and optimizer
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    return model, criterion, optimizer, scheduler
    print("Your model is created.")

def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    
    # change model to work with cuda
    model.to('cuda')

    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validloader):
    
        # Change images and labels to work with cuda
        images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy
    print( "Validation complete!")        
    
def train(model, trainloader, validloader, criterion, optimizer, epochs):
    steps = 0
    print_every = 30
    model.to('cuda')
    print("Training process is starting.")

    for e in range(epochs):
        running_loss = 0
    
    # Iterating over data to carry out training step
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            # zeroing parameter gradients
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            # Carrying out validation step
     
            if steps % print_every == 0:
            # setting model to evaluation mode during validation
                model.eval()
            
            # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
                
                print('Epoch: {}/{}'.format(e+1, epochs), 
                'Training Loss:         {:.4f}'.format(running_loss/print_every),
                'Validation Loss: {:.4f}'.format(valid_loss/len(validloader)),
                'Validation Accuracy: {:.4f}'.format(accuracy/len(validloader)))
            
                running_loss = 0
                model.train() # Turning training back on
    print("\nTraining process is now complete!!")
    return model        
    

def test(model, trainloader):
    correct = 0
    total = 0
    model.to('cuda')

    with torch.no_grad():
        model.eval()
        for data in trainloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Test Accuracy:{}'.format(100 * correct / total))

def main_function(arch = 'vgg11', hidden_units = 500, learning_rate=0.001):
    input = get_args()
    # LOAD DATA 
    # load and process images from data directory
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' 
    #Load Data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)

    # The trainloader will have shuffle=True so that the order of the images do not affect the model
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=20)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    class_to_idx = train_data.class_to_idx
        #PRE-TRAIN MODEL
    # Load pre-trained model
    model =  getattr(models,arch)(pretrained=True)
    in_features = model.classifier[0].in_features
    # Freeze Parameters
 
    for param in model.parameters():
        param.requires_grad = False
  
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2',nn.Linear(hidden_units,512)),
                          ('relu2',nn.ReLU()),
                          ('Dropout2',nn.Dropout(p=0.15)),
                          ('fc3',nn.Linear(512,102)),
                          ('output',nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    model.to('cuda')
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) # define criterion and optimizer
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    print("Your model is created.")
    model = train(model, trainloader, validloader, criterion, optimizer, epochs)
    
    test(model, trainloader)
    save_model(model,arch, hidden_units, train_data, optimizer, epochs, save_to)                                    

main_function(arch = 'vgg11', hidden_units = 500, learning_rate=0.001)                                
                                             
