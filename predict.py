import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from PIL import Image
import json
from os import listdir
from collections import OrderedDict
import seaborn as sns

image_path = 'flowers/test/10/image_07090.jpg'
checkpoint = 'checkpoint.pth'
topk = 5
category_names = 'cat_to_name.json'
gpu = 'TRUE'

# Function Definitions
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--topk", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    
    return parser.parse_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filename):
 
    # Checkpoint for when using GPU
    checkpoint = torch.load(filename)
    
    model = models.vgg11 (pretrained = True)
    # Freeze Parameters
 
    for param in model.parameters():
        param.requires_grad = False
  
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    print("checkpoint loaded!")
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image (test image) for use in a PyTorch model
    pil_image = Image.open(image_path)
    
    img_loader = transforms.Compose([transforms.Resize(256), 
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
        
    pil_trans = img_loader(pil_image).float()

    # Converrt to numpy
    np_image = np.array(pil_trans)    
    print( "image read.")
    return np_image   

def predict(image_path, model,cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Loading model - using .cpu() for working with CPUs
    model = model.cpu()
    # Pre-processing image
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    model.eval()
    with torch.no_grad():
        # Running image through network
        output = model.forward(img)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
    top_flowers = [cat_to_name[str(lab)] for lab in classes_top_list]
    print("prediction done.")
    return probs_top_list, classes_top_list, top_flowers

def main():
   
    input = get_args()
    
    # load --json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print("JSON loaded.")    
    # load trained model
    model = load_checkpoint(checkpoint)
    
    # Process images, predict classes, and display results
    image = process_image(image_path)
    probs_top_list, classes_top_list, top_flowers = predict(image_path, model,cat_to_name, topk)
    print("Flower Name:")
    print(top_flowers)
    print("Associated Image Class:")
    print(classes_top_list)
    print("Associated Probabilty:")
    print(probs_top_list)
    
main ()