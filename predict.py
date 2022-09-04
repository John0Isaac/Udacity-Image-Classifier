import argparse
import json
import torch
from torch.nn import Sequential, ReLU, Dropout, LogSoftmax, Linear
from collections import OrderedDict
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_input_args():
    parser = argparse.ArgumentParser()
    # Required Arguments
    parser.add_argument('image_path', type = str)
    parser.add_argument('checkpoint_path', type = str)
    
    # Optional Arguments
    parser.add_argument('--top_k', type = int, default = 3,
                        help = 'top K most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'category names path')
    parser.add_argument('--gpu', action='store_true', default = False,
                        help = 'choose gpu for training')
    return parser.parse_args()

# Get inputs from Command Line
inputs = get_input_args()
image_path = inputs.image_path
checkpoint_path = inputs.checkpoint_path
top_k = inputs.top_k
category_names = inputs.category_names
gpu = inputs.gpu

def load_and_build(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    # Create a new model
    vgg_model = models.vgg13(pretrained=True)
    # Freeze the model layers
    for param in vgg_model.parameters():
        param.requires_grad = False
    
    vgg_model.class_to_idx = checkpoint['class_to_idx']
    
    vgg_model.classifier = Sequential(OrderedDict([
                                    ('hidden_layer_1', Linear(25088, 120)),
                                    ('relu_1', ReLU()),
                                    ('dropout', Dropout(0.2)),
                                    ('hidden_layer_2', Linear(120, 60)),
                                    ('relu_2', ReLU()),
                                    ('hidden_layer_3', Linear(60, 102)),
                                    ('outputs', LogSoftmax(dim=1))]))
    
    vgg_model.load_state_dict(checkpoint['state_dict'])
    
    return vgg_model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    original_image = Image.open(image_path)
    
    transform_image = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
                                ])
    transformed_image = transform_image(original_image)
    
    processed_image = np.array(transformed_image)
    
    processed_image = processed_image.transpose((0, 2, 1))
    
    return processed_image

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        model.to('cuda')
    # TODO: Implement the code to predict the class from an image file
    else:
        model.to('cpu')
    model.eval()
    if gpu:
        image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to('cuda')
    # TODO: Implement the code to predict the class from an image file
    else:
        image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to('cpu')
    
    
    with torch.no_grad():
        output = model.forward(image)
    
        probabilty = torch.exp(output)

        highest_prob, highest_indecies = probabilty.topk(topk)
        
        highest_prob = (highest_prob[0]).tolist()
        highest_indecies = (highest_indecies[0]).tolist()
        highest_classes = []
        idx_to_class = {value:key for key, value in model.class_to_idx.items()}
        
        for index in highest_indecies:
            highest_classes.append(idx_to_class[int(index)])
        
    return highest_prob, highest_classes


model = load_and_build(checkpoint_path)
highest_prob, highest_classes = predict(image_path, model, top_k, gpu)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
heighest_flowers = [cat_to_name[h_class] for h_class in highest_classes]

print('Flowers list {}, Probability List {}'.format(heighest_flowers, highest_prob))