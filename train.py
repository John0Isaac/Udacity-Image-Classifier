import torch
from torch import nn
from torch.nn import Sequential, ReLU, Dropout, LogSoftmax, Linear
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models

import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    # Required Arguments
    parser.add_argument('data_dir', type = str)
    
    # Optional Arguments
    parser.add_argument('--save_dir', type = str, default = 'save_directory',
                        help = 'directory to save the model checkpoint in')
    parser.add_argument('--arch', type = str, default = 'vgg13',
                        help = 'model architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'hyperparameters')
    parser.add_argument('--hidden_units', type = int, default = 120,
                        help = 'hyperparameters')
    parser.add_argument('--epochs', type = int, default = 10,
                        help = 'hyperparameters')
    parser.add_argument('--gpu', action='store_true', default = False,
                        help = 'choose gpu for training')
    return parser.parse_args()

# Get inputs from Command Line
inputs = get_input_args()
data_dir = inputs.data_dir
save_dir = inputs.save_dir
arch = inputs.arch
learning_rate = inputs.learning_rate
hidden_units = inputs.hidden_units
epochs = inputs.epochs
gpu = inputs.gpu


def transform_data(data_dir):
    print("Started Transforming Data..")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
                                    ])
    valid_data_transforms = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
                                    ])
    test_data_transforms = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.229, 0.224, 0.225), (0.485, 0.456, 0.406))
                                    ])

    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)
    print("Finished Transforming Data..")
    return train_image_datasets, valid_image_datasets, test_image_datasets


def load_data(train_image_datasets, valid_image_datasets, test_image_datasets):
    print("Started Loading Data..")
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=32, shuffle=True)
    print("Finished Loading Data..")
    return train_dataloaders, valid_dataloaders, test_dataloaders

def build_the_model(arch, learning_rate, hidden_units):
    if arch.lower() == 'vgg13':
        print("Started Building VGG 13 Model..")
        base_model = models.vgg13(pretrained=True)
        # Freeze the parameters
        for param in base_model.parameters():
            param.requires_grad = False

        base_model.classifier = Sequential(OrderedDict([
                                            ('hidden_layer_1', Linear(25088, hidden_units)),
                                            ('relu_1', ReLU()),
                                            ('dropout', Dropout(0.2)),
                                            ('hidden_layer_2', Linear(hidden_units, hidden_units//2)),
                                            ('relu_2', ReLU()),
                                            ('hidden_layer_3', Linear(hidden_units//2, 102)),
                                            ('outputs', LogSoftmax(dim=1))]))

        criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(base_model.classifier.parameters(), lr=learning_rate)
    else:
        print("Started Building DenseNet 121 Model..")
        base_model = models.densenet121(pretrained=True)
        # Freeze the parameters
        for param in base_model.parameters():
            param.requires_grad = False

        base_model.classifier = Sequential(OrderedDict([
                                            ('hidden_layer_1', Linear(1024, hidden_units)),
                                            ('relu_1', ReLU()),
                                            ('dropout', Dropout(0.2)),
                                            ('hidden_layer_2', Linear(hidden_units, hidden_units//2)),
                                            ('relu_2', ReLU()),
                                            ('hidden_layer_3', Linear(hidden_units//2, 102)),
                                            ('outputs', LogSoftmax(dim=1))]))

        criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(base_model.classifier.parameters(), lr=learning_rate)
    print("Finished Building Model..")
    return base_model, criterion, optimizer
        
def train_model(base_model, criterion, optimizer, train_dataloaders, valid_dataloaders, epochs, gpu):
    if gpu:
        print("Started Training Using GPU..")
        base_model.to('cuda')

        steps = 0
        training_loss = 0
        print_every = 5

        for epoch in range(epochs):
            for index, (inputs, labels) in enumerate(train_dataloaders):
                steps += 1
                # Move input and label tensors to cuda
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                outputs = base_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    base_model.eval()
                    with torch.no_grad():
                        for index, (inputs, labels) in enumerate(valid_dataloaders):
                            inputs, labels = inputs.to('cuda'), labels.to('cuda')
                            outputs = base_model.forward(inputs)
                            batch_loss = criterion(outputs, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(outputs)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {training_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(valid_dataloaders):.3f}.. "
                          f"Valid accuracy: {accuracy/len(valid_dataloaders):.3f}")
                    training_loss = 0
                    base_model.train()
    else:
        print("Started Training Using CPU..")
        base_model.to('cpu')

        steps = 0
        training_loss = 0
        print_every = 5

        for epoch in range(epochs):
            for index, (inputs, labels) in enumerate(train_dataloaders):
                steps += 1
                # Move input and label tensors to cuda
                inputs, labels = inputs.to('cpu'), labels.to('cpu')

                optimizer.zero_grad()

                outputs = base_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    base_model.eval()
                    with torch.no_grad():
                        for index, (inputs, labels) in enumerate(valid_dataloaders):
                            inputs, labels = inputs.to('cpu'), labels.to('cpu')
                            outputs = base_model.forward(inputs)
                            batch_loss = criterion(outputs, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(outputs)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {training_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(valid_dataloaders):.3f}.. "
                          f"Valid accuracy: {accuracy/len(valid_dataloaders):.3f}")
                    training_loss = 0
                    base_model.train()
    print("Finished Training..")
    return base_model, criterion, optimizer 

    
def test_model(test_dataloaders):
    right_predictions = 0
    total = 0
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(test_dataloaders):
            # Move input and label tensors to cuda
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = base_model(inputs)

            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            right_predictions += (predictions == labels).sum().item()
    print("The Accuracy of the model is: {} %".format((right_predictions/total) * 100))
    
def save_model(train_image_datasets, base_model, optimizer, arch, save_dir):
    base_model.class_to_idx = train_image_datasets.class_to_idx

    checkpoint = {
        'structure' :arch,
        'state_dict': base_model.state_dict(),
        'hidden_layers': [each for each in base_model.classifier],
        'class_to_idx':base_model.class_to_idx,
        'optimizer_dict':optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir+'/checkpoint.pth')
    print("Model saved to {}".format(save_dir+'/checkpoint.pth'))




# Transform data
train_image_datasets, valid_image_datasets, test_image_datasets = transform_data(data_dir)
# Load data
train_dataloaders, valid_dataloaders, test_dataloaders = load_data(train_image_datasets, valid_image_datasets, test_image_datasets)
# Build the Model
base_model, criterion, optimizer = build_the_model(arch, learning_rate, hidden_units)
# Train the Model
base_model, criterion, optimizer = train_model(base_model, criterion, optimizer, train_dataloaders, valid_dataloaders, epochs, gpu)
# Test Model
test_model(test_dataloaders)
# Save Model
save_model(train_image_datasets, base_model, optimizer, arch, save_dir)


