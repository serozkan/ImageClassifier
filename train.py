# Imports here

from extras import get_input_for_training
import time
import torch
import os, random
import matplotlib.pyplot as plt 
import numpy as np
from torchvision import datasets, models, transforms
from torch import nn, optim
from PIL import Image


in_arg = get_input_for_training()

data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])

modified_transforms = transforms.Compose([
                              transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])
#Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=modified_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=modified_transforms)
test_data  = datasets.ImageFolder(test_dir, transform=modified_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloaders  = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)




import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
# TODO: Build and train your network
cnnarch=in_arg.arch
    # select model
if cnnarch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif cnnarch == 'vgg11_bn':
    model = models.vgg11_bn(pretrained=True)
else:
    print('select: "vgg11_bn" or "vgg16"')
print(model)

hu = in_arg.hidden_units
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hu)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hu, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
 
learnRate = in_arg.learn_rate    
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learnRate)
    
    
# TRAINING
device_selection = in_arg.gpu
print('Selected device:')

if(device_selection is True):
    device = torch.device("cuda:0")
elif (device_selection is False):
    device = torch.device("cpu")

print (device)


epochs = in_arg.epochs
accuracy_his= []   #history for plot

print_every = 40
steps = 0

model.to(device)
model.train()


start = time.time()
print('started')    
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloaders:
        steps += 1

        inputs   = images.to(device) 
        labels   = labels.to(device)

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            accuracy = 0
            valid_loss=0
            model.eval()
            with torch.no_grad():
                for images, labels in validloaders:
                
                    images = images.to(device) #GPU
                    labels = labels.to(device) #GPU
                    
                    output = model(images)

                    loss = criterion(output, labels)
                    valid_loss += loss.item()


                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(1)[1])
                  
                    accuracy += torch.mean(equality.type(torch.FloatTensor))    
                    
            print(  "Epoch: {} of {}... ".format(e+1, epochs),
                    "Train Loss: {:.4f}".format(running_loss/print_every),
                    "Validation Loss: {:.4f}  ".format(valid_loss/len(validloaders)),
                    "\nValidation Accuracy: {:.4f}".format(accuracy/len(validloaders) * 100))
            accuracy_his.append(accuracy/len(validloaders) * 100)
            running_loss = 0
print("TRAINING COMPLETE")

elapsed_time = time.time() - start
print("\n Time spent: {:.0f}m {:.0f}s".format(elapsed_time//60, elapsed_time % 60))






# TODO: Do validation on the test set
accuracy = 0

with torch.no_grad():
    for images, labels in testloaders:

        images = images.to(device) #GPU
        labels = labels.to(device) #GPU

        output = model(images)

        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
                  
        accuracy += torch.mean(equality.type(torch.FloatTensor)) 

print(f"Test accuracy: {accuracy/len(testloaders)* 100:.3f}%")

# TODO: Save the checkpoint 

model.class_to_idx = train_data.class_to_idx

checkpoint = {'transfer_model': model,
              'input_size': 25088,
              'output_size': 102,
              'features': model.features,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epochs' : 'epochs',
              'learnRate': learnRate,
              'idx_to_class': {v: k for k, v in train_data.class_to_idx.items()}
             }

torch.save(checkpoint, in_arg.save_dir + 'checkpoint.pth')
