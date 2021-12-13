from extras import get_input_for_predict
import time
import torch
import os, random
import numpy as np
from torchvision import datasets, models, transforms
from torch import nn, optim
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import pylab as plt  #plots are not shown in Part2 - Workspace somehow, but they're working normally on Jupyter notebook and local.
matplotlib.pyplot.show(block=True) 

out_arg = get_input_for_predict()
img_path = out_arg.image_dir
top_k  = out_arg.top_k
category = out_arg.category_names
device_selection = out_arg.gpu
print('Selected device:')

if(device_selection is True):
    device = torch.device("cuda:0")
elif (device_selection is False):
    device = torch.device("cpu")

print (device)
import json
with open(category, 'r') as f:
    cat_to_name = json.load(f)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['transfer_model']
    #model = models.vgg16(pretrained=True)
    learnRate = checkpoint['learnRate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['idx_to_class']
    model.load_state_dict(checkpoint['state_dict'])  

    return model, checkpoint, learnRate

checkpointname = out_arg.load_dir
print('\nCheckPoint path:')
print(checkpointname)

model, checkpoint, learnRate = load_checkpoint(checkpointname)
print(learnRate)
optimizer = optim.Adam(model.classifier.parameters(), lr=learnRate)
criterion = nn.NLLLoss()

print(model)



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    image = process_image(image_path)
    
    model.to(device)
    image = torch.from_numpy(image).float()
    #print(image.shape) #test
    shape = image.unsqueeze(0).float().to(device)
    #print(shape.shape)
    output = model(shape)
    
            
    probability, index = torch.exp(output).topk(top_k)

    probs = probability[0].tolist()
    classes = index[0].add(1).tolist()
        
    return probs, classes


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im_processed = Image.open(image_path)
    im_processed = im_processed.resize((256,256))
    cropleft = (256-224) * 0.5
    cropbottom = (256-224) * 0.5
    croptop = cropbottom + 224
    cropright = cropleft + 224
    im_processed = im_processed.crop((cropleft,cropbottom,croptop,cropright))
    
    numpy_im = np.array(im_processed)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_im = (numpy_im - mean) / std

    return numpy_im.transpose(2,0,1)
    
    # TODO: Process a PIL image for use in a PyTorch model

with  Image.open(img_path) as image:
    plt.imshow(image)
    
predicted_list=[]
    
prob, classes = predict(img_path, model,top_k )
print(prob)
print(classes)
#print(cat_to_name)

#Modified below to print text to increase readability
k=0 
for i in classes:
    
    i = str(i)
    print("Class:"+ i,  cat_to_name[i], '\n          Prediction probability: {:.3f}%'.format(prob[k]*100))
    k += 1
    predicted_list.append(cat_to_name[i])
    
    
#SANITY CHECK
# TODO: Display an image along with the top 5 classes
probs, classes = predict(img_path, model, top_k)
max_ind = np.argmax(probs)
#print(max_ind)
label = classes[max_ind]

plt.figure(figsize=(14,6))

title = predicted_list[max_ind]

plt.subplot(1, 2, 1)
import matplotlib.image as mpimg
img=mpimg.imread(img_path)
imgplot = plt.imshow(img)



fig, ax = plt.subplots()
ax.barh(predicted_list, probs, align='center')
plt.xlabel('Probability')
plt.ylabel('Flower type')
plt.title(title)

ax.invert_yaxis()  

plt.show()

