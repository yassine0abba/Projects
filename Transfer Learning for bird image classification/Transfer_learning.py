#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning

# In[1]:


# Packages 
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms , models
from torchvision.utils import save_image
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.autograd import Variable
import PIL.Image as Image
from torchvision import models
import time
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import glob
import matplotlib.image as mpimg


# In[2]:


root='../input/mybirds/Recvis/train_images/'
dirlist = np.sort([ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ])
label2classes = {k:dirlist[k] for k in range(len(dirlist)) }
classes2label = {dirlist[k]:k for k in range(len(dirlist)) }


# In[3]:


normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

train_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.50),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                normalize,
            ])

valid_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.50),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                normalize,
            ])
train_dataset = ImageFolder(
            root='../input/mybirds/Recvis/train_images', 
            transform=train_transform,
        )

valid_dataset = ImageFolder(
            root='../input/mybirds/Recvis/val_images', 
            transform=valid_transform,
        )


# In[14]:


batch_size = 10

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, shuffle=True,
        num_workers=1, pin_memory=True,
    )

valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=10,
        num_workers=1, pin_memory=True,
    )


# In[15]:


# CUDA MEMORY
torch.cuda.memory_summary(device=None, abbreviated=False)


# In[23]:


# model ResNet 101
# I used the resnet model as embedding

model = models.resnet101() 
num_output = model.fc.in_features
model.lay1 = nn.Linear(num_output, 2400) 
model.lay2 = nn.ReLU(inplace=True)
model.lay3 = nn.Linear(2400, 220, bias=False)
model.lay4 = nn.ReLU(inplace=True)
model.lay5 = nn.Tanh()
model.lay6 = nn.Linear(220, 20, bias=False)
# train on GPU
if torch.cuda.is_available():                                 
    model = model.cuda()       

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Optimizer and Loss function
loss_fn = nn.CrossEntropyLoss()          
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)


# In[30]:


best_acc_train = 0.0
best_acc_valid = 0.0

def train(num_epochs):
    global best_acc_train
    global best_acc_valid
    for epoch in range(num_epochs):
        print("Epoch : ",str(epoch))
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            
            train_acc += (prediction == labels).sum().item()

        train_acc = train_acc / 1082
        train_loss = train_loss / 1082
        ## Compute the accuracy on validation set
        valid_acc = test()
        model.eval()
        valid_acc = 0.0
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _,prediction = torch.max(outputs.data, 1)
            valid_acc += (prediction == labels).sum().item()

        valid_acc = valid_acc / 103
        # Seve the model
        if valid_acc >= best_acc_valid:
            torch.save(model.state_dict(), "model_{}.model".format(epoch))
            print("New model saved : ", "model_{}.model".format(epoch))
            best_acc_valid = test_acc
            best_acc_train = train_acc



# In[31]:


# training
num_epochs = 100
train(num_epochs)


# In[36]:


# Reload the best model (last saved model)
model.load_state_dict(torch.load('./model_80.model'))


# In[49]:


# Generate the submission csv
test_dir = '../input/mybirds/Recvis/test_images/mistery_category'
output_file = open("kaggle.csv", "w")
output_file.write("Id,Category\n")

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

for f in os.listdir(test_dir):
    if 'jpg' in f:
        data = valid_transform(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if torch.cuda.is_available():
            data = data.cuda()
        model.eval()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()


# In[ ]:




