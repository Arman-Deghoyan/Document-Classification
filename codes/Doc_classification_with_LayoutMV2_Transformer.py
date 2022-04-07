#!/usr/bin/env python
# coding: utf-8

# # Neccesarry installations and imports 

# In[ ]:


# Note you might need to restrart runtime after installation

# Restart the runtime for the packages to take effect.

# ! pip install imgaug==0.2.5 folium==0.2.1 Pillow==9.0.1

# ! apt install tesseract-ocr
# ! apt install libtesseract-dev
# ! pip install 'git+https://github.com/facebookresearch/detectron2.git'
# !pip install transformers datasets 
# ! pip install pytesseract


# In[ ]:


import os
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data

import PIL
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import LayoutLMv2Processor, LayoutLMv2ForSequenceClassification

from torch import nn
import torch
from tqdm.notebook import tqdm
from torchvision import models


# # How to mount the content of google drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


data_dir = "/content/drive/MyDrive/train/"


# # Copy data to Colab, it's extremely slow to train from gdrive directly.

# In[ ]:


# ! mkdir -p train
# ! cp /content/drive/MyDrive/train/* ./train -r


# # Create a folder to save the weights during training.

# In[ ]:


# ! mkdir -p /content/drive/MyDrive/Layout_MV2_weights


# # Create the image dataset 

# In[ ]:


image_datasets = datasets.ImageFolder(data_dir)


# # Split the data into train and validation 

# In[ ]:


train_set_size = int(len(image_datasets) * 0.8)
valid_set_size = len(image_datasets) - train_set_size
train_set, valid_set = data.random_split(image_datasets, [train_set_size, valid_set_size])


# # Creating the Transformer class 

# In[ ]:


class MV2Transform(object):
    """ Apply MV2 processor on the image.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        
        result = processor(image, return_tensors="pt", max_length=50, padding="max_length", truncation=True)
        
        for key, value in result.items():
            result[key] = torch.squeeze(value)
            
        return result


# # Import the feature extractor and classification model from Hugging Face

# In[ ]:


num_classes = 13

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
LayoutMV2Model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=num_classes) 


# # Construct the desired torch dataset 

# In[ ]:


train_data_transforms = MV2Transform(processor)
test_data_transforms = MV2Transform(processor)

train_set.dataset.transform = train_data_transforms
valid_set.dataset.transform = test_data_transforms


# # Construct the correspongid torch dataloader 

# In[ ]:


batch_size = 16
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)


# # Open an example image from the dataset

# In[ ]:


image = PIL.Image.open(f"{data_dir}/ad/83679.jpg").convert("RGB")
plt.imshow(image)


# # Training phase

# # Auxiliary functions for training process

# In[ ]:


def evaluate(model, test_loader, device):
    with torch.no_grad():
        correct_preds, n_preds = 0, 0
        for batch_idx, (batch_data, batch_labels) in tqdm(enumerate(test_loader), "Evaluating...", total=len(test_loader)):
            for (key, value) in batch_data.items():
                batch_data[key] = value.to(device)
            batch_labels = batch_labels.to(device)
            prediction = model(**batch_data)
            correct_preds += sum(torch.argmax(prediction.logits, dim=1) == batch_labels)
            n_preds += len(batch_labels)
     
    return int(correct_preds) / n_preds


# In[ ]:


def compute_avg_loss(model, loader, device):
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels) in tqdm(enumerate(loader), "Computing validation loss...", total=len(loader)):
            for (key, value) in batch_data.items():
                batch_data[key] = value.to(device)
            batch_labels = batch_labels.to(device)

            prediction = model(**batch_data, labels=batch_labels)
            loss = prediction.loss
            print("Loss is ", loss.item())
            total_loss += loss.item()
            total_count = total_count + 1
     
    return total_loss / total_count


# In[ ]:


print('__CUDA Device Name:',torch.cuda.get_device_name(0))
print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)


# # Move model to device and load the desired weights ( if already have saved weights )

# In[ ]:


device = 'cuda' if torch.cuda.is_available else 'cpu'
LayoutMV2Model.to(device)

# LayoutMV2Model.load_state_dict(torch.load("/content/drive/MyDrive/Layout_MV2_weights/Layout_MV2_weights_0_batch_850.pth"))


# # Initilize epochs, loss and optimizer 

# In[ ]:


n_epochs = 20

# Hyper params taken from the paper, don't touch please.
optimizer = torch.optim.Adam(LayoutMV2Model.parameters(), lr=0.00002, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.3)


# # Train the model 

# In[ ]:


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_losses_batch = []

for epoch in range(n_epochs):
    LayoutMV2Model.train()
    for batch_idx, (batch_data, batch_labels) in tqdm(enumerate(train_loader), f"Training epoch {epoch}", total=len(train_loader)):
        if batch_idx != 0 and batch_idx % 100 == 0:
          # Save every once in a while, colab keeps throwing out.
          torch.save(LayoutMV2Model.state_dict(), f"/content/drive/MyDrive/Layout_MV2_weights/Layout_MV2_weights_{epoch}_batch_{batch_idx}.pth")

        for (key, value) in batch_data.items():
            batch_data[key] = value.to(device)
        batch_labels = batch_labels.to(device)

        outputs = LayoutMV2Model(**batch_data, labels=batch_labels)
        loss = outputs.loss
        prediction = outputs.logits

        train_losses_batch.append(loss.item())
        loss.backward()
        optimizer.step()
        LayoutMV2Model.zero_grad()

    # Free up the GPU so we can run the evals on it.
    del batch_data
    del batch_labels

    LayoutMV2Model.eval()
    torch.save(LayoutMV2Model.state_dict(), f"/content/drive/MyDrive/Layout_MV2_weights/Layout_MV2_weights_{epoch}.pth")

    val_loss = compute_avg_loss(LayoutMV2Model, val_loader, device)
    scheduler.step(val_loss)

    train_losses.append(sum(train_losses_batch) / len(train_losses_batch))
    val_losses.append(val_loss)

    train_accuracies.append( evaluate(LayoutMV2Model, train_loader, device) )
    val_accuracies.append( evaluate(LayoutMV2Model, val_loader, device) )
    


# # Evaluate the model on validation dataset

# In[ ]:


LayoutMV2Model.eval()
val_accuracy = evaluate(LayoutMV2Model, val_loader, device)
print(val_accuracy)


# # Display train and validation losses

# In[ ]:


print('Train losses' + '\n')
print(train_losses)
print('Validation losses' + '\n')
print(val_losses)


# # Plot the lossess and the accuracies both on train and on validation

# In[ ]:


plt.title("Batch loss")
plt.plot(train_losses_batch)

plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

plt.plot(train_accuracies, label='Train accuracy')
plt.plot(val_accuracies, label='Validation accuracy')
plt.legend()
plt.show()


# # Display the train and validation accuracies 

# In[ ]:


print(train_accuracies)
print(val_accuracies)
print(optimizer)

