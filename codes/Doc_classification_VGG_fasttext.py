#!/usr/bin/env python
# coding: utf-8

# # Neccesary imports

# In[ ]:


import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from torch import nn
import torch
from tqdm.notebook import tqdm
import os 
import pandas as pd
# import pytesseract
import numpy as np


# # How to mount google drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# # Copy both images and texts into the current notebook

# In[ ]:


get_ipython().system(' cp /content/drive/MyDrive/train/ . -r')
get_ipython().system(' cp /content/drive/MyDrive/train_texts/ . -r')


# # Create a dataframe with 'text', 'text_path', 'image_path', 'label' columns

# In[ ]:


data_text_dir = "./train_texts/"
data_image_dir = "./train/"

data = pd.DataFrame(columns=['text', 'text_path', 'image_path', 'label'])

for folder in os.listdir(data_text_dir):
    count = 0 
    current_directory = os.path.join(data_text_dir, folder)
    for text_of_image in tqdm(os.listdir(current_directory)):
        text_i_path = os.path.join(data_text_dir + folder, text_of_image)

        text_i = open(os.path.join(data_text_dir + folder, text_of_image),encoding = 'latin-1').read()
        text_i = text_i.replace('\n',' ')
        text_i = text_i.replace(',',' ')

        image_i_path = os.path.join(data_image_dir + folder, text_of_image[:-3] + 'jpg')

        data = data.append({'text': text_i, 'text_path': text_i_path, 'image_path': image_i_path, 'label': folder}, ignore_index=True)


# # Map the classes to corresponding labels from 0 - 12

# In[ ]:


def get_label_mapping(value):
    return class_mapping[value]

class_mapping = {
    'memorandum': 0,
    'email': 1, 
    'cv': 2, 
    'report': 3, 
    'newspaper': 4 ,
    'survey': 5,
    'specification':6,
    'publication':7,
    'invoice':8,
    'letter':9, 
    'ad':10, 
    'handwritten':11,
    'file':12,
}

data['label'] = data['label'].apply(get_label_mapping)

print(data['label'].value_counts())


# # Understand what portion of your dataset contains more than 100 symbols of text extracted 

# In[ ]:


mask = (data['text'].str.len() > 100)

print(data.loc[mask].shape[0] / data.shape[0])


# In[ ]:


print(data.isna().sum())
data = data.dropna()
data.shape


# # Text preprocessing 

# # Neccessary installations and imports 

# In[ ]:


# pip install nltk
# pip install -U pip setuptools wheel
# pip install -U spacy
# python -m spacy download en_core_web_sm


# In[ ]:


from bs4 import BeautifulSoup
from html import unescape
import os
import spacy
import nltk 
from nltk.stem import SnowballStemmer
nltk.download('words')


try:
    spacy_en = spacy.load("en_core_web_sm")
except:
    os.system('python -m spacy download en_core_web_sm')
    spacy_en = spacy.load("en_core_web_sm")
    
snowball = SnowballStemmer("english")
stops_spacy = sorted(spacy.lang.en.stop_words.STOP_WORDS)
stops_spacy.extend(["is", "to"])


# # Define all auxiliary functions for text preprocessing

# In[ ]:


def textLower(text):
    return text.lower()

def remove_punctuation(text):
    
    text = ''.join([char if char.isalnum() or char == ' ' else ' ' for char in text])
    text = ' '.join(text.split())  # remove multiple whitespace
    
    return text

def normalize(text):
    
    # replace urls
    soup = BeautifulSoup(unescape(text), 'html')
    for a_tag in soup.find_all('a'):
        a_tag.string = 'URL'
    
    text = soup.text
    return text

words = set(nltk.corpus.words.words())

def remove_stopwords_spacy(text, stopwords=stops_spacy):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def lemmatize_spacy(text):
    text = spacy_en(text)
    lemmas = [token.lemma_ for token in text]
    return " ".join(lemmas)

def remove_non_eng_words(text):

    return " ".join(w for w in nltk.wordpunct_tokenize(text)              if w.lower() in words or not w.isalpha())


# # Lets apply all the text preprocessings to text column 

# In[ ]:


data["text"] = data["text"].apply(normalize)
print('Stemming done')

data["text"] = data["text"].apply(remove_punctuation)
print('Stemming done')

data["text"] = data["text"].apply(textLower)
print('Stemming done')

data["text"] = data["text"].apply(snowball.stem)
print('Stemming done')

# data["text"] = data["text"].apply(lemmatize_spacy)
# print('Lemmatiztion done')

data["text"] = data["text"].apply(remove_stopwords_spacy)
print('Stopwords removal done')

data["text"] = data["text"].apply(remove_non_eng_words)
print('Non english words removal done')

data.head()


# # Lets print the wordcloud of words and get all the common words 

# In[ ]:


from collections import Counter

def word_counts(text, top_k=15, stopwords=None, only_alpha=False, min_len = 3):
    words = [word for word in text.split(' ') if (word != '') and (len(word)>=min_len)]
    if stopwords is not None:
        stopwords = {stopword.lower() for stopword in stopwords}
        words = [word for word in words if (word not in stopwords) and (len(word)>=min_len)]
    if only_alpha:
        words = [word for word in words if (word.isalpha()) and (len(word)>=min_len)]
    counts = Counter(words)
    return counts.most_common(top_k)

word_counts(' '.join(data['text']))


# # Training fasttext

# ## Necessary installations and imports 

# In[ ]:


# !pip install fasttext


# In[ ]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fasttext import train_supervised


# # Train test split 

# In[ ]:


X = data['text']
y = data['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=124)

X_train = np.array(X_train)
X_val = np.array(X_val)

y_train = np.array(y_train)
y_val = np.array(y_val)

print(X_train.shape)
print(X_val.shape)

print(y_train.shape)
print(y_val.shape)


# # Define auxiliary methods for training fasttext

# In[ ]:


def to_fasttext_format(data: list, labels: list, save_path: str=None):
    ft_data = []
    for d, l in zip(data, labels):
        ft_data.append("__label__{} {}".format(l, d))
    if save_path:
        np.savetxt(save_path, ft_data, fmt='%s')
    else:
        return ft_data
    
def train_fasttext(X_train, y_train, wordNgrams=1, minCount=1, ft_train_path="./tmp_train.txt", **kwargs):
    
    to_fasttext_format(X_train, y_train, save_path=ft_train_path)
    ft_model = train_supervised(ft_train_path, wordNgrams=wordNgrams, minCount=minCount, epoch=10, loss="softmax",  **kwargs)
    train_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_train))[0]]

    train_score = round(accuracy_score(np.array(train_preds).astype(np.integer), y_train), 3)
    print(f'train accuracy {train_score}')
    
    return ft_model, train_score


# # Training Fasttext with charngrams 3,3

# In[ ]:


ft_model, train_score = train_fasttext(X_train, y_train, minn=3, maxn=3)
val_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_val))[0]]
val_score = round(accuracy_score(y_val, np.array(val_preds).astype(np.integer)), 3)

print(f'val accuracy {val_score}')
ft_model = fasttext.load_model("model_filename.bin")


# # Define function which will take text as input and output the output the fastext embedding of the text as output

# In[ ]:


def get_vector_for_text(text):
    if len(text.split()) == 0:
        return np.zeros(100, dtype=np.float32)
    return np.average([ft_model.get_word_vector(x) for x in text.split()], axis=0)


# # Test the function

# In[ ]:


get_vector_for_text("aim environment alist supervision purpose e")


# # Display and change the device to gpu if available

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print('__CUDA Device Name:',torch.cuda.get_device_name(0))
print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)


# # Get the text embeddings for all the texts of images in separate column

# In[ ]:


data["fasttext_features"] = data["text"].apply(get_vector_for_text)


# # Create custom torch dataset 

# In[ ]:


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe["image_path"].values[idx], self.dataframe["fasttext_features"].values[idx], self.dataframe["label"].values[idx]
    
torch_data = CustomDataset(data)


# # Split torch dataset into train test split

# In[ ]:


train_set_size = int(len(torch_data) * 0.8)
valid_set_size = len(torch_data) - train_set_size
train_set, valid_set = torch.utils.data.random_split(torch_data, [train_set_size, valid_set_size])


# # Create the corresponding data loader

# In[ ]:


batch_size = 128
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)


# # Create our custom VGG + fasttext model where we concatenate the text embeddings of fasttext to one of the classifier layer outputs of VGG

# In[ ]:


num_classes = 13

class OurCustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        self.vgg16.classifier = nn.Linear(25088, num_classes)
        self.vgg16.load_state_dict(torch.load("/content/drive/MyDrive/VGGWeightsBest/vgg_weights_11.pth"))

        self.test_data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.9123, std=0.2116)
            ])
        
        self.fc1 = nn.Linear(25088, 100)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def apply_vgg(self, image_path):
        with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            image = self.test_data_transforms(image)
            image = image[None, :]
            image = image.to(device)
            features = self.vgg16.avgpool(self.vgg16.features(image))
            features = torch.squeeze(features)
            return features

    def forward(self, image_paths, fasttext_features):
        all_vgg_features = []
        for image_path in image_paths:
            vgg_features = self.apply_vgg(image_path)
            all_vgg_features.append(vgg_features)
        all_vgg_features_tensor = torch.stack(all_vgg_features)
        all_vgg_features_tensor = torch.flatten(all_vgg_features_tensor, start_dim=1)
        X = self.fc1(torch.relu(all_vgg_features_tensor))
        X = torch.cat((X, fasttext_features), axis=1)
        X = self.fc2(torch.relu(X))
        return self.fc3(torch.relu(X))


# # Initilize the optimizer, loss and learning rate scheduler

# In[ ]:


optimizer = torch.optim.Adam(MyModel.parameters(), lr=0.00005, weight_decay=0.022)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)


# # Auxiliary functions for training 

# In[ ]:


def evaluate(model, test_loader, device):
    with torch.no_grad():
        correct_preds, n_preds = 0, 0
        for batch_idx, (image_paths, fasttext_features, batch_labels) in enumerate(test_loader):
            fasttext_features = fasttext_features.to(device)

            prediction = MyModel(image_paths, fasttext_features)
            prediction = prediction.cpu()

            correct_preds += sum(torch.argmax(prediction, dim=1) == batch_labels)
            n_preds += len(batch_labels)
     
    return int(correct_preds) / n_preds


# In[ ]:


def compute_avg_loss(model, loader, device):
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for batch_idx, (image_paths, fasttext_features, batch_labels) in enumerate(loader):
            fasttext_features = fasttext_features.to(device)
            batch_labels = batch_labels.to(device)

            prediction = MyModel(image_paths, fasttext_features)
            loss = criterion(prediction, batch_labels)
            
            # change here
            total_loss += loss.item()
            total_count = total_count + 1
     
    return total_loss / total_count


# # Move our model to GPU if available

# In[ ]:


MyModel = OurCustomModel(num_classes)
MyModel.to(device)
MyModel


# In[ ]:


n_epochs = 22

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_losses_batch = []


for epoch in range(n_epochs):
    MyModel.train()
    for batch_idx, (image_paths, fasttext_features, batch_labels) in tqdm(enumerate(train_loader), f"Training epoch {epoch}", total=len(train_loader)):
        fasttext_features = fasttext_features.to(device)
        batch_labels = batch_labels.to(device)
        
        prediction = MyModel(image_paths, fasttext_features)
        
        loss = criterion(prediction, batch_labels)
        train_losses_batch.append(loss.item())
        loss.backward()
        optimizer.step()
        MyModel.zero_grad()

    # Free up the GPU so we can run the evals on it.
    del fasttext_features
    del batch_labels

    MyModel.eval()
    torch.save(MyModel.state_dict(), f"/content/drive/MyDrive/MyModelWeights/MyModel_{epoch}.pth")

    val_loss = compute_avg_loss(MyModel, val_loader, device)
    scheduler.step(val_loss)

    train_losses.append(sum(train_losses_batch) / len(train_losses_batch))
    val_losses.append(val_loss)

    train_accuracies.append( evaluate(MyModel, train_loader, device) )
    val_accuracies.append( evaluate(MyModel, val_loader, device) )
    


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

