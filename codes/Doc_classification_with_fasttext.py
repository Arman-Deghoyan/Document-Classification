#!/usr/bin/env python
# coding: utf-8

# # Trying NLP lightweight models to classify documents

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
import pytesseract
import numpy as np


# # Install the library with the help of which we will get text from the image

# In[ ]:


# Note its a tricky library and in each OS needs to installed uniquely 
# ! pip install pytesseract


# # Extract texts from all the documents and save all txt files in the appropriate folder. Note: Grab a coffee, go breath some fresh air as it might take long 

# In[ ]:


# import os
# data_dir = '/home/ml_user/Downloads/final_project/train/'

# destination_dir = '/home/ml_user/Downloads/final_project/train_texts/'
# if not os.path.exists(destination_dir):
#     os.mkdir(destination_dir)

# for folder in os.listdir(data_dir):

#     current_directory = os.path.join(destination_dir, folder)

#     if not os.path.exists(current_directory):
#         os.mkdir(current_directory)
#     for image_name in tqdm(os.listdir(os.path.join(data_dir, folder))):

#         image = Image.open(os.path.join(data_dir + folder, image_name)).convert("RGB")

#         ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
#         ocr_df = ocr_df.dropna().reset_index(drop=True)
#         float_cols = ocr_df.select_dtypes('float').columns
#         ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
#         ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
#         try:
#             words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])
#         except:
#             print('Could not find text for this image')
#             print(image_name, folder)
#             words = ''

#         with open(os.path.join(destination_dir + folder, image_name[:-3] + 'txt'), "w") as dest_file:
#             dest_file.write(words)


# # Extract the text from images and get the dataframe with text and label columns

# In[ ]:


import os

data_dir = '/home/ml_user/Downloads/final_project/train/'
destination_dir = '/home/ml_user/Downloads/final_project/train_texts/'

if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)

data = pd.DataFrame(columns=['text', 'label'])

for folder in os.listdir(data_dir):
    count = 0 
    current_directory = os.path.join(destination_dir, folder)

    if not os.path.exists(current_directory):
        os.mkdir(current_directory)
    for image_name in tqdm(os.listdir(os.path.join(data_dir, folder))):
        if count < 10:
            image = Image.open(os.path.join(data_dir + folder, image_name)).convert("RGB")

            ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
            ocr_df = ocr_df.dropna().reset_index(drop=True)
            float_cols = ocr_df.select_dtypes('float').columns
            ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
            ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
            words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])

            data = data.append({'text': words, 'label': folder}, ignore_index=True)


# # Read the texts from the saved folder of txt files and create the dataframe with text and label

# In[ ]:


# data_dir = '/home/ml_user/Downloads/final_project/train_texts/'

# text_list = []
# labels_list = []
# data = pd.DataFrame(columns=['text', 'label'])

# for folder in os.listdir(data_dir):
#     current_directory = os.path.join(data_dir, folder)
#     for text_of_image in tqdm(os.listdir(current_directory)):
        
#             text_i = open(os.path.join(data_dir + folder, text_of_image),encoding = 'latin-1').read()
#             text_i = text_i.replace('\n',' ')
#             text_i = text_i.replace(',',' ')
            
#             if len(text_i) != 0: 
#                 data =data.append({'text': text_i, 'label': folder}, ignore_index=True)
#             else:
#                 data =data.append({'text': text_i, 'label': folder}, ignore_index=True)

# print(f'The shape of the data is {data.shape}')

# print(data['label'].value_counts())

# data.head()


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


# # Training part

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


# # Training Naive Bayes and Multinomial Naive Bayes 

# In[ ]:


train_scores, val_scores, models, model_names = [], [], [], []

# ### count vectorizer + naive bayes

nb = Pipeline([('countVec', CountVectorizer(lowercase=False, token_pattern='\w+', min_df=3)),
               ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_val)
train_score = round(accuracy_score(nb.predict(X_train), y_train), 3)
val_score = round(accuracy_score(y_pred, y_val), 3)

print(f'train accuracy {train_score}')
print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(nb)
model_names.append('cv_ng1_nb')


# # Count vectorizer with bigrams + naive bayes

# In[ ]:


nb = Pipeline([('countVec', CountVectorizer(lowercase=False, token_pattern='\w+', ngram_range=(1, 2), min_df=3)),
               ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_val)
train_score = round(accuracy_score(nb.predict(X_train), y_train), 3)
val_score = round(accuracy_score(y_pred, y_val), 3)

print(f'train accuracy {train_score}')
print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(nb)
model_names.append('cv_ng2_nb')


# # Count vectorizer with trigrams + naive bayes

# In[ ]:


nb = Pipeline([('countVec', CountVectorizer(lowercase=False, token_pattern='\w+', ngram_range=(1, 3), min_df=3)),
               ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_val)
train_score = round(accuracy_score(nb.predict(X_train), y_train), 3)
val_score = round(accuracy_score(y_pred, y_val), 3)

print(f'train accuracy {train_score}')
print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(nb)
model_names.append('cv_ng3_nb')


# # Tf-idf vectorizer with bigrams + naive bayes

# In[ ]:


nb = Pipeline([('tfidf', TfidfVectorizer(lowercase=False, token_pattern='\w+', ngram_range=(1, 2), 
                                         min_df=3)),
               ('clf', MultinomialNB()),])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_val)
train_score = round(accuracy_score(nb.predict(X_train), y_train), 3)
val_score = round(accuracy_score(y_pred, y_val), 3)

print(f'train accuracy {train_score}')
print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(nb)
model_names.append('tf_ng2_nb')


# # Logistic Regression

# In[ ]:


logreg = Pipeline([('countVec', CountVectorizer(lowercase=False, token_pattern='\w+', max_features=100)),
                   ('clf', LogisticRegression(random_state=42, solver='liblinear')),])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)
train_score = round(accuracy_score(y_train, logreg.predict(X_train)), 3)
val_score = round(accuracy_score(y_val, y_pred), 3)

print(f'train accuracy {train_score}')
print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(logreg) 
model_names.append('cv_ng1_lr')


# # Count vectorizer with bigrams + logistic regression

# In[ ]:


logreg = Pipeline([('countVec', CountVectorizer(lowercase=False, token_pattern='\w+', ngram_range=(1, 2), 
                                                min_df=3)),
                   ('clf', LogisticRegression(random_state=42, solver='liblinear')),])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)
train_score = round(accuracy_score(y_train, logreg.predict(X_train)), 3)
val_score = round(accuracy_score(y_val, y_pred), 3)

print(f'train accuracy {train_score}')
print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(logreg)
model_names.append('cv_ng2_lr')


# # Count vectorizer with trigrams + logistic regression

# In[ ]:


logreg = Pipeline([('countVec', CountVectorizer(lowercase=False, token_pattern='\w+', ngram_range=(1, 3), 
                                                min_df=3)),
                   ('clf', LogisticRegression(random_state=42, solver='liblinear')),])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)
train_score = round(accuracy_score(y_train, logreg.predict(X_train)), 3)
val_score = round(accuracy_score(y_val, y_pred), 3)

print(f'train accuracy {train_score}')
print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(logreg)
model_names.append('cv_ng3_lr')


# # Fasttext

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


# # Training fasttext

# In[ ]:


ft_model, train_score = train_fasttext(X_train, y_train)
val_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_val))[0]]

val_score = round(accuracy_score(y_val, np.array(val_preds).astype(np.integer)), 3)

print(f'val accuracy {val_score}')


train_scores.append(train_score)
val_scores.append(val_score)
models.append(ft_model)
model_names.append('ft_ng1')


# # Fasttext with trigrams

# In[ ]:


ft_model, train_score = train_fasttext(X_train, y_train, wordNgrams=3)
val_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_val))[0]]
val_score = round(accuracy_score(y_val, np.array(val_preds).astype(np.integer)), 3)

print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(ft_model)
model_names.append('ft_ng3')


# # Fasttext with charngrams 3,3

# In[ ]:


ft_model, train_score = train_fasttext(X_train, y_train, minn=3, maxn=3)
val_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_val))[0]]
val_score = round(accuracy_score(y_val, np.array(val_preds).astype(np.integer)), 3)

print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(ft_model)
model_names.append('ft_charngrams_33')


# # Fasttext with charngrams 3,4

# In[ ]:


ft_model, train_score = train_fasttext(X_train, y_train, minn=3, maxn=4)
val_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_val))[0]]
val_score = round(accuracy_score(y_val, np.array(val_preds).astype(np.integer)), 3)

print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(ft_model)
model_names.append('ft_charngrams_34')


# # Fasttext with charngrams 4,5

# In[ ]:


ft_model, train_score = train_fasttext(X_train, y_train, minn=4, maxn=5)
val_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_val))[0]]
val_score = round(accuracy_score(y_val, np.array(val_preds).astype(np.integer)), 3)

print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(ft_model)
model_names.append('ft_charngrams_45')


# # Fasttext with charngrams 5,7

# In[ ]:


ft_model, train_score = train_fasttext(X_train, y_train, minn=5, maxn=7)
val_preds = [i[0].split('_')[-1] for i in ft_model.predict(list(X_val))[0]]
val_score = round(accuracy_score(y_val, np.array(val_preds).astype(np.integer)), 3)

print(f'val accuracy {val_score}')

train_scores.append(train_score)
val_scores.append(val_score)
models.append(ft_model)
model_names.append('ft_charngrams_57')


# # Plot all training and validation scores 

# In[ ]:


plt.plot(train_scores)
plt.plot(val_scores)

