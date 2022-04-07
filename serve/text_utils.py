from bs4 import BeautifulSoup
from html import unescape
import os
import spacy

import nltk

nltk.download('words')
words = set(nltk.corpus.words.words())


try:
    spacy_en = spacy.load("en_core_web_sm")
except:
    os.system('python -m spacy download en_core_web_sm')
    spacy_en = spacy.load("en_core_web_sm")


stops_spacy = sorted(spacy.lang.en.stop_words.STOP_WORDS)
stops_spacy.extend(["is", "to"])


def remove_stopwords_spacy(text, stopwords=stops_spacy):
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


def lemmatize_spacy(text):
    text = spacy_en(text)
    lemmas = [token.lemma_ for token in text]
    return " ".join(lemmas)


def remove_non_eng_words(text):
    return " ".join(w for w in nltk.wordpunct_tokenize(text) \
                    if w.lower() in words or not w.isalpha())


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
