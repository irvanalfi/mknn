import re
import csv
import contractions
import emoji
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

import db
from db import *


def clean_text(text):
    # bersih2 username (@blabla), hapus selain alfabet dan hapus url
    tweet_bersih = ''.join(re.sub("(@[\w]+)|(\w+:\/\/\S+)", " ", text))
    # ubah emoji menjadi text
    tweet_bersih = emoji.demojize(tweet_bersih)
    # ubah perulangan huruf yang banyak menjadi dua huruf
    tweet_bersih = re.sub(r'(.)\1+', r'\1\1', tweet_bersih)
    # hapus selain abjad
    tweet_bersih = re.sub(r'[^a-zA-Z]', ' ', tweet_bersih)
    # hapus hastag
    tweet_bersih = re.sub('#+', ' ', tweet_bersih)
    # ubah singkatan menjadi kepanjangan
    tweet_bersih = contractions.fix(tweet_bersih)
    # lowercase
    tweet_bersih = tweet_bersih.lower()
    return tweet_bersih


def tokenizing(clean_text: str):
    return tokenize.word_tokenize(clean_text)


def stop_word_removal(token_text):
    manual_stop = ("windows", "pc", "microsoft", "android", "computer", "device", "google", "chrome", "edge", "browser")
    stops = set(stopwords.words("english"))
    stops.update(manual_stop)
    stopword = [word for word in token_text if word not in stops]
    return stopword


# proses pencarian kata baku
def lemmitization(text: list):
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word) for word in text]
    lemma_words = ' '.join(lemma)
    return lemma_words


# proses preprocessing
def preprocessing(tweet):
    c_text = clean_text(tweet)
    tokenize_text = tokenizing(c_text)
    stopwords = stop_word_removal(tokenize_text)
    lemmawords = lemmitization(stopwords)
    return c_text, tokenize_text, stopwords, lemmawords


def get_data_preprocessing(path):
    file = open(path, encoding="utf-8")
    contents = csv.reader(file, delimiter=';')
    data = []
    print(contents)
    for row in contents:
        c_text, tokenize_text, stopwords, lemmawords = preprocessing(row[1])
        data.append([row[0], row[1], c_text, "'" + "','".join(map(str, tokenize_text)),
                     "'" + "','".join(map(str, stopwords)), lemmawords, row[2]])
    return data


def pelabelanOtomatis():
    data = [db.get_hasil_training(), db.get_hasil_testing()]

    for index, i in enumerate(data):
        label = ""
        for index_2, tweet in enumerate(i):
            dictTweet = {}
            # bersih2 regex dan url
            analysis = TextBlob(tweet)
            if analysis.sentiment.polarity > 0.0:
                label="positif"
            elif analysis.sentiment.polarity == 0.0:
                label="netral"
            else:
                label="negatif"
            if(index == 0):
                db.updatePolaritasAwalTraining(label, index_2)
            else:
                db.updatePolaritasAwalTesting(label, index_2)