from math import sqrt, pow
import pandas as pd
import numpy as np
from IPython.display import display

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#kodingan irvan
def showDataTraining():
    df = pd.read_csv("Upload/testing.csv", encoding="ISO-8859-1")
    df.head()

def countDataTraining():
    pass
    #TODO

def tfidf(tweet_bersih):
    bow_transformer = CountVectorizer().fit(tweet_bersih)
    print(bow_transformer.vocabulary_)

    tokens = bow_transformer.get_feature_names()
    print(tokens)

    text_bow = bow_transformer.transform(tweet_bersih)
    print(text_bow)

    X = text_bow.toarray()
    print(X)
    X.shape

    tfidf_transformer = TfidfTransformer().fit(text_bow)
    print(tfidf_transformer)

    title_tfidf = tfidf_transformer.transform(text_bow)
    print(title_tfidf)
    print(title_tfidf.shape)

    dd = pd.DataFrame(data=title_tfidf.toarray(), columns=tokens)
    display(dd)
    dd.to_csv('C:/Users/IRVAN/backendmknn/Upload/tfidftesting.csv', index= False)

    return X

def classification():
    pass
    # TODO


def jarakeuclideanDTDT(df):
    dict = {}
    df2 = []
    for i in range(df.shape[0]):
        dtdt = []
        for j in range(df.shape[0]):
            sum = 0
            if i != j:
                for k in df:
                    sum += pow(df[k][i]-df[k][j], 2)
                dtdt.append(sqrt(sum))
                # print(sqrt(sum))


                # dict = {i : dtdt}
                # df2['D'+str(i)] = pd.Series(dtdt)
            else:
                dtdt.append(0)
                # dict = {i: dtdt}
                # df2['D' + str(i)] = pd.Series(dtdt)
        df2.append(dtdt)
    x = pd.DataFrame(df2)
    x.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', index=False)
    # return

def testaccuracy():
    pass
    # TODO
