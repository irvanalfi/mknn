from math import sqrt, pow
import pandas as pd
import numpy as np
from IPython.display import display
import numpy as np

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
    dd.to_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', index= False)

    return X

def classification():
    pass
    # TODO


def jarakeuclideanDTDT(df):
    # df = df.iloc[1:, :]
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
    # x.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', index=False, float_format="%.2f")
    x.to_csv('D:/github/mknn/Upload/euclideandtdt.csv', index=False, float_format="%.2f")
    # return


def jarakeuclideanDTDS(dtdf,dsdf):
    # df = df.iloc[1:, :]
    dict = {}
    df2 = []
    for i in range(dtdf.shape[0]):
        dtdt = []
        for j in range(dsdf.shape[0]):
            sum = 0
            for k in dtdf:
                sum += pow(df[k][i]-df[k][j], 2)
            dtdt.append(sqrt(sum))
            # print(sqrt(sum))


            # dict = {i : dtdt}
            # df2['D'+str(i)] = pd.Series(dtdt)
        df2.append(dtdt)
    x = pd.DataFrame(df2)
    x.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', index=False, float_format="%.2f")
    # return

def small(df):
    a = df.iloc[0]
    array=[0]*len(a) # Membuat array kosong untuk menyimpan data per baris
    print(type(array))
    for x in range(len(a)):
        array[x]=df.iloc[[x]] # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist() # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0] # Menghilangkan Bracket
        array[x].sort() # Proses sorting data
    df2 = pd.DataFrame(array)
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    df2.to_csv('D:/github/mknn/Upload/smalleuclideandtdt.csv', index=False)

def k11_euclidean(df):
    a = df.iloc[0]
    array=[0]*len(a) # Membuat array kosong untuk menyimpan data per baris
    print(type(array))
    for x in range(len(a)):
        array[x]=df.iloc[[x]] # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist() # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0] # Menghilangkan Bracket
        array[x] = array[x][1:12]
    df2 = pd.DataFrame(array)
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    df2.to_csv('D:/github/mknn/Upload/k11smalleuclideandtdt.csv', index=False)

def lokasi(df):
    a = df.iloc[0]
    array=[0]*len(a) # Membuat array kosong untuk menyimpan data per baris
    print(type(array))
    for x in range(len(a)):
        array[x]= df.iloc[[x]] # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist() # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0] # Menghilangkan Bracket
        np_array = np.array(array[x]).argsort()[1:12]

        array[x] = np_array # Proses sorting data
    df2 = pd.DataFrame(array)
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    df2.to_csv('D:/github/mknn/Upload/argsmalleuclideandtdt.csv', index=False)

def pelabelan(df1, df2):
    pd.options.mode.chained_assignment = None 
    # print(df1)
    for x, y in df1.iterrows():
        # print(x)
        for a in df1:
            df1[a][x] = df2['0'][y[a]]
            # print(df2['0'][y[a]])
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    df1.to_csv('D:/github/mknn/Upload/labelargsmalleuclideandtdt.csv', index=False)

def validitas(df1, df2):
    a = df1.iloc[0]
    array=[0]*len(a) # Membuat array kosong untuk menyimpan data per baris
    print(type(array))
    for x in range(len(a)):
        sum = 0
        if a[x] == df2['0'][x]:
            sum += 1
        result = 1/(x+1)*(sum)
        array[x] = result
        # array[x] = df1.iloc[[x]] # Menyimpan setiap kolumn dalam index array
        # array[x] = array[x].values.tolist() # Merubah values menjadi dalam bentuk list
        # array[x] = array[x][0] # Menghilangkan Brac # Proses sorting data
    df2 = pd.DataFrame(array)
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    df2.to_csv('D:/github/mknn/Upload/valargsmalleuclideandtdt.csv', index=False)

def testaccuracy():
    pass
    # TODO
