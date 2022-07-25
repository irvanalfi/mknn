from math import sqrt, pow
import pandas as pd
import numpy as np
from IPython.display import display
import numpy as np
from numba import jit
from tqdm import tqdm
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

@jit(target_backend = 'cuda')
def jarakeuclideanDTDT(df):
    # df = df.iloc[1:, :]
    dict = {}
    df2 = []
    for i in tqdm(range(df.shape[0])):
        dtdt = []
        for j in tqdm(range(df.shape[0]), leave=False):
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
    return x

@jit(target_backend = 'cuda')
def jarakeuclideanDTDS(dtdf,dsdf,split):
    # df = df.iloc[1:, :]
    dict = {}
    df2 = []
    dsdf.reset_index()
    display(dsdf)
    # print(pd.Index.min(dsdf))
    for i in tqdm(range(int(dtdf.shape[0]))):
        dtdt = []
        for j in tqdm(range(int(dsdf[dsdf.columns[0]].count())), leave=False):
            sum = 0
            for k in dtdf:
                # print(i,j,k,l)
                sum += pow(dtdf[str(k)][i]-dsdf[str(k)][split+j], 2)
                # print(sum)
            dtdt.append(sqrt(sum))
            
            # print(sqrt(sum))


            # dict = {i : dtdt}
            # df2['D'+str(i)] = pd.Series(dtdt)
        df2.append(dtdt)
    x = pd.DataFrame(df2)
    x.to_csv('D:/github/mknn/Upload/euclideandtds.csv', index=False, float_format="%.2f")
    return x

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
    # df2.to_csv('D:/github/mknn/Upload/smalleuclideandtdt.csv', index=False)
    return df2

def k11_euclidean(df):
    a = df.iloc[0]
    array=[0]*len(a) # Membuat array kosong untuk menyimpan data per baris
    print(type(array))
    for x in range(len(a)):
        array[x]=df.iloc[[x]] # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist() # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0] # Menghilangkan Bracket
        array[x] = array[x][1:6]
    df2 = pd.DataFrame(array)
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    # df2.to_csv('D:/github/mknn/Upload/k11smalleuclideandtdt.csv', index=False)
    return df2

def lokasi(df):
    a = df.iloc[0]
    array=[0]*len(a) # Membuat array kosong untuk menyimpan data per baris
    
    for x in range(len(a)):
        # print(x)
        array[x]= df.iloc[[x]] # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist() # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0] # Menghilangkan Bracket
        np_array = np.array(array[x]).argsort()[1:6]
        # np_array = np_array+1
        array[x] = np_array # Proses sorting data
    df2 = pd.DataFrame(array)
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    # df2.to_csv('D:/github/mknn/Upload/argsmalleuclideandtdt.csv', index=False)
    return df2

def pelabelan(df1, df2):
    pd.options.mode.chained_assignment = None 
    # print(df1)
    for x, y in df1.iterrows():
        # print(x)
        for a in df1:
            df1[a][x] = df2['2'][y[a]]
            # print(df2['0'][y[a]])
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    # df1.to_csv('D:/github/mknn/Upload/labelargsmalleuclideandtdt.csv', index=False)
    return df1

def validitas(df1, df2):
    a = df1[df1.columns[0]].count()
    n = len(df1.iloc[0])
    array=[0]*a # Membuat array kosong untuk menyimpan data per baris
    # print(type(array))
    for x, y in df1.iterrows():
        # print(x, y)
        sum = 0
        for z in df1:
            
            if y[z] == df2['2'][x]:
                sum += 1
            result = 1/(n)*(sum)
            # print(n)
            array[x] = result
        # array[x] = df1.iloc[[x]] # Menyimpan setiap kolumn dalam index array
        # array[x] = array[x].values.tolist() # Merubah values menjadi dalam bentuk list
        # array[x] = array[x][0] # Menghilangkan Brac # Proses sorting data
    df2 = pd.DataFrame(array)
    # df2.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv')
    # df2.to_csv('D:/github/mknn/Upload/valargsmalleuclideandtdt.csv', index=False)
    return df2

def ranking(df1, df2):
    # array = df2.values.tolist()
    # x = df1[df1.columns[0]].count()
    # display(df2)
    for x, y in df1.iterrows():
        # print(
        #     df1[0][x], 
        #     y[0], 
        #     df2[x]
        #     )
        df1[0][x] = y[0]*(1/(df2[x]+0.5))
    a = df1[0].values.tolist()
    label = np.array(a)
    a.sort()
    label = np.argsort(label)
    # print(label)
    df2 = pd.DataFrame(a[:5])
    return df2, label[:5]


def testaccuracy():
    pass
    # TODO
