from time import time
from math import sqrt, pow
import pandas as pd
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from db import get_polaritas
# from multigpu import device_controller, main
# from numba import jit
# from threading import Thread

def tfidf(tweet_bersih):
    bow_transformer = CountVectorizer().fit(tweet_bersih)
    tokens = bow_transformer.get_feature_names()
    text_bow = bow_transformer.transform(tweet_bersih)
    X = text_bow.toarray()
    X.shape
    tfidf_transformer = TfidfTransformer().fit(text_bow)
    title_tfidf = tfidf_transformer.transform(text_bow)
    dd = pd.DataFrame(data=title_tfidf.toarray(), columns=tokens)
    dd.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/tfidf.json')
    dd.to_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', index=False)
    return X


# @jit(target_backend=device_controller(0))
def jarakeuclideanDTDT(df):
    df2 = []
    for i in tqdm(range(df.shape[0])):
        dtdt = []
        for j in tqdm(range(df.shape[0]), leave=False):
            sum = 0
            if i != j:
                for k in df:
                    sum += pow(df[k][i] - df[k][j], 2)
                dtdt.append(sqrt(sum))
            else:
                dtdt.append(0)
        df2.append(dtdt)
    x = pd.DataFrame(df2)
    x.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/euclideandtdt.json')
    x.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', index=False, float_format="%.2f")
    return x

# @jit(target_backend=device_controller(0))
def jarakeuclideanDTDS(dtdf,dsdf,split):
    df2 = []
    dsdf.reset_index()
    for i in tqdm(range(int(dtdf.shape[0]))):
        dtdt = []
        for j in tqdm(range(int(dsdf[dsdf.columns[0]].count())), leave=False):
            sum = 0
            for k in dtdf:
                sum += pow(dtdf[str(k)][i] - dsdf[str(k)][split + j], 2)
            dtdt.append(sqrt(sum))
        df2.append(dtdt)
    x = pd.DataFrame(df2)
    x.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/euclideandtds.json')
    x.to_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', index=False, float_format="%.2f")
    return x


def small(df):
    a = df.iloc[0]
    array = [0] * len(a)  # Membuat array kosong untuk menyimpan data per baris
    for x in range(len(a)):
        array[x] = df.iloc[[x]]  # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist()  # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0]  # Menghilangkan Bracket
        array[x].sort()  # Proses sorting data
    df2 = pd.DataFrame(array)
    return df2


def k_euclidean(df, k):
    a = df.iloc[0]
    array = [0] * len(a)  # Membuat array kosong untuk menyimpan data per baris
    for x in range(len(a)):
        array[x] = df.iloc[[x]]  # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist()  # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0]  # Menghilangkan Bracket
        array[x] = array[x][1:k + 1]
    df2 = pd.DataFrame(array)
    df2.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/k11euclideandtdt.json')
    return df2


def lokasi(df, k):
    a = df.iloc[0]
    array = [0] * len(a)  # Membuat array kosong untuk menyimpan data per baris
    for x in range(len(a)):
        array[x] = df.iloc[[x]]  # Menyimpan setiap kolumn dalam index array
        array[x] = array[x].values.tolist()  # Merubah values menjadi dalam bentuk list
        array[x] = array[x][0]  # Menghilangkan Bracket
        np_array = np.array(array[x]).argsort()[1:k + 1]
        array[x] = np_array  # Proses sorting data
    df2 = pd.DataFrame(array)
    df2.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/lokasiargsmalleuclideandtdt.json')
    return df2


def pelabelan(df1, df2):
    pd.options.mode.chained_assignment = None
    for x, y in df1.iterrows():
        for a in df1:
            df1[a][x] = df2[y[a]]
    df1.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/labelargsmalleuclideandtdt.json')
    return df1


def validitas(df1, df2):
    a = df1[df1.columns[0]].count()
    n = len(df1.iloc[0])
    array = [0] * a  # Membuat array kosong untuk menyimpan data per baris
    for x, y in df1.iterrows():
        sum = 0
        for z in df1:
            if y[z] == df2[x]:
                sum += 1
            result = 1 / (n) * (sum)
            array[x] = result
    df2 = pd.DataFrame(array)
    df2.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/valargsmalleuclideandtdt.json')
    return df2


def ranking(df1, df2, k):
    for x, y in df1.iterrows():
        df1[0][x] = y[0] * (1 / (df2[x] + 0.5))
    a = df1[0].values.tolist()
    label = np.array(a)
    a.sort()
    label = np.argsort(label)
    dataframedummy = pd.DataFrame(a[:k])
    dataframedummy = dataframedummy.rename(columns={0:1})
    df2 = dataframedummy
    df3 = pd.DataFrame(label[:k])
    weightvoting = pd.concat([df3, df2], axis=1)
    weightvoting.to_csv('C:/Users/IRVAN/backendmknn/Upload/weightvoting.csv', index=False)
    weightvoting.to_json('C:/xampp/htdocs/mknnfrontend/assets/json/weightvoting.json')
    return df2, label[:k]


def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num


# Menghitung tingkat akurasi mknn dengan confusion matrix
def testaccuracy(polaritas_awal, polaritas_k):
    polaritas_awal = get_polaritas(polaritas_awal)
    polaritas_k = get_polaritas(polaritas_k)
    t = time()
    y_pred = polaritas_k
    y_test = polaritas_awal
    data_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(data_report).transpose()
    a = df.to_json(orient='index')
    return a
