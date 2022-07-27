import sqlite3
from IPython.display import display
from nltk.sem.chat80 import sql_query
import pandas as pd
import preprocess


def db_connection():
    conn = None
    try:
        conn = sqlite3.connect("Database/mknn.sqlite")
    except sqlite3.error as e:
        print(e)
    return conn


def get_conn_cursor():
    conn = db_connection()
    cursor = conn.cursor()
    return conn, cursor


def db_import(query, path):
    conn, cursor = get_conn_cursor()
    data_insert = preprocess.get_data_preprocessing(path)
    cursor.executemany(query, data_insert)
    conn.commit()
    cursor.close()


# insert data training dari file csv ke tabel db
def db_import_data_training(path):
    sql_query = """INSERT INTO training(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, 
    polaritas) VALUES(?, ?, ?, ?, ?, ?, ?)"""
    db_import(sql_query, path)
    print("Success import data training from csv")


# insert data testing dari file csv ke tabel db
def db_import_data_testing(path):
    sql_query = """INSERT INTO testing(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, 
    polaritas_awal) VALUES(?, ?, ?, ?, ?, ?, ?)"""
    db_import(sql_query, path)
    print("Success import data testing from csv")


# tampil data hasil insert
def db_get_all_training():
    conn, cursor = get_conn_cursor()
    sql_query = "SELECT * FROM training"
    row = cursor.execute(sql_query).fetchall()
    for r in row:
        print(r)


def get_hasil():
    conn, cursor = get_conn_cursor()
    sql_query = "select tweet_hasil FROM training"
    row = cursor.execute(sql_query).fetchall()
    temp = []
    for x in row:
        for y in x:
            temp.append(y)
    tagged = temp
    sql_query = "select tweet_hasil FROM testing"
    row = cursor.execute(sql_query).fetchall()
    for x in row:
        for y in x:
            temp.append(y)
    return temp


def get_label(data):
    conn, cursor = get_conn_cursor()
    sql_query = "select polaritas FROM " + data
    row = cursor.execute(sql_query).fetchall()
    temp = []
    for x in row:
        for y in x:
            temp.append(y)
    dd = pd.DataFrame(temp)
    display(dd)
    dd.to_csv('C:/Users/IRVAN/backendmknn/Upload/traininglabel.csv', index=False)
    return temp


def updateData(nama_polaritas, nilai_polaritas, id):
    conn, cursor = get_conn_cursor()
    query = "Update testing set " + nama_polaritas + " = " + "'"+str(nilai_polaritas) + "'" + " where id_testing = " + str(id)
    cursor.execute(query)
    conn.commit()
    conn.close()
