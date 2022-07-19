import csv
import sqlite3

import preprocess


def db_connection():
    conn = None
    try:
        conn = sqlite3.connect("Database/mknn.sqlite")
    except sqlite3.error as e:
        print(e)
    return conn

# insert data training dari file csv ke tabel db


def db_import_data_training(path):
    conn = db_connection()
    cursor = conn.cursor()
    sql_query = """INSERT INTO training(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, 
    polaritas) VALUES(?, ?, ?, ?, ?, ?, ?)"""
    data_insert = []

    file = open(path, encoding="utf-8")
    contents = csv.reader(file, delimiter=';')

    for row in contents:
        c_text, tokenize_text, stopwords, lemmawords = preprocess.preprocessing(row[1])
        data_insert.append([row[0], row[1], c_text, "'" + "','".join(map(str, tokenize_text)),
                            "'" + "','".join(map(str, stopwords)), lemmawords, row[2]])

    cursor.executemany(sql_query, data_insert)
    conn.commit()
    print("Success import data training from csv")
    cursor.close()

# insert data testing dari file csv ke tabel db


def db_import_data_testing(path):
    conn = db_connection()
    cursor = conn.cursor()
    sql_query = """INSERT INTO testing(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, 
    polaritas) VALUES(?, ?, ?, ?, ?, ?, ?)"""
    file = open(path, encoding="utf-8")
    contents = csv.reader(file, delimiter=';')
    data_insert = []

    for row in contents:
        c_text, tokenize_text, stopwords, lemmawords = preprocess.preprocessing(row[1])
        data_insert.append([row[0], row[1], c_text, "'" + "','".join(map(str, tokenize_text)),
                            "'" + "','".join(map(str, stopwords)), lemmawords, row[2]])

    cursor.executemany(sql_query, contents)
    conn.commit()
    print("Success import data testing from csv")
    cursor.close()

# tampil data hasil insert


def db_get_all_training():
    conn = db_connection()
    cursor = conn.cursor()

    sql_query = "SELECT * FROM training"

    row = cursor.execute(sql_query).fetchall()

    for r in row:
        print(r)
