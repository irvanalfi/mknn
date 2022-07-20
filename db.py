import sqlite3

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
