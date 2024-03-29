import sqlite3
import preprocess
from preprocess import *


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


def db_import_data_training(path):
    sql_query = """INSERT INTO training(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, 
    polaritas) VALUES(?, ?, ?, ?, ?, ?, ?)"""
    db_import(sql_query, path)
    print("Success import data training from csv")


def db_delete_all(tabel):
    try:
        conn, cursor = get_conn_cursor()
        query = "DELETE FROM " + tabel
        cursor.execute(query)
        conn.commit()
        qwery = "DELETE FROM sqlite_sequence where name='"+tabel+"'"
        cursor.execute(qwery)
        conn.commit()
        message = "success"
    except:
        conn.rollback()
        message = "failed"
    finally:
        conn.close()
    return message


def db_import_data_testing(path):
    sql_query = """INSERT INTO testing(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, 
    polaritas_awal) VALUES(?, ?, ?, ?, ?, ?, ?)"""
    db_import(sql_query, path)
    print("Success import data testing from csv")


def get_all_crawling():
    crawls = []
    try:
        conn, cursor = get_conn_cursor()
        sql_query = "SELECT * FROM crawling"
        rows = cursor.execute(sql_query).fetchall()
        for i in rows:
            crawl = {}
            crawl["id_crawling"] = i[0]
            crawl["username"] = i[1]
            crawl["tweet"] = i[2]
            crawl["polaritas"] = i[3]
            crawls.append(crawl)
    except:
        crawls = []
    return crawls


def add_crawling(username, tweet, polaritas):
    conn, cursor = get_conn_cursor()
    query = "INSERT INTO crawling(username, tweet, polaritas) VALUES('" + username + "', '" + tweet + "', '" + polaritas + "')"
    cursor.execute(query)
    conn.commit()


def get_crawling_by_id(id):
    conn, cursor = get_conn_cursor()
    query = "SELECT * FROM crawling WHERE crawling.id_crawling = '" + str(id) + "'"
    row = cursor.execute(query).fetchone()
    if row is not None:
        data = {
            'id_crawling': row[0],
            'username': row[1],
            'tweet': row[2],
            'polaritas': row[3]
        }
    else:
        data = {}
    return data


def update_crawling(id, polaritas):
    conn, cursor = get_conn_cursor()
    query = "UPDATE crawling SET polaritas = '" + polaritas + "' WHERE id_crawling =" + str(id) + ""
    cursor.execute(query)
    conn.commit()
    data = get_crawling_by_id(id)
    return data


def db_dell_crawling(id):
    try:
        conn, cursor = get_conn_cursor()
        query = "DELETE FROM crawling WHERE id_crawling = " + str(id) + ""
        cursor.execute(query)
        conn.commit()
        message = "success"
    except:
        conn.rollback()
        message = "failed"
    finally:
        conn.close()
    return message


def add_training(crawling, id_data):
    conn, cursor = get_conn_cursor()
    preprocess = get_data_preprocessing_by_one(crawling)
    cursor.execute( "INSERT INTO training(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, polaritas) " \
            "VALUES(?, ?, ?, ?, ?, ?, ?)",(preprocess['username'], preprocess['tweet_asli'], preprocess[
        'clean_text'], preprocess['tokenize'], preprocess['stopword_r'], preprocess['lemmawords'], preprocess[
        'polaritas']))
    conn.commit()
    query = "DELETE FROM crawling WHERE id_crawling = " + str(id_data) + ""
    cursor.execute(query)
    conn.commit()
    status = "success"
    return status


def add_testing(crawling, id_data):
    conn, cursor = get_conn_cursor()
    preprocess = get_data_preprocessing_by_one(crawling)
    cursor.execute( "INSERT INTO testing(username, tweet_asli, clean_text, tokenize, stopword_r, tweet_hasil, polaritas_awal) " \
            "VALUES(?, ?, ?, ?, ?, ?, ?)",(preprocess['username'], preprocess['tweet_asli'], preprocess[
        'clean_text'], preprocess['tokenize'], preprocess['stopword_r'], preprocess['lemmawords'], preprocess[
        'polaritas']))
    conn.commit()
    query = "DELETE FROM crawling WHERE id_crawling = " + str(id_data) + ""
    cursor.execute(query)
    conn.commit()
    status = "success"
    return status


def db_get_all_training():
    trains = []
    try:
        conn, cursor = get_conn_cursor()
        sql_query = "SELECT * FROM training"
        rows = cursor.execute(sql_query).fetchall()
        for i in rows:
            train = {}
            train["id_training"] = i[0]
            train["username"] = i[1]
            train["tweet_asli"] = i[2]
            train["clean_text"] = i[3]
            train["tokenize"] = i[4]
            train["stopword_r"] = i[5]
            train["tweet_hasil"] = i[6]
            train["polaritas"] = i[7]
            trains.append(train)
    except:
        trains = []
    return trains


def db_get_limapuluh_training():
    trains = []
    try:
        conn, cursor = get_conn_cursor()
        sql_query = "SELECT * FROM training LIMIT 50"
        rows = cursor.execute(sql_query).fetchall()
        for i in rows:
            train = {}
            train["id_training"] = i[0]
            train["username"] = i[1]
            train["tweet_asli"] = i[2]
            train["clean_text"] = i[3]
            train["tokenize"] = i[4]
            train["stopword_r"] = i[5]
            train["tweet_hasil"] = i[6]
            train["polaritas"] = i[7]
            trains.append(train)
    except:
        trains = []
    return trains


def db_get_all_testing():
    tests = []
    try:
        conn, cursor = get_conn_cursor()
        sql_query = "SELECT * FROM testing"
        rows = cursor.execute(sql_query).fetchall()
        for i in rows:
            test = {}
            test["id_testing"] = i[0]
            test["username"] = i[1]
            test["tweet_asli"] = i[2]
            test["clean_text"] = i[3]
            test["tokenize"] = i[4]
            test["stopword_r"] = i[5]
            test["tweet_hasil"] = i[6]
            test["polaritas_awal"] = i[7]
            test["polaritas_akhir_k3"] = i[8]
            test["polaritas_akhir_k5"] = i[9]
            test["polaritas_akhir_k7"] = i[10]
            test["polaritas_akhir_k9"] = i[11]
            test["polaritas_akhir_k11"] = i[12]
            test["polaritas_akhir_k13"] = i[13]
            test["polaritas_akhir_k15"] = i[14]
            test["polaritas_akhir_k17"] = i[15]
            test["polaritas_akhir_k19"] = i[16]
            test["polaritas_akhir_k21"] = i[17]
            tests.append(test)
    except:
        tests = []
    return tests


def db_get_limapuluh_testing():
    tests = []
    try:
        conn, cursor = get_conn_cursor()
        sql_query = "SELECT * FROM testing LIMIT 50"
        rows = cursor.execute(sql_query).fetchall()
        for i in rows:
            test = {}
            test["id_testing"] = i[0]
            test["username"] = i[1]
            test["tweet_asli"] = i[2]
            test["clean_text"] = i[3]
            test["tokenize"] = i[4]
            test["stopword_r"] = i[5]
            test["tweet_hasil"] = i[6]
            test["polaritas_awal"] = i[7]
            test["polaritas_akhir_k3"] = i[8]
            test["polaritas_akhir_k5"] = i[9]
            test["polaritas_akhir_k7"] = i[10]
            test["polaritas_akhir_k9"] = i[11]
            test["polaritas_akhir_k11"] = i[12]
            test["polaritas_akhir_k13"] = i[13]
            test["polaritas_akhir_k15"] = i[14]
            test["polaritas_akhir_k17"] = i[15]
            test["polaritas_akhir_k19"] = i[16]
            test["polaritas_akhir_k21"] = i[17]
            tests.append(test)
    except:
        tests = []
    return tests


def db_get_count_polaritas(nilai_k, nama_polaritas):
    conn, cursor = get_conn_cursor()
    query = "SELECT count(polaritas_akhir_" + nilai_k + ") " \
                                                        "FROM testing " \
                                                        "WHERE polaritas_akhir_" + nilai_k + " = " + "'" \
            + nama_polaritas + "'"

    row = cursor.execute(query).fetchone()[0]
    return row


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


def get_hasil_training():
    conn, cursor = get_conn_cursor()
    sql_query = "select tweet_hasil FROM training"
    row = cursor.execute(sql_query).fetchall()
    temp = []
    for x in row:
        for y in x:
            temp.append(y)
    return temp


def get_hasil_testing():
    conn, cursor = get_conn_cursor()
    sql_query = "select tweet_hasil FROM testing"
    row = cursor.execute(sql_query).fetchall()
    temp = []
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
    return temp


def updateData(nama_polaritas, nilai_polaritas, id):
    conn, cursor = get_conn_cursor()
    query = "Update testing set " + nama_polaritas + " = " + "'" + str(
        nilai_polaritas) + "'" + " where id_testing = " + str(id)
    cursor.execute(query)
    conn.commit()
    conn.close()


def updatePolaritasAwalTraining(nilai_polaritas, id):
    conn, cursor = get_conn_cursor()
    query = "Update training set polaritas = " + "'" + str(nilai_polaritas) + "'" + " where id_training = " + str(id)
    cursor.execute(query)
    conn.commit()
    conn.close()


def updatePolaritasAwalTesting(nilai_polaritas, id):
    conn, cursor = get_conn_cursor()
    query = "Update testing set polaritas_awal = " + "'" + str(nilai_polaritas) + "'" + " where id_testing = " + str(id)
    cursor.execute(query)
    conn.commit()
    conn.close()


def get_polaritas(nama_polaritas):
    conn, cursor = get_conn_cursor()
    query = "select " + nama_polaritas + " FROM testing"
    row = cursor.execute(query).fetchall()
    temp = []
    for x in row:
        for y in x:
            temp.append(y)
    return temp


def count_polaritas(nama_sentimen, nama_tabel, nama_kolom):
    conn, cursor = get_conn_cursor()
    query = "SELECT COUNT(" + nama_kolom + ") FROM " + nama_tabel \
            + " WHERE " + nama_kolom + " = '" + nama_sentimen + "' "
    row = cursor.execute(query).fetchone()[0]
    return row


def get_user_by_uname_pass(username, password):
    conn, cursor = get_conn_cursor()
    query = "SELECT user.id_user, user.nama, user.username, user.email, user.password FROM user " \
            "WHERE user.username = '" + username + "' AND user.password = '" + password + "'"
    row = cursor.execute(query).fetchone()
    if row is not None:
        data = {
            'id_user': row[0],
            'nama': row[1],
            'username': row[2],
            'email': row[3],
            'password': row[4],
        }
    else:
        data = {}
    return data


def get_user_by_id(id):
    conn, cursor = get_conn_cursor()
    query = "SELECT * FROM user WHERE user.id_user = '" + str(id) + "'"
    row = cursor.execute(query).fetchone()
    if row is not None:
        data = {
            'id_user': row[0],
            'nama': row[1],
            'username': row[2],
            'email': row[3],
            'password': row[4]
        }
    else:
        data = {}
    return data


def get_all_user():
    users = []
    try:
        conn, cursor = get_conn_cursor()
        sql_query = "SELECT * FROM user"
        rows = cursor.execute(sql_query).fetchall()
        for i in rows:
            user = {}
            user["id_user"] = i[0]
            user["nama"] = i[1]
            user["username"] = i[2]
            user["email"] = i[3]
            user["password"] = i[4]
            users.append(user)
    except:
        users = []
    return users


def update_user(id, nama, username, email, password):
    conn, cursor = get_conn_cursor()
    query = "UPDATE user SET nama = '" + nama + "', username = '" + username + "', email = '" + email + "', password = '" + password + "' WHERE id_user =" + str(
        id) + ""
    cursor.execute(query)
    conn.commit()
    data = get_user_by_id(id)
    return data


def add_user(nama, username, email, password):
    conn, cursor = get_conn_cursor()
    query = "INSERT INTO user(nama, username, email, password) VALUES('" + nama + "', '" + username + "', '" + email + "', '" + password + "')"
    cursor.execute(query)
    conn.commit()
    data = get_user_by_id(cursor.lastrowid)
    return data


def db_dell_user(id):
    print(id)
    try:
        conn, cursor = get_conn_cursor()
        query = "DELETE FROM user WHERE id_user = " + str(id) + ""
        cursor.execute(query)
        conn.commit()
        message = "success"
    except:
        conn.rollback()
        message = "failed"
    finally:
        conn.close()
    return message
