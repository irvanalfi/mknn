from flask import Flask, request, jsonify
import json
import sqlite3
import socket
from db import *
from process import *
import pandas as pd
import hashlib

my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return "server berjalan"


@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    data = get_user_by_uname_pass(username, password)

    return jsonify(data)


@app.route('/dashboard-data', methods=['GET', 'POST'])
def dashboard_data():
    training_positif = count_polaritas("positif", "training", "polaritas")
    training_negatif = count_polaritas("negatif", "training", "polaritas")
    training_netral = count_polaritas("netral", "training", "polaritas")
    total_training = training_positif + training_negatif + training_netral

    testing_positif = count_polaritas("positif", "testing", "polaritas_awal")
    testing_negatif = count_polaritas("negatif", "testing", "polaritas_awal")
    testing_netral = count_polaritas("netral", "testing", "polaritas_awal")
    total_testing = testing_positif + testing_negatif + testing_netral

    dataset_postif = training_positif + testing_positif
    dataset_negatif = training_negatif + testing_negatif
    dataset_netral = training_netral + testing_netral
    total_dataset = dataset_postif + dataset_negatif + dataset_netral

    k = ['K3', 'K5', 'K7', 'K9', 'K11', 'K13', 'K15', 'K17', 'K19', 'K21']

    data = {
        "jml_training_positif": training_positif,
        "jml_training_negatif": training_negatif,
        "jml_training_netral": training_netral,
        "total_training": total_training,
        "jml_testing_positif": testing_positif,
        "jml_testing_negatif": testing_negatif,
        "jml_testing_netral": testing_netral,
        "total_testing": total_testing,
        "jml_dataset_positif": dataset_postif,
        "jml_dataset_negatif": dataset_negatif,
        "jml_dataset_netral": dataset_netral,
        "total_dataset": total_dataset,
        "k": k,
        "jml_k": len(k)
    }
    return jsonify(data)


@app.route('/training-data', methods=['GET', 'POST'])
def training_data():
    return jsonify(db_get_all_training())


@app.route('/testing-data', methods=['GET', 'POST'])
def testing_data():
    return jsonify(db_get_all_testing())


@app.route('/import-training', methods=['GET', 'POST'])
def import_training():
    db_import_data_training("Upload/training.csv")


@app.route('/import-testing', methods=['GET', 'POST'])
def import_testing():
    db_import_data_testing("Upload/testing.csv")


@app.route('/tfidf-proses', methods=['GET', 'POST'])
def tfidf_proses():
    # proses tf-idf training dan testing
    tfidf(get_hasil())


@app.route('/halaman-pengujian', methods=['GET', 'POST'])
def halaman_proses_pengujian():
    preprocessing = db_get_all_testing()
    tfidf = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', sep=';')
    euclideanDTDT = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', sep=';')
    euclideanDTDS = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', sep=';')
    validitas = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/valargsmalleuclideandtdt.csv', sep=';')
    weightvoting = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/weightvoting.csv', sep=';')
    classterdekat = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/labelterdekat.csv', sep=';')
    data = {
        "preprocessing": jsonify(preprocessing),
        "tfidf": tfidf,
        "euclideanDTDT": euclideanDTDT,
        "euclideanDTDS": euclideanDTDS,
        "validitas": validitas,
        "weightvoting": weightvoting,
        "class_terdekat": classterdekat
    }
    return jsonify(data)


@app.route('/pengujian', methods=['GET', 'POST'])
def pengujian():
    # read csv hasil tf-idf
    df = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/tfidf.csv', sep=';')
    # membagi hasil tf idf dari total data set ke 600 training, 400 testing
    split1 = (df.shape[0] * 6) / 10
    # baca hasil split tf-idf data training dan testing
    df1 = pd.DataFrame(df.iloc[:int(split1)])
    df2 = pd.DataFrame(df.iloc[int(split1):])
    # proses jarak euclidean dan menyimpan data ke dalam file csv
    # jarak_dtdt = jarakeuclideanDTDT(df1)
    # jarak_dtds = jarakeuclideanDTDS(df1, df2, split1)
    # read csv jarak_dtdt dan jarak_dtds
    jarak_dtdt = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtdt.csv', sep=',')
    jarak_dtds = pd.read_csv('C:/Users/IRVAN/backendmknn/Upload/euclideandtds.csv', sep=',')
    # mengurutkan data jarak_dtdt
    small_dtdt = small(jarak_dtdt)
    # jumlah k yang digunakan
    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    for j in tqdm(k):
        idx = 1
        nama_polaritas = "polaritas_akhir_k" + str(j)
        k_dtdt = k_euclidean(small_dtdt, j)
        k_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/k11euclideandtdt.csv', index=False)
        lokasi_dtdt = lokasi(jarak_dtdt, j)
        lokasi_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/lokasiargsmalleuclideandtdt.csv', index=False)
        label_train = get_label("training")
        label_test = get_polaritas("polaritas_awal")
        pelabelan_dtdt = pelabelan(lokasi_dtdt, label_train)
        pelabelan_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/labelargsmalleuclideandtdt.csv', index=False)
        validitas_dtdt = validitas(pelabelan_dtdt, label_train)
        validitas_dtdt.to_csv('C:/Users/IRVAN/backendmknn/Upload/valargsmalleuclideandtdt.csv', index=False)

        lbl = []
        for i in tqdm(range(len(jarak_dtds.iloc[0])), leave=False):
            rank, label = ranking(validitas_dtdt, jarak_dtds[str(i)], j)
            lbl2 = []
            for i in label:
                lbl2.append(label_train[i])
            labelterdekat = pd.DataFrame(lbl2)
            labelterdekat.to_csv('C:/Users/IRVAN/backendmknn/Upload/labelterdekat.csv')
            lbl.append(most_frequent(lbl2))
            updateData(nama_polaritas, most_frequent(lbl2), idx)
            idx += 1


@app.route('/test-akurasi', methods=['GET', 'POST'])
def test_akurasi():
    cm_list = []
    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    for i in k:
        print("pengujian k" + str(i))
        cm_list.append(testaccuracy("polaritas_awal", "polaritas_akhir_k" + str(i)))
    return jsonify(cm_list)


if __name__ == "__main__":
    app.run(host=my_ip)
    # training_data()
    # login()
