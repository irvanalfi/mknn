from flask import Flask, request, jsonify
import json
import sqlite3
import socket
from db import *
from process import *
import pandas as pd


my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return "server berjalan"


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

@app.route('/test-akurasi', methods=['GET', 'POST'])
def test_akurasi():
    cm_list = []
    key = ""
    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    for i in k:
        print("pengujian k" + str(i))
        cm_list.append(testaccuracy("polaritas_awal", "polaritas_akhir_k" + str(i)))
        key += testaccuracy("polaritas_awal", "polaritas_akhir_k" + str(i))+"\n"
    df = pd.DataFrame(cm_list).to_json(orient="index")
    print(key)
    return cm_list


if __name__ == "__main__":
    app.run(host=my_ip)
    # training_data()
