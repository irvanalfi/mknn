from flask import Flask, request, jsonify
import json
import sqlite3
import socket
from db import count_polaritas

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




if __name__ == "__main__":
    app.run(host=my_ip)
